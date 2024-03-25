import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.discriminant_analysis import StandardScaler
from models.logistic_regression import LogisticRegressionWrapper
from models.svm import SVCWrapper
from models.random_forest import RandomForestModel
from models.sci import SCI
from models.resnet import ResNetClassifier
from utils.data_loader import DatasetLoader

from utils.evaluation_metrics import ClassificationEvaluator

IMPLEMENTED_MODELS = ['logistic', 'svm', 'resnet', 'random_forest', 'sci']
RANDOM_STATE = 2

tf.random.set_seed(2)

def train(train_set, model, args):
    scaler = StandardScaler()
    
    def normalize_sample(feature, label):
        mean = tf.math.reduce_mean(feature)
        std = tf.math.reduce_std(feature)
        feature = (feature - mean) / std
        return feature, label

    train_set = train_set.map(lambda feature, label: normalize_sample(feature, label))
    print("Data normalized.")
    
    if args.model in ("logistic", 'svm', 'random_forest'):
        def flatten_feature(feature, label):
            flattened_feature = tf.reshape(feature, [-1])
            return flattened_feature, label
        train_set = train_set.map(lambda x, y: flatten_feature(x, y))
    
    # for sample in train_set.take(1):
    #     print(sample)
    
    model.model.summary()
    train_set = train_set.batch(32)
    model.fit(train_set, args)
    return model

def test(test_set, model, args):
    X_test, y_test = test_set.drop("label", axis=1), test_set['label']
    
    scaler = StandardScaler()
    
    if args.model in ("logistic", 'svm', 'random_forest'):
        X_test = [x.flatten() for x in X_test["data"]]
        X_test = scaler.fit_transform(X_test)
    
    else:
        normalized_data = []
        for sample in X_test['data']:
            
            # Normalize each channel separately
            normalized_sample = np.zeros_like(sample)
            for i in range(sample.shape[-1]):
                channel = sample[:, :, i]
                normalized_channel = scaler.fit_transform(channel)
                normalized_sample[:, :, i] = normalized_channel
            
            # Append the normalized sample to the list
            normalized_data.append(normalized_sample)

        # Convert the list of normalized samples back to a DataFrame
        X_test = normalized_data
        
    print("Data normalized.")
    
    y_test_pred = model.predict(X_test)
    evaluator = ClassificationEvaluator(y_test, y_test_pred)

    return (y_test_pred, evaluator)

def evaluate():
    pass

def main(args):  # sourcery skip: extract-duplicate-method, extract-method
    
    operation = args.operation
    dataset_path = args.dataset_path
    model_path = Path(args.model_path)
    
    data_loader = DatasetLoader(dataset_path)
    
    # Load data
    if(args.model in ("resnet", "sci")):
        label_type = "integer"
    else:
        label_type = "string"
        
    df = data_loader.load_dataset(label_type=label_type)

    assert args.model in IMPLEMENTED_MODELS, "Model not implemented yet"

    if args.model == "logistic":
        model = LogisticRegressionWrapper(args)
    elif args.model == "svm":
        model = SVCWrapper(args)
    elif args.model == "random_forest":
        model = RandomForestModel(num_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_split=args.min_samples_split)
    elif args.model == "resnet":
        
        input_shape = data_loader.max_shape
        unique_labels = data_loader.labels
        label_map = {lab: i for i, lab in enumerate(unique_labels)}
        num_classes = len(unique_labels)
        
        model = ResNetClassifier(input_shape=input_shape, num_classes=num_classes, epochs=args.epochs, batch_size=args.batch_size)
        model.compile_model()
        
    elif args.model == "sci":
        
        input_shape = data_loader.max_shape
        unique_labels = data_loader.labels
        label_map = {lab: i for i, lab in enumerate(unique_labels)}
        num_classes = len(unique_labels)
        
        model = SCI(input_shape=input_shape, num_classes=num_classes)
        model.compile_model()
        
        def map_labels(y):
            numeric_label = label_map[y.numpy().decode('utf-8')]
            numeric_label = tf.constant(numeric_label, shape=[1], dtype=tf.int32)
            return numeric_label
        
        df = df.map(lambda x, y: (x, tf.py_function(map_labels, [y], tf.int32)))
        
    # Calculate the dataset cardinality
    dataset_size = df.reduce(0, lambda x, _: x + 1).numpy()
    train_size = int(0.8 * dataset_size)  # 80% dei dati per il training set
    test_size = int(dataset_size - train_size)
    
    df = df.shuffle(dataset_size, seed=2)
    
    train_dataset = df.take(train_size)
    test_dataset = df.skip(train_size)
    print(f"Total training samples: {train_size}")
    print(f"Total testing samples: {test_size}")
    
    if operation == "train":
        print(f"Training {args.model}")
    
        trained_model = train(train_dataset, model,args)
        
        model_path = model_path / str(args.model)
        with model_path.open('wb') as fp:
            pickle.dump(trained_model, fp)
        
    elif operation == "test":
        # load the model from disk
        with model_path.open("rb") as fp:
            loaded_model = pickle.load(fp)
            
        print(f"Testing {args.model}")
        
        y_pred, evaluator = test(test_dataset, loaded_model, args)
        evaluator.print_metrics(title=f"{str(args.model)}-metrics")
        if args.model == "resnet" or args.model == "sci":
            loaded_model.plot_training_history()
        
    elif operation == "predict":
        pass
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compressed Image Classifier - Using latent Spaces")
    
    parser.add_argument("operation", help="Operation to compute: train, test, evaluation, predict")
    subparsers = parser.add_subparsers(dest='model', help='Select the model to train/load/evaluate')
    parser.add_argument("dataset_path", type=str, help="Path to the main folder containing the dataset (each subfolder shoud represents the relative class)")
    parser.add_argument("model_path", default = "./", help="Path where to save/load the trained model.")

    # Subparser for logistic regression model
    logistic_parser = subparsers.add_parser('logistic', help='Logistic Regression Model')
    logistic_parser.add_argument("--regularization",  default="l2", type=str, choices=["l1", "l2", "elasticnet"],help="Regularization strength (C) for logistic regression")
    logistic_parser.add_argument("--solver", default="lbfgs", type=str, choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"], help="Solver for logistic regression")
    logistic_parser.add_argument("--max_iterations", default=100, type=int, help="Maximum number of iterations for logistic regression")
    logistic_parser.add_argument("--tolerance", default=1e-2, type=float, help="Tolerance for logistic regression convergence")
    logistic_parser.add_argument("--verbose", default=1, type=int, choices=[0, 1, 2, 3], help="Verbosity level for logistic regression")
    logistic_parser.add_argument("--plot-title", type=str, help="")

    # Subparser for support vector machine model
    svm_parser = subparsers.add_parser('svm', help='Support Vector Machine Model')
    svm_parser.add_argument("--kernel", default="rbf", type=str, choices=["linear", "poly", "rbf", "sigmoid"], help="Kernel type for support vector machine")
    svm_parser.add_argument("--svm_regularization", default=1.0, type=float, help="Regularization parameter (C) for support vector machine")
    svm_parser.add_argument("--svm_tolerance", default=1e-3, type=float, help="Tolerance for support vector machine convergence")
    svm_parser.add_argument("--verbose", default=True, type=bool, help="Verbosity for support vector machine")
    svm_parser.add_argument("--plot-title", type=str, help="")

    # Subparser for random forest model
    random_forest_parser = subparsers.add_parser('random_forest', help='Random Forest Model')
    random_forest_parser.add_argument("--n_estimators", default=100, type=int, help="Number of trees in the random forest")
    random_forest_parser.add_argument("--max_depth", default=None, type=int, help="Maximum depth of the tree")
    random_forest_parser.add_argument("--min_samples_split", default=2, type=int, help="Minimum number of samples required to split an internal node")
    random_forest_parser.add_argument("--plot-title", type=str, help="")

    # Subparser for resnet model
    resnet_parser = subparsers.add_parser('resnet', help='ResNet Model')
    resnet_parser.add_argument("--epochs", default=32, type=int, help="")
    resnet_parser.add_argument("--batch_size", default=32, type=int, help="Number of elements in the batch")
    resnet_parser.add_argument("--verbose", default=1, type=int, help="Minimum number of samples required to split an internal node")
    
    # Source Camera Identification model
    sci_subparser = subparsers.add_parser("sci", help="Source Camera Identification Model")
    
    args = parser.parse_args()
    print(args)
    main(args)