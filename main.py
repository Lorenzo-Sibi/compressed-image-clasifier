import argparse
from tabulate import tabulate
from pathlib import Path
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = info, 1 = warning, 2 = error, 3 = fatal
import tensorflow as tf
from models.logistic_regression import LogisticRegressionTF
from models.svm import SVMClassifier
from models.random_forest import RandomForestModel
from models.sci import SCI
from models.resnet import ResNetClassifier
from models.inceptionv3 import InceptionV3Classifier
from utils.data_loader import DatasetLoader

from utils.data_preprocessing import apply_mirror_padding
from utils.evaluation_metrics import ClassificationEvaluator

IMPLEMENTED_MODELS = ['logistic', 'svm', 'random_forest', 'sci', 'resnet', 'inceptionv3']
SEED = 2

tf.random.set_seed(SEED)

def train(train_set, model, args):
    
    def normalize_sample(feature, label):
        mean = tf.math.reduce_mean(feature)
        std = tf.math.reduce_std(feature)
        feature = (feature - mean) / std
        return feature, label

    train_set = train_set.map(lambda feature, label: normalize_sample(feature, label))
    print("Train data normalized.")
    
    if args.model in ("logistic", 'svm', 'random_forest'):
        def flatten_feature(feature, label):
            flattened_feature = tf.reshape(feature, [-1])
            return flattened_feature, label
        train_set = train_set.map(lambda x, y: flatten_feature(x, y))
    
    train_set = train_set.batch(32)
    
    model.fit(train_set, epochs=args.epochs)
    return model

def test(test_set, model, args):
    def normalize_sample(feature, label):
        mean = tf.math.reduce_mean(feature)
        std = tf.math.reduce_std(feature)
        feature = (feature - mean) / std
        return feature, label

    test_set = test_set.map(lambda feature, label: normalize_sample(feature, label))
    print("Test data normalized.")
    
    if args.model in ("logistic", 'svm', 'random_forest'):
        def flatten_feature(feature, label):
            flattened_feature = tf.reshape(feature, [-1])
            return flattened_feature, label
        test_set = test_set.map(lambda x, y: flatten_feature(x, y))
    
    test_set = test_set.batch(32)
        
    evaluator = ClassificationEvaluator()
    
    results = evaluator.evaluate(test_set, model)
    
    # evaluator.print_metrics()
    print(tabulate(list(results.items()), headers=["Hyperparameter", "Value"], tablefmt="pretty"))
    evaluator.plot_confusion_matrix(save_path=f"{args.model}_confusion_matrix.png",normalize=False)  # Use normalize=True for a normalized confusion matrix

    return results

def evaluate():
    pass

def main(args):  # sourcery skip: extract-duplicate-method, extract-method
    
    operation = args.operation
    dataset_path = args.dataset_path
    model_path = Path(args.model_path)
    
    data_loader = DatasetLoader(dataset_path)
    
    input_shape = data_loader.max_shape
    num_classes = len(data_loader.labels)
    label_map = data_loader.label_map
    
    # Load data
    df = data_loader.load_dataset()

    assert args.model in IMPLEMENTED_MODELS, "Model not implemented yet"
    
    if args.model == "logistic":
        model = LogisticRegressionTF(input_shape, num_classes, penalty=args.penalty, C=args.C, learning_rate=args.learning_rate, max_iter=args.epochs)
    elif args.model == "svm":
        model = SVMClassifier(input_shape, num_classes, C=args.C)
    elif args.model == "random_forest":
        model = RandomForestModel(num_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_split=args.min_samples_split)
    elif args.model == "sci":
        model = SCI(input_shape, num_classes)
    elif args.model == "resnet":
        model = ResNetClassifier(input_shape, num_classes)
    elif args.model == 'inceptionv3':
        df = df = df.map(lambda image, label: apply_mirror_padding(image, label, target_size=(75, 75)))
        for features, _ in df.take(1):
            input_shape = features.shape
        model = InceptionV3Classifier(input_shape, num_classes)
        
    
    model.compile()
        
        
    # Calculate the dataset cardinality
    dataset_size = df.reduce(0, lambda x, _: x + 1).numpy()
    train_size = int(0.8 * dataset_size)  # 80% dei dati per il training set
    test_size = int(dataset_size - train_size)
    
    # Shuffle=False to prevent the process running out of memory during shuffle for each epoch 
    df = df.shuffle(dataset_size, seed=SEED, reshuffle_each_iteration=False)
    
    train_dataset = df.take(train_size)
    test_dataset = df.skip(train_size)
    print(tabulate(list(label_map.items()), headers=["Labels", "Integers values"], tablefmt="pretty"))
    print(f"Total training samples: {train_size}")
    print(f"Total testing samples: {test_size}")
    
    if operation == "train":
        print(f"Training {args.model}")
    
        trained_model = train(train_dataset, model, args)
        
        model_path = model_path / str(args.model + ".keras")
        model.save(model_path)
        # with model_path.open('wb') as fp:
        #     pickle.dump(trained_model, fp)
        
    elif operation == "test":
        # load the model from disk
        loaded_model = tf.keras.models.load_model(model_path)
        # with model_path.open("rb") as fp:
        #     loaded_model = pickle.load(fp)
            
        print(f"Testing {args.model}")
        
        results = test(test_dataset, loaded_model, args)
        
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
    logistic_parser.add_argument("--penalty",  default="l2", type=str, choices=["l1", "l2", "elasticnet"],help="Regularization strength (C) for logistic regression")
    logistic_parser.add_argument("--learning_rate", default=1e-2, type=float, help="Learning rate for logistic regression")
    logistic_parser.add_argument("--epochs", default=20, type=int, help="")
    logistic_parser.add_argument("--C", default=1.0, type=float, help="Inverse of regularization strength; must be a positive float")
    logistic_parser.add_argument("--verbose", default=1, type=int, choices=[0, 1, 2, 3], help="Verbosity level for logistic regression")
    logistic_parser.add_argument("--plot-title", type=str, help="")


    # Subparser for support vector machine model
    svm_parser = subparsers.add_parser('svm', help='Support Vector Machine Model')
    svm_parser.add_argument("--C", default=1.0, type=float, help="Regularization parameter (C) for support vector machine")
    svm_parser.add_argument("--epochs", default=20, type=int, help="")
    svm_parser.add_argument("--verbose", default=True, type=bool, help="Verbosity for support vector machine")
    svm_parser.add_argument("--plot-title", type=str, help="")


    # Subparser for random forest model
    random_forest_parser = subparsers.add_parser('random_forest', help='Random Forest Model')
    random_forest_parser.add_argument("--n_estimators", default=100, type=int, help="Number of trees in the random forest")
    random_forest_parser.add_argument("--max_depth", default=None, type=int, help="Maximum depth of the tree")
    random_forest_parser.add_argument("--epochs", default=10, type=int, help="")
    random_forest_parser.add_argument("--min_samples_split", default=2, type=int, help="Minimum number of samples required to split an internal node")
    random_forest_parser.add_argument("--plot-title", type=str, help="")


    # Source Camera Identification model
    sci_subparser = subparsers.add_parser("sci", help="Source Camera Identification Model")
    sci_subparser.add_argument("--epochs", default=10, type=int, help="")
    sci_subparser.add_argument("--batch_size", default=32, type=int, help="Number of elements in the batch")
    sci_subparser.add_argument("--verbose", default=1, type=int, help="Minimum number of samples required to split an internal node")
    
    
    # Subparser for resnet model
    resnet_parser = subparsers.add_parser('resnet', help='ResNet Model')
    resnet_parser.add_argument("--epochs", default=10, type=int, help="")
    resnet_parser.add_argument("--batch_size", default=32, type=int, help="Number of elements in the batch")
    resnet_parser.add_argument("--verbose", default=1, type=int, help="Minimum number of samples required to split an internal node")
    
    
    # InceptionV3 model
    inception_subparser = subparsers.add_parser("inceptionv3", help="InceptionV3 Model")
    inception_subparser.add_argument("--epochs", default=10, type=int, help="")
    inception_subparser.add_argument("--batch_size", default=32, type=int, help="Number of elements in the batch")
    inception_subparser.add_argument("--verbose", default=1, type=int, help="Minimum number of samples required to split an internal node")
    
    args = parser.parse_args()
    print(args)
    main(args)