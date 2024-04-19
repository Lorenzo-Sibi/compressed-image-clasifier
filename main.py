import argparse
from tabulate import tabulate
from pathlib import Path
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = info, 1 = warning, 2 = error, 3 = fatal
import tensorflow as tf
from models.logistic_regression import LogisticRegressionTF
from models.svm import SVMClassifier
# from models.random_forest import RandomForestModel
from models.sci import SCI
from models.resnet import ResNetClassifier
from models.inceptionv3 import InceptionV3Classifier
from utils.data_loader import DatasetLoader, DatasetWrapper

from utils.data_preprocessing import normalize_sample, flatten_feature, apply_mirror_padding
from utils.evaluation_metrics import ClassificationEvaluator

IMPLEMENTED_MODELS = ['logistic', 'svm', 'random_forest', 'sci', 'resnet', 'inceptionv3']
SEED = 2

tf.config.optimizer.set_jit(False)  # Disables XLA JIT compilation

tf.random.set_seed(SEED)
    
def train(ds_wrapper, args):
    model_name = args.model
    model_path = Path(args.model_path)
    
    # load the dataset
    training_set = ds_wrapper.dataset

    input_shape = ds_wrapper.max_shape
    num_classes = ds_wrapper.num_classes

    assert model_name in IMPLEMENTED_MODELS, "Model not implemented yet"

    if model_name == "logistic":
        model = LogisticRegressionTF(input_shape, num_classes, penalty=args.penalty, C=args.C, learning_rate=args.learning_rate, max_iter=args.epochs)
    elif model_name == "svm":
        model = SVMClassifier(input_shape, num_classes, C=args.C)
    elif model_name == "random_forest":
        exit(print("NOT IMPLEMENTED YET"))
        model = RandomForestModel(num_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_split=args.min_samples_split)
    elif model_name == "sci":
        model = SCI(input_shape, num_classes)
    elif model_name == "resnet":
        model = ResNetClassifier(input_shape, num_classes)
    elif model_name == 'inceptionv3':
        training_set = training_set = training_set.map(lambda image, label: apply_mirror_padding(image, label, target_size=(75, 75)))
        for sample in training_set.take(1):
            image, _ = sample
        input_shape = image.shape
        model = InceptionV3Classifier(input_shape, num_classes)


    model.compile()
    

    print(f"Training {model_name}")
    print(tabulate(list(ds_wrapper.label_map.items()), headers=["Labels", "Integers values"], tablefmt="pretty"))

    if args.model in ("logistic", 'svm', 'random_forest'):
        training_set = training_set.map(lambda x, y: flatten_feature(x, y))
    
    training_set = training_set.batch(32)
    
    history = model.fit(training_set, epochs=args.epochs)

    model_path = model_path / str(f"{model_name}.keras")

    model.save(model_path)

def test(ds_wrapper, args):
    model_path = Path(args.model_path)
    
    # load the model from disk
    loaded_model = tf.keras.models.load_model(model_path)
    model_name = model_path.stem
    
    # load the dataset
    test_set = ds_wrapper.dataset
    
    print(f"Testing {model_name}")
    print(tabulate(list(ds_wrapper.label_map.items()), headers=["Labels", "Integers values"], tablefmt="pretty"))

    if model_name in ("logistic", 'svm', 'random_forest'):
        test_set = test_set.map(lambda x, y: flatten_feature(x, y))
    
    test_set = test_set.batch(32)
        
    evaluator = ClassificationEvaluator()
    
    results = evaluator.evaluate(test_set, loaded_model)
    # evaluator.print_metrics()
    evaluator.plot_confusion_matrix(save_path=f"{model_name}_confusion_matrix.png",normalize=False)  # Use normalize=True for a normalized confusion matrix
    
    return results

def main(args):  # sourcery skip: extract-duplicate-method, extract-method
    
    command = args.command
    
    if command == "dataset":
        if args.operation == "split":
            DatasetLoader.split_dataset(args.split_parameter, Path(args.input_dir), Path(args.output_dir), args.shuffle)
        else:
            DatasetLoader.create_dataset(Path(args.input_dir), Path(args.output_dir), args.label_map)
        return
    
    dataset_path = Path(args.dataset_path)
    
    # Load data
    ds_wrapper = DatasetWrapper(dataset_path)
    
    ds_wrapper.dataset = ds_wrapper.dataset.map(lambda feature, label: normalize_sample(feature, label))
    print("Data normalized.")
    
    dataset_size = int(ds_wrapper.dataset.reduce(0, lambda x, _: x + 1).numpy())
    print(f"Dataset contains {dataset_size} samples")
    
    if command == "test":
        results = test(ds_wrapper, args)
        
        print(tabulate(list(results.items()), headers=["Valuation Metric", "Value"], tablefmt="pretty"))
        
    elif command == "train":
        train(ds_wrapper, args)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compressed Image Classifier - Using latent Spaces")
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Main command (split, train, test)')
    
    # Dataset command
    dataset_cmd_parser = subparsers.add_parser('dataset', help='Split dataset opration.')
    dataset_op = dataset_cmd_parser.add_subparsers(dest='operation', help='Operation to compute on dataset: create/split')
    
    # subparser for create operation
    create_operation = dataset_op.add_parser('create', help="Create the dataset starting from a directory, organized in subdirectories")
    create_operation.add_argument('input_dir', type=str, help='Directory of the dataset to be split')
    create_operation.add_argument('output_dir', type=str, help='Directory where to save splits')
    create_operation.add_argument("--label_map", default=None, type=str, help="")

    # subparser for split operation
    split_operation = dataset_op.add_parser('split', help="Split the dataset into two sets, training and test sets, using split_parameter as training set size ratio")
    split_operation.add_argument('split_parameter', type=float, default=0.8, help='Training set size ratio (default 0.8)')
    split_operation.add_argument('input_dir', type=str, help='Dataset to split directory')
    split_operation.add_argument('output_dir', type=str, help='Path to output splitted dataset (The training and test set will be saved here)')
    split_operation.add_argument('--shuffle', default=True, help='Shuflle the data before splitting into train and test set')



    # Test command
    test_parser = subparsers.add_parser('test', help='Test specified model')
    test_parser.add_argument('model_path', type=str, help='Path to the model for testing')
    test_parser.add_argument('dataset_path', type=str, help='Path to the model for testing')
    test_parser.add_argument('--output_path', type=str, default="./", help='Path to the model for testing')



    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    model_parser = train_parser.add_subparsers(dest='model', help='Select the model to train/load/evaluate (e.g., logistic, svm, random_forest, sci, resnet, inceptionv3)')
    train_parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    train_parser.add_argument('model_path', type=str, help='Path to save the trained model')

    # Subparser for logistic regression model
    logistic_parser = model_parser.add_parser('logistic', help='Logistic Regression Model')
    logistic_parser.add_argument("--penalty",  default="l2", type=str, choices=["l1", "l2", "elasticnet"],help="Regularization strength (C) for logistic regression")
    logistic_parser.add_argument("--learning_rate", default=1e-2, type=float, help="Learning rate for logistic regression")
    logistic_parser.add_argument("--C", default=1.0, type=float, help="Inverse of regularization strength; must be a positive float")

    # Subparser for support vector machine model
    svm_parser = model_parser.add_parser('svm', help='Support Vector Machine Model')
    svm_parser.add_argument("--C", default=1.0, type=float, help="Regularization parameter (C) for support vector machine")

    # Subparser for random forest model
    random_forest_parser = model_parser.add_parser('random_forest', help='Random Forest Model')
    random_forest_parser.add_argument("--n_estimators", default=100, type=int, help="Number of trees in the random forest")
    random_forest_parser.add_argument("--max_depth", default=None, type=int, help="Maximum depth of the tree")
    random_forest_parser.add_argument("--min_samples_split", default=2, type=int, help="Minimum number of samples required to split an internal node")

    # Source Camera Identification model
    sci_subparser = model_parser.add_parser("sci", help="Source Camera Identification Model")
    
    # Subparser for resnet model
    resnet_parser = model_parser.add_parser('resnet', help='ResNet Model')
    
    # InceptionV3 model
    inception_subparser = model_parser.add_parser("inceptionv3", help="InceptionV3 Model")
    
    for k, subparser in model_parser.choices.items():
        subparser.add_argument("--epochs", default=10, type=int, help="Number of epochs for training")
        subparser.add_argument("--batch_size", default=32, type=int, help="Batch size")
        subparser.add_argument("--verbose", default=1, type=int, choices=[0, 1, 2], help="Verbosity level")
    
    
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
    else:
        print(args)
        main(args)