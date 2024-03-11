import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.discriminant_analysis import StandardScaler
from models.logistic_regression import LogisticRegressionWrapper
from models.svm import SVCWrapper
from models.random_forest import RandomForestModel
from models.sci import SCI
from models.resnet import ResNetClassifier
from utils.data_loader import DatasetLoader

from utils.evaluation_metrics import ClassificationEvaluator

IMPLEMENTED_MODELS = ['logistic', 'svm', 'resnet', 'random_forest']
RANDOM_STATE = 2

def train(train_set, model, args):
    X_train, y_train = train_set.drop("label", axis=1), train_set['label']
    
    scaler = StandardScaler()
    
    if args.model in ("logistic", 'svm', 'random_forest'):
        X_train = [x.flatten() for x in X_train["data"]]
        print(f"Total elements (flattened sample): {len(X_train)}")
        X_train = scaler.fit_transform(X_train)
    
    else:
        normalized_data = []
        for sample in X_train['data']:
            
            # Normalize each channel separately
            normalized_sample = np.zeros_like(sample)
            for i in range(sample.shape[-1]):
                channel = sample[:, :, i]
                normalized_channel = scaler.fit_transform(channel)
                normalized_sample[:, :, i] = normalized_channel
            
            # Append the normalized sample to the list
            normalized_data.append(normalized_sample)

        # Convert the list of normalized samples back to a DataFrame
        X_train = normalized_data

    print("Data normalized.")
    
    model.fit(X_train, y_train, args)
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
    df = data_loader.create_dataset()
    # X, y = df.drop('label', axis=1), df['label']

    assert args.model in IMPLEMENTED_MODELS, "Model not implemented yet"

    if args.model == "logistic":
        model = LogisticRegressionWrapper(args)
    elif args.model == "svm":
        model = SVCWrapper(args)
    elif args.model == "random_forest":
        model = RandomForestModel(num_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_split=args.min_samples_split)
    elif args.model == "resnet":
        
        input_shape = df["data"][0].shape
        num_classes = len(df["label"].unique())
        
        model = ResNetClassifier(input_shape=input_shape, num_classes=num_classes, epochs=args.epochs, batch_size=args.batch_size)
        model.compile_model()
        
        labels_string = df["label"].unique().tolist()
        label_to_int = {label: i for i, label in enumerate(labels_string)} # map labels in a dictionary

        y = pd.DataFrame({"label": [label_to_int[label] for label in df["label"]]}) # convert labels to integers
        df["label"] = y["label"]
        
    elif args.model == "sci":
        
        input_shape = df["data"][0].shape
        num_classes = len(df["label"].unique())
        
        model = SCI(input_shape=input_shape, num_classes=num_classes)
        model.compile_model()
        
        labels_string = df["label"].unique().tolist()
        label_to_int = {label: i for i, label in enumerate(labels_string)} # map labels in a dictionary

        y = pd.DataFrame({"label": [label_to_int[label] for label in df["label"]]}) # convert labels to integers
        df["label"] = y["label"]
        
    df_training, df_testing = DatasetLoader.split_dataset(df, 0.2, shuffle=True, random_state=2)
    print(f"Total training samples: {len(df_training)}")
    print(f"Total testing samples: {len(df_testing)}")
    
    print(f"Each sample has as feature ('data') an array of shape: {df['data'][0].shape}")
    
    if operation == "train":
        print(df_training[:5])
        print("Training...")
    
        trained_model = train(df_training ,model, args)
        
        model_path = model_path / str(args.model)
        with model_path.open('wb') as fp:
            pickle.dump(trained_model, fp)
        
    elif operation == "test":
        # load the model from disk
        with model_path.open("rb") as fp:
            loaded_model = pickle.load(fp)
            
        print("Testing...")
        
        y_pred, evaluator = test(df_testing, loaded_model, args)
        evaluator.print_metrics(title=f"{str(args.model)}-metrics")
        if args.model == "resnet" or "sci":
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


# if args.model == "logistic":

#         X_flat = [x.flatten() for x in X["data"]]
#         print(f"Total elements (flattened sample): {len(X_flat)}")

#         train_logistic_regression(X_flat, y)

#     elif args.model == "svm":

#         X_flat = [x.flatten() for x in X["data"]]
#         print(f"Total elements (flattened sample): {len(X_flat)}")

#         train_svm(X_flat, y, tolerance=args.svm_tolerance, verbose=args.verbose)
    
#     elif args.model == "random_forest":
        
#         X_flat = [x.flatten() for x in X["data"]]
#         print(f"Total elements (flattened sample): {len(X_flat)}")
        
#         X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)
#         X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, shuffle=True, random_state=RANDOM_STATE)
        
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#         X_val_scaled = scaler.transform(X_val)
        
#         random_forest = RandomForestModel(num_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_split=args.min_samples_split)
#         random_forest.train(X_train_scaled, y_train)
        
#         y_val_pred = random_forest.predict(X_val_scaled)
#         y_test_pred = random_forest.predict(X_test_scaled)
        
#         print("\nRandom Forest Model:\n")
#         random_forest.print_params()
        
#         val_evaluator = ClassificationEvaluator(y_val, y_val_pred)
#         print("\nValidation Set metrics:")
#         val_evaluator.print_metrics(title="random-forest-validation")
        
#         test_evaluator = ClassificationEvaluator(y_test, y_test_pred)
#         print("\nTest Set metrics:")
#         test_evaluator.print_metrics(title="random-forest-test")


# elif args.model == "resnet":
        
#         input_shape = df["data"][0].shape
#         num_classes = len(y.unique())

#         labels_string = y.unique().tolist()
#         label_to_int = {label: i for i, label in enumerate(labels_string)} # map labels in a dictionary

#         y = np.array([label_to_int[label] for label in y]) # convert labels to integers
        
#         data = df["data"]
#         scaler = StandardScaler()
        
#         normalized_data = []
#         for sample in data:
            
#             # Normalize each channel separately
#             normalized_sample = np.zeros_like(sample)
#             for i in range(sample.shape[-1]):
#                 channel = sample[:, :, i]
#                 normalized_channel = scaler.fit_transform(channel)
#                 normalized_sample[:, :, i] = normalized_channel
            
#             # Append the normalized sample to the list
#             normalized_data.append(normalized_sample)

#         # Convert the list of normalized samples back to a DataFrame
#         # X_scaled = pd.DataFrame({'data': normalized_data})
#         X_scaled = normalized_data
        
#         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)
#         X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, shuffle=True, random_state=RANDOM_STATE)


#         resnet = ResNetClassifier(input_shape=input_shape, num_classes=num_classes)
#         resnet.compile_model()
        
#         # training...
#         history = resnet.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose=1)
        
#         y_test_pred = resnet.predict(X_test)
#         y_val_pred = resnet.predict(X_val)

#         val_evaluator = ClassificationEvaluator(y_val, y_val_pred)
#         print("ResNet\nValidation Set metrics:")
#         val_evaluator.print_metrics(title="validation")

#         test_evaluator = ClassificationEvaluator(y_test, y_test_pred)
#         print("ResNet\nTest Set metrics:")
#         test_evaluator.print_metrics(title="test")
        
#         # Plot training history
#         resnet.plot_training_history(history)