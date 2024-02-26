import argparse
import numpy as np

from sklearn.discriminant_analysis import StandardScaler
from models.logistic_regression import train_logistic_regression
from sklearn.model_selection import train_test_split
from models.svm import train_svm
from models.random_forest import RandomForestModel
from models.resnet import ResNetClassifier
from utils.data_loader import DatasetLoader
from tensorflow.keras.applications.resnet50 import ResNet50

from utils.evaluation_metrics import ClassificationEvaluator

IMPLEMENTED_MODELS = ['logistic', 'svm', 'resnet']
RANDOM_STATE = 2

def main(args):  # sourcery skip: extract-duplicate-method, extract-method
    data_loader = DatasetLoader(args.main_folder)

    # Load data
    df = data_loader.create_dataset()

    X, y = df.drop('label', axis=1), df['label']

    print(f"Each sample has as feature ('data') an array of shape: {X['data'][0].shape}")

    if args.model == "logistic":

        X_flat = [x.flatten() for x in X["data"]]
        print(f"Total elements (flattened sample): {len(X_flat)}")

        train_logistic_regression(X_flat, y)

    elif args.model == "svm":

        X_flat = [x.flatten() for x in X["data"]]
        print(f"Total elements (flattened sample): {len(X_flat)}")

        train_svm(X_flat, y, tolerance=args.svm_tolerance, verbose=args.verbose)
    
    elif args.model == "random_forest":
        
        X_flat = [x.flatten() for x in X["data"]]
        print(f"Total elements (flattened sample): {len(X_flat)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, shuffle=True, random_state=RANDOM_STATE)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)
        
        random_forest = RandomForestModel(num_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_split=args.min_samples_split)
        random_forest.train(X_train_scaled, y_train)
        
        y_val_pred = random_forest.predict(X_val_scaled)
        y_test_pred = random_forest.predict(X_test_scaled)
        
        val_evaluator = ClassificationEvaluator(y_val, y_val_pred)
        print(f"Random Forest\nModel's parameters:\n{random_forest.get_param()}\nValidation Set metrics:")
        val_evaluator.print_metrics()
        
        test_evaluator = ClassificationEvaluator(y_test, y_test_pred)
        print("\nTest Set metrics:")
        test_evaluator.print_metrics()

    elif args.model == "resnet":
        input_shape = X["data"][0].shape
        num_classes = len(y.unique())

        labels_string = y.unique().tolist()
        label_to_int = {label: i for i, label in enumerate(labels_string)} # map labels in a dictionary

        y = np.array([label_to_int[label] for label in y]) # convert labels to integers
        
        data = X["data"]
        scaler = StandardScaler()
        
        normalized_data = []
        for sample in data:
            
            # Normalize each channel separately
            normalized_sample = np.zeros_like(sample)
            for i in range(sample.shape[-1]):
                channel = sample[:, :, i]
                normalized_channel = scaler.fit_transform(channel)
                normalized_sample[:, :, i] = normalized_channel
            
            # Append the normalized sample to the list
            normalized_data.append(normalized_sample)

        # Convert the list of normalized samples back to a DataFrame
        # X_scaled = pd.DataFrame({'data': normalized_data})
        X_scaled = normalized_data
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, shuffle=True, random_state=RANDOM_STATE)

        resnet = ResNetClassifier(input_shape=input_shape, num_classes=num_classes)
        resnet.compile_model()
        
        # training...
        history = resnet.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose=1)
        
        y_test_pred = resnet.predict(X_test)
        y_val_pred = resnet.predict(X_val)

        val_evaluator = ClassificationEvaluator(y_val, y_val_pred)
        print("ResNet\nValidation Set metrics:")
        val_evaluator.print_metrics()

        test_evaluator = ClassificationEvaluator(y_test, y_test_pred)
        print("ResNet\nTest Set metrics:")
        test_evaluator.print_metrics()
        
        # Plot training history
        resnet.plot_training_history(history)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compressed Image Classifier - Using latent Spaces")
    
    parser.add_argument("main_folder", type=str, help="Path to the main folder containing subfolders with images")
    
    subparsers = parser.add_subparsers(dest='model', help='Select the model to evaluate')

    # Subparser for logistic regression model
    logistic_parser = subparsers.add_parser('logistic', help='Logistic Regression Model')
    logistic_parser.add_argument("--regularization", default="l2", type=str, choices=["l1", "l2", "elasticnet"],help="Regularization strength (C) for logistic regression")
    logistic_parser.add_argument("--solver",default="lbfgs", type=str, choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"], help="Solver for logistic regression")
    logistic_parser.add_argument("--max_iterations",default=100, type=int, help="Maximum number of iterations for logistic regression")
    logistic_parser.add_argument("--tolerance",default=1e-2, type=float, help="Tolerance for logistic regression convergence")
    logistic_parser.add_argument("--verbose",default=1, type=int, choices=[0, 1, 2, 3], help="Verbosity level for logistic regression")

    # Subparser for support vector machine model
    svm_parser = subparsers.add_parser('svm', help='Support Vector Machine Model')
    svm_parser.add_argument("--kernel", type=str, default="rbf" ,choices=["linear", "poly", "rbf", "sigmoid"], help="Kernel type for support vector machine")
    svm_parser.add_argument("--svm_regularization", default=1.0, type=float, help="Regularization parameter (C) for support vector machine")
    svm_parser.add_argument("--svm_tolerance", default=1e-3, type=float, help="Tolerance for support vector machine convergence")
    svm_parser.add_argument("--verbose", default=True, type=bool, help="Verbosity for support vector machine")

    # Subparser for random forest model
    random_forest_parser = subparsers.add_parser('random_forest', help='Random Forest Model')
    random_forest_parser.add_argument("--n_estimators", default=100, type=int, help="Number of trees in the random forest")
    random_forest_parser.add_argument("--max_depth", default=None, type=int, help="Maximum depth of the tree")
    random_forest_parser.add_argument("--min_samples_split", default=2, type=int, help="Minimum number of samples required to split an internal node")

    # Subparser for resnet model
    resnet_parser = subparsers.add_parser('resnet', help='ResNet Model')
    
    args = parser.parse_args()

    main(args)
