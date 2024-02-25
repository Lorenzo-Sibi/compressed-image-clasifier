import argparse
from models.logistic_regression import train_logistic_regression
from models.svm import train_svm
from utils.data_loader import DatasetLoader
from tensorflow.keras.applications.resnet50 import ResNet50

IMPLEMENTED_MODELS = ['logistic', 'svm', 'resnet']

def main(args):
    data_loader = DatasetLoader(args.main_folder)

    # Load data
    df = data_loader.create_dataset()
    
    X, y = df.drop('label', axis=1), df['label']
    
    print(f"Each entry has as feature an array of shape: {X['data'][0].shape}")
    
    if args.model == "logistic":
    
        X_flat = [x.flatten() for x in X["data"]]
        print(f"Total entries (flattened): {len(X_flat)}")
        
        train_logistic_regression(X_flat, y)
        
    elif args.model == "svm":
        
        X_flat = [x.flatten() for x in X["data"]]
        print(f"Total entries (flattened): {len(X_flat)}")
        
        train_svm(X_flat, y)
    
    elif args.model == "resnet":
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Logistic Regression with Dataset Loader")
    parser.add_argument("main_folder", type=str, help="Path to the main folder containing subfolders with images")
    
    subparsers = parser.add_subparsers(dest='model', help='Select the model to evaluate')

    # Subparser for logistic regression model
    logistic_parser = subparsers.add_parser('logistic_regression', help='Logistic Regression Model')
    logistic_parser.add_argument("--regularization", type=float, help="Regularization strength (C) for logistic regression")
    logistic_parser.add_argument("--solver", type=str, choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"], help="Solver for logistic regression")
    logistic_parser.add_argument("--max_iterations", type=int, help="Maximum number of iterations for logistic regression")
    logistic_parser.add_argument("--tolerance", type=float, help="Tolerance for logistic regression convergence")
    logistic_parser.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3], help="Verbosity level for logistic regression")

    # Subparser for support vector machine model
    svm_parser = subparsers.add_parser('support_vector_machine', help='Support Vector Machine Model')
    svm_parser.add_argument("--kernel", type=str, choices=["linear", "poly", "rbf", "sigmoid"], help="Kernel type for support vector machine")
    svm_parser.add_argument("--svm_regularization", type=float, help="Regularization parameter (C) for support vector machine")
    svm_parser.add_argument("--svm_tolerance", type=float, help="Tolerance for support vector machine convergence")
    svm_parser.add_argument("--svm_verbosity", type=int, choices=[0, 1, 2, 3], help="Verbosity level for support vector machine")

    parser.add_argument("model", type=str, choices=IMPLEMENTED_MODELS, help="Type of model to use for the classification task: 'logistic' or 'svm'")
    args = parser.parse_args()

    main(args)
