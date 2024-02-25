import argparse
from models.logistic_regression import train_logistic_regression
from models.svm import train_svm
from utils.data_loader import DatasetLoader
from tensorflow.keras.applications.resnet50 import ResNet50

IMPLEMENTED_MODELS = ['logistic', 'svm', 'resnet']

def main(main_folder, model):
    data_loader = DatasetLoader(main_folder)

    # Load data
    df = data_loader.create_dataset()
    
    X, y = df.drop('label', axis=1), df['label']
    
    print(f"Each entry has as feature an array of shape: {X['data'][0].shape}")
    
    if model == "logistic":
    
        X_flat = [x.flatten() for x in X["data"]]
        print(f"Total entries (flattened): {len(X_flat)}")
        
        train_logistic_regression(X_flat, y)
        
    if model == "svm":
        
        X_flat = [x.flatten() for x in X["data"]]
        print(f"Total entries (flattened): {len(X_flat)}")
        
        train_svm(X_flat, y)
    
    if model == "resnet":
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Logistic Regression with Dataset Loader")
    parser.add_argument("main_folder", type=str, help="Path to the main folder containing subfolders with images")
    parser.add_argument("model", type=str, choices=IMPLEMENTED_MODELS, help="Type of model to use for the classification task: 'logistic' or 'svm'")
    args = parser.parse_args()

    main(args.main_folder, args.model)
