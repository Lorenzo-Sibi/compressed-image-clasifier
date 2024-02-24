import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.data_loader import DatasetLoader

from tensorflow.keras.applications.resnet50 import ResNet50

IMPLEMENTED_MODELS = ['logistic', 'svm']

def train_logistic_regression(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", accuracy)

def train_svm(X_train, y_train, X_test, y_test):
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train SVM model
    model = SVC()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print("SVM Accuracy:", accuracy)

def main(main_folder, model):
    data_loader = DatasetLoader(main_folder)

    # Load data
    df = data_loader.create_dataset()
    
    X, y = df.drop('label', axis=1), df['label']
    
    print(f"Each entry has as feature an array of shape: {X['data'][0].shape}")
    
    if model == "logistic":
    
        # Each sample from a 3D-tensor becomes a 1D-tensor
        X_flat = [x.flatten() for x in X["data"]]

        print(f"Total entries (flattened): {len(X_flat)}")

        X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=2, shuffle=True)
    
        train_logistic_regression(X_train, y_train, X_test, y_test)
        
    if model == "svm":
        X_flat = [x.flatten() for x in X["data"]]

        print(f"Total entries (flattened): {len(X_flat)}")

        X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=2, shuffle=True)
        
        train_svm(X_train, y_train, X_test, y_test)
    
    if model == "resnet":
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Logistic Regression with Dataset Loader")
    parser.add_argument("main_folder", type=str, help="Path to the main folder containing subfolders with images")
    parser.add_argument("model", type=str, choices=IMPLEMENTED_MODELS, help="Type of model to use for the classification task: 'logistic_regression' or 'svm'")
    args = parser.parse_args()

    main(args.main_folder, args.model)
