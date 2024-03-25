import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class ResNetClassifier:
    def __init__(self, input_shape=(32, 32, 3), num_classes=3, epochs=32, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.build_model()
        self.history = None
        
    def build_model(self):
        base_model = ResNet50(include_top=False, weights=None, input_shape=self.input_shape)

        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)

        for layer in base_model.layers:
            layer.trainable = False

        return model

    def compile_model(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self, train_set, args):
        verbose = args.verbose
        
        early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

        history = self.model.fit(train_set, epochs=self.epochs, callbacks=[early_stopping], verbose=verbose)
        self.history = history

    def predict(self, X):
        """
        Predict class labels for input samples.
        
        Parameters:
        - X: Input data, numpy array of shape (num_samples, height, width, channels).
        
        Returns:
        - y_pred: Predicted class labels (integers), numpy array of shape (num_samples,).
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        prediction = self.model.predict(X)
        y_pred = np.argmax(prediction, axis=-1)
        return y_pred
    

    def plot_training_history(self, save_path="./"):
        history = self.history
        plt.figure(figsize=(12, 6))

        plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
        plt.plot(history.history['loss'], label='Train Loss', color='orange')
        
        plt.title('Training Accuracy and Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        save_path = Path(save_path, "history_plot.png")
        plt.savefig(save_path)