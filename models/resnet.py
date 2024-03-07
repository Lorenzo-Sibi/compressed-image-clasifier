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
    def __init__(self, input_shape=(32, 32, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
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

    def fit(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose=1):
        # Convert NumPy arrays to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train)
        y_train = tf.convert_to_tensor(y_train)
        X_val = tf.convert_to_tensor(X_val)
        y_val = tf.convert_to_tensor(y_val)
        
        early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=verbose)

        return history

    def predict(self, X):
        """
        Predict class labels for input samples.
        
        Parameters:
        - X: Input data, numpy array of shape (num_samples, height, width, channels).
        
        Returns:
        - y_pred: Predicted class labels (integers), numpy array of shape (num_samples,).
        """
        y_pred = np.argmax(self.model.predict(X), axis=-1)
        return y_pred
    

    def plot_training_history(self, history, save_path=Path("./")):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)