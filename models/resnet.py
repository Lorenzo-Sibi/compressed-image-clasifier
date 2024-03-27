import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class ResNetClassifier(Model):
    def __init__(self, input_shape, num_classes, **kwargs):
        super(ResNetClassifier, self).__init__(**kwargs)
        self.inp_shape = input_shape
        self.num_classes = num_classes

        self.model = self.build_model()
        self.history = None
        
    def build_model(self):
        base_model = ResNet50(include_top=False, weights=None, input_shape=self.inp_shape)

        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)

        for layer in base_model.layers:
            layer.trainable = False

        return model

    def call(self, inputs, training=False):
        x = self.model(inputs, training=training)
        return x

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], **kwargs):
        super(ResNetClassifier, self).compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(self, train_data, validation_data=None, epochs=10, batch_size=32, **kwargs):
        
        early_stopping = EarlyStopping(monitor="accuracy", patience=1, restore_best_weights=True)
        
        history = super(ResNetClassifier, self).fit(train_data, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], **kwargs)
        self.history = history
        return history
    
    def predict(self, X):
        predictions = self(X, training=False)
        return tf.argmax(predictions, axis=1)


    def evaluate(self, test_set):
        model = self.model
        results = model.evaluate(test_set)
        return results

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