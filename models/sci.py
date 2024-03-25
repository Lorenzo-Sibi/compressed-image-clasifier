"""
This code is the implementation of the following paper:
'Deep learning for source camera identification on mobile devices'
by David Freire-Obregón, Fabio Narducci, Silvio Barra, Modesto Castrillón-Santana 

Code's Author: Lorenzo Sibi
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU


class SCI(Model):
    def __init__(self, input_shape=(32, 32, 3), num_classes=3):  # 32x32 input shape as default (change for latent spaces) same for num_classes
        super(SCI, self).__init__()
        
        self.optimizer = None
        self.loss = None
        self.metrics = None
        
        self.history = None

        self.convs = tf.keras.Sequential([
            Conv2D(64, (3, 3), strides=(2, 2), padding='valid', input_shape=input_shape),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(128, (3, 3), strides=(2, 2), padding='valid'),
            BatchNormalization(),
            LeakyReLU(),

            MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        ])

        self.features = tf.keras.Sequential([
            Dropout(0.5),
            Flatten(),
            Dense(256),
            BatchNormalization(),
            LeakyReLU(),

            Dropout(0.5),
            Dense(512),
            BatchNormalization(),
            LeakyReLU(),

            Dense(num_classes )
        ])
        
    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.model
        self.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
    def fit(self, train_set, epochs=10, batch_size=32, validation_split=0.2):
        self.history = self.fit(train_set, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return self.history

    def call(self, inputs):
        x = self.convs(inputs)
        x = self.features(x)
        return x
    
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
