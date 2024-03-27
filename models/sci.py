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
from tensorflow.keras.callbacks import EarlyStopping

class SCI(Model):
    def __init__(self, input_shape, num_classes):  # 32x32 input shape as default (change for latent spaces) same for num_classes
        super(SCI, self).__init__()
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
        
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        super(SCI, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def call(self, inputs):
        x = self.convs(inputs)
        x = self.features(x)
        return x

    def fit(self, train_data, validation_data=None, epochs=10, batch_size=32, **kwargs):
        early_stopping = EarlyStopping(monitor="accuracy", patience=1, restore_best_weights=True)
        
        history = super(SCI, self).fit(train_data, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], **kwargs)
        self.history = history
        return history
    
    def predict(self, x, return_probabilities=False):
        predictions = super(SCI, self).predict(x)
        return predictions if return_probabilities else tf.argmax(predictions, axis=1)

    def evaluate(self, x, y, batch_size=None, verbose=1, sample_weight=None, return_dict=False):
        # Qui potresti inserire logica personalizzata se necessario
        return super(SCI, self).evaluate(x, y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight, return_dict=return_dict)


class SCIWrapper():
    def __init__(self, input_shape=(32, 32, 3), num_classes=3) -> None:
        self.model = SCI(input_shape=input_shape, num_classes=num_classes)
        self.model.compile_model()
        self.history = None
        
    def fit(self, train_set, args):
        
        
        
        self.history = self.model.fit(train_set, epochs=args.epochs, verbose=args.verbose, callbacks=[early_stopping])
        return self.history
        
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