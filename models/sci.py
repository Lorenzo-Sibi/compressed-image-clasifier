"""
This code is the implementation of the following paper:
'Deep learning for source camera identification on mobile devices'
by David Freire-Obregón, Fabio Narducci, Silvio Barra, Modesto Castrillón-Santana 

Code's Author: Lorenzo Sibi
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU


class SCI(Model):
    def __init__(self, input_shape=(32, 32, 3), num_classes=3):  # 32x32 input shape as default (change for latent spaces) same for num_classes
        super(SCI, self).__init__()

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

    def call(self, inputs):
        x = self.convs(inputs)
        x = self.features(x)
        return x
