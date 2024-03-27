from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class InceptionV3Classifier(Model):
    def __init__(self, input_shape, num_classes, **kwargs):
        """
        Initializes the InceptionV3 model to train from scratch on a dataset with a custom shape.
        Ensure that include_top=False, as the latent spaces used as features for classification are custom and have no shape (299, 299, 3).
        
        Parameters:
        - input_shape: tuple, the shape of the input images. Must be at least (75, 75, 3).
        - num_classes: int, the number of classes for the output layer.
        - include_top: bool, should be False when training from scratch for custom classes.
        """
        super(InceptionV3Classifier, self).__init__(**kwargs)
        self.inp_shape = input_shape
        self.num_classes = num_classes
        self.inception = InceptionV3(include_top=False, input_shape=input_shape, pooling='avg', weights=None)
        
        # custom layers at the end
        self.classifier = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.inception(inputs)
        return self.classifier(x)
    
    # def train_step(self, data):
    #     x, y = data
        
    #     # Convert labels to one-hot if they're in integer format (IMPORTANT!!!)
    #     y_one_hot = tf.one_hot(y, depth=self.num_classes)
        
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)
    #         loss = self.compiled_loss(y_one_hot, y_pred, regularization_losses=self.losses)
        
    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
        
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))# Update weights
        
    #     self.compiled_metrics.update_state(y_one_hot, y_pred)
        
    #     return {m.name: m.result() for m in self.metrics}

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], **kwargs):
        super(InceptionV3Classifier, self).compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(self, train_data, validation_data=None, epochs=10, batch_size=32, **kwargs):
        return super(InceptionV3Classifier, self).fit(train_data, validation_data=validation_data, epochs=epochs, batch_size=batch_size, **kwargs)
        
    def predict(self, X):
        predictions = self(X, training=False)
        return tf.argmax(predictions, axis=1)

    def predict_proba(self, X):
        return self.predict(X)
    
    def get_config(self):
        config = super(InceptionV3Classifier, self).get_config()
        config.update({
            'input_shape': self.inp_shape,
            'num_classes': self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Not implemented yet
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