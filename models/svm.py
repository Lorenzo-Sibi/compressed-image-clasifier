import tensorflow as tf
from tabulate import tabulate
    
class SVMClassifier(tf.keras.Model):
    def __init__(self, inp_shape, num_classes, C=1.0, epochs=20, **kwargs):
        super(SVMClassifier, self).__init__(**kwargs)
        self.inp_shape = inp_shape
        self.num_classes = num_classes
        self.C = C
        self.epochs = epochs
        # Layer setup
        self.flatten = tf.keras.layers.Flatten(input_shape=self.inp_shape)
        self.dense = tf.keras.layers.Dense(num_classes, use_bias=False)  # Linear kernel
        self.hinge_loss_layer = HingeLossLayer(num_classes, C=self.C)

    def call(self, inputs, training=False, labels=None):
        x = self.flatten(inputs)
        x = self.dense(x)
        if training:
            if labels is None:
                raise ValueError("Labels must not be None for training")
            return self.hinge_loss_layer(x, labels)
        return x

    def train_step(self, data):
        x, y = data  # The dataset yields (features, labels)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, labels=y)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars)) # Update weights
        self.compiled_metrics.update_state(y, y_pred) # Update the metrics.
        
        return {m.name: m.result() for m in self.metrics}
    
    def compile(self, optimizer="adam", **kwargs):
        super(SVMClassifier, self).compile(optimizer=optimizer, metrics=['accuracy'] ,**kwargs)

    def fit(self, dataset, **kwargs):
        kwargs['epochs'] = kwargs.get('epochs', self.epochs)
        return super(SVMClassifier, self).fit(dataset, **kwargs)

    def predict(self, X):
        predictions = self(X, training=False)
        return tf.argmax(predictions, axis=1)
    
    def get_config(self):
        config = super(SVMClassifier, self).get_config()
        config.update({
            'inp_shape': self.inp_shape,
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'C': self.C,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)