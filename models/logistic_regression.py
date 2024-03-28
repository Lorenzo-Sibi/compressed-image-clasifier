import tensorflow as tf
from tabulate import tabulate

RANDOM_STATE = 2

class LogisticRegressionWrapper():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def fit(self, train_set, args):
        batch_size = train_set.reduce(0, lambda x, _: x + 1).numpy()
        self.model.fit(train_set, verbose=args.verbose, epochs=32)

    def predict(self, X_test):
        # Effettua le predizioni sul dataset di test
        y_pred = self.model.predict(X_test)
        return y_pred

    

class LogisticRegressionTF(tf.keras.Model):
    def __init__(self, inp_shape, num_classes, max_iter=10, penalty='l2', C=1.0, learning_rate=1e-2, **kwargs):
        """
        Initializes the Logistic Regression model as per TensorFlow best practices.

        Parameters are similar to the previous version, but adapted for TensorFlow's OO approach.
        """
        super(LogisticRegressionTF, self).__init__(**kwargs)
        self.inp_shape = inp_shape
        self.num_classes = num_classes
        self.penalty = penalty
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.flatten = tf.keras.layers.Flatten(input_shape=self.inp_shape)
        self.dense = self._get_dense_layer()

    def _get_dense_layer(self):
        """Creates the dense layer with appropriate regularization."""
        if self.penalty == 'l2':
            regularizer = tf.keras.regularizers.l2(1./self.C)
        elif self.penalty == 'l1':
            regularizer = tf.keras.regularizers.l1(1./self.C)
        elif self.penalty == 'elasticnet':
            # ElasticNet regularization is not directly supported in Keras layers, this is my implementation
            l1_ratio = 0.5  # This should be a parameter if you're using elasticnet recularization
            l1 = l1_ratio / self.C
            l2 = (1 - l1_ratio) / self.C
            regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
        else:
            regularizer = None

        return tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizer)

    def call(self, inputs):
        x = self.flatten(inputs)
        return self.dense(x)

    def compile(self, **kwargs):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        super(LogisticRegressionTF, self).compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], **kwargs)

    def fit(self, *args, **kwargs):
        kwargs['epochs'] = kwargs.get('epochs', self.max_iter)
        return super(LogisticRegressionTF, self).fit(*args, **kwargs)

    def predict(self, X, batch_size=32):
        predictions = super(LogisticRegressionTF, self).predict(X, batch_size=batch_size)
        return tf.argmax(predictions, axis=-1).numpy()
    
    def get_config(self):
        config = super(LogisticRegressionTF, self).get_config()
        config.update({
            'inp_shape': self.inp_shape,
            'num_classes': self.num_classes,
            'penalty': self.penalty,
            'C': self.C,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def print_params(self):
        param_table = param_table = list(
            ("penalty", self.penalty),
            ("C", self.C), 
            ("learning rate", self.learning_rate), 
            ("Epochs", self.max_iter)
        )
        print(tabulate(param_table, headers=["Hyperparameter", "Value"], tablefmt="pretty"))