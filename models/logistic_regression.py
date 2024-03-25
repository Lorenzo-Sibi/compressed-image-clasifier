import tensorflow as tf
from tabulate import tabulate
from utils.evaluation_metrics import ClassificationEvaluator

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 2

class LogisticRegressionWrapper():
    def __init__(self, args):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, train_set, args):
        batch_size = train_set.reduce(0, lambda x, _: x + 1).numpy()
        self.model.fit(train_set, verbose=args.verbose)

    def predict(self, X_test):
        # Effettua le predizioni sul dataset di test
        y_pred = self.model.predict(X_test)
        return y_pred

    def print_params(self):
        # Stampa i parametri del modello
        params = self.model.get_config()
        param_table = list(params.items())
        print(tabulate(param_table, headers=["Hyperparameter", "Value"], tablefmt="pretty"))
    

def train_logistic_regression(X, y):  # sourcery skip: extract-duplicate-method
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, shuffle=True, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(tol=1e-3, verbose=True, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_test_pred = model.predict(X_test_scaled)
    y_val_pred = model.predict(X_val_scaled)

    print("\nLogistic Regression Model:\n")
    print_params(model)
    
    val_evaluator = ClassificationEvaluator(y_val, y_val_pred)
    print("\nValidation Set metrics:")
    val_evaluator.print_metrics(title="logistic-validation")
    
    test_evaluator = ClassificationEvaluator(y_test, y_test_pred)
    print("\nTest Set metrics:")
    test_evaluator.print_metrics(title="logistic-test")

def print_params(model):
    params = model.get_params()
    param_table = list(params.items())
    print(tabulate(param_table, headers=["Hyperparameter", "Value"], tablefmt="pretty"))