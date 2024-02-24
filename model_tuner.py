import optuna
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class HyperparameterOptimizer:
    def __init__(self, model_cls, data_loader):
        self.model_cls = model_cls
        self.data_loader = data_loader

    def objective(self, trial):
        # Define hyperparameters to optimize
        params = self._get_params(trial)

        # Load and preprocess data
        X_train, y_train = self.data_loader.load_data()
        X_train_flat = [x.flatten() for x in X_train]

        # Create and train model
        model = self.model_cls(**params)
        model.fit(X_train_flat, y_train)

        # Evaluate model performance
        accuracy = self._evaluate(model, X_train_flat, y_train)

        return accuracy

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    def _get_params(self, trial):
        raise NotImplementedError("Subclasses must implement _get_params method")

    def _evaluate(self, model, X, y):
        raise NotImplementedError("Subclasses must implement _evaluate method")


class LogisticRegressionOptimizer(HyperparameterOptimizer):
    def _get_params(self, trial):
        C = trial.suggest_loguniform('C', 1e-3, 1e3)
        solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        return {'C': C, 'solver': solver}

    def _evaluate(self, model, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred = model.predict(X_scaled)
        return accuracy_score(y, y_pred)

class SVMOptimizer(HyperparameterOptimizer):
    def _get_params(self, trial):
        C = trial.suggest_loguniform('C', 1e-3, 1e3)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        degree = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else None
        gamma = trial.suggest_loguniform('gamma', 1e-4, 1e1) if kernel in ['rbf', 'poly'] else 'scale'
        return {'C': C, 'kernel': kernel, 'degree': degree, 'gamma': gamma}

    def _evaluate(self, model, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred = model.predict(X_scaled)
        return accuracy_score(y, y_pred)
