# compressed-image-clasifier

TODO: Review the Hyperparameter Optimizer

# Hyperparameter Optimizer API

## HyperparameterOptimizer Class

### Constructor:

- **Parameters:**
  - `model_cls`: The class of the machine learning model to optimize (e.g., `LogisticRegression`, `SVC`).
  - `data_loader`: An instance of the `DatasetLoader` class or any class that provides similar functionality for loading data.

### Methods:

1. **`optimize(n_trials: int)`**
   - Starts the hyperparameter optimization process using Optuna.
   - **Parameters:**
     - `n_trials`: The number of optimization trials to perform.
   - **Returns:** None

2. **`objective(trial: optuna.trial.Trial) -> float`**
   - The objective function to be optimized.
   - **Parameters:**
     - `trial`: An Optuna `Trial` object representing a single optimization trial.
   - **Returns:** The value of the objective function (e.g., accuracy) for the given hyperparameters.

3. **`_get_params(trial: optuna.trial.Trial) -> dict`**
   - Abstract method to be implemented by subclasses.
   - Returns the hyperparameters to be optimized as a dictionary based on the provided Optuna `Trial`.
   - **Parameters:**
     - `trial`: An Optuna `Trial` object representing a single optimization trial.
   - **Returns:** A dictionary containing the hyperparameters to optimize.

4. **`_evaluate(model, X, y) -> float`**
   - Abstract method to be implemented by subclasses.
   - Evaluates the performance of the model using the given data.
   - **Parameters:**
     - `model`: The machine learning model instance.
     - `X`: Input features.
     - `y`: Target labels.
   - **Returns:** The evaluation metric (e.g., accuracy) of the model on the given data.

## Subclasses (e.g., `LogisticRegressionOptimizer`, `SVMOptimizer`):

- Subclasses inherit the constructor and methods of the `HyperparameterOptimizer` class.
- They implement the `_get_params` and `_evaluate` methods specific to their respective models (e.g., logistic regression, SVM).
- Users interact with these subclasses to perform hyperparameter optimization for specific types of machine learning models.

This API design allows users to easily perform hyperparameter optimization for different machine learning models by simply providing the appropriate model class and data loader. The abstract methods `_get_params` and `_evaluate` ensure flexibility and customization for different types of models and evaluation metrics.
