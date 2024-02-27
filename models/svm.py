from tabulate import tabulate
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.evaluation_metrics import ClassificationEvaluator  # Import DataLoader class from the data loader module

def train_svm(X, y, test_size=0.2, tolerance=1e-2, verbose=False):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, shuffle=True, random_state=2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    model = SVC(tol=tolerance, verbose=verbose)
    model.fit(X_train_scaled, y_train)
    
    
    if model.kernel == 'poly':
        print(f"Kernel: {model.kernel}, Degree: {model.degree}")
    else:
        print(f"Kernel: {model.kernel}")
        
    print_params(model)
    
    y_test_pred = model.predict(X_test_scaled)
    print("Support Vector Machine\nValidation Set metrics:")
    y_val_pred = model.predict(X_val_scaled)

    val_evaluator = ClassificationEvaluator(y_val, y_val_pred)
    val_evaluator.print_metrics()
    
    test_evaluator = ClassificationEvaluator(y_test, y_test_pred)
    print("Support Vector Machine\nTest Set metrics:")
    test_evaluator.print_metrics()
    
def print_params(model):
    params = model.get_params()
    param_table = list(params.items())
    print(tabulate(param_table, headers=["Hyperparameter", "Value"], tablefmt="pretty"))