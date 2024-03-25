from tabulate import tabulate
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class RandomForestModel:
    def __init__(self, num_estimators=100, max_depth=2, min_samples_split=2, random_state=2):
        self.num_estimators = num_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=num_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            random_state=random_state,
        )

    def fit(self, train_set, args):
        self.model.fit(train_set)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()
    
    def print_params(self):
        params = self.get_params()
        param_table = list(params.items())
        print(tabulate(param_table, headers=["Hyperparameter", "Value"], tablefmt="pretty"))