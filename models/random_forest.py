from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class RandomForestModel:
    def __init__(self, num_estimators=100, max_depth=2, min_samples_split=2, min_samples_leaf=1,
                 bootstrap=True, oob_score=True, n_jobs=1, random_state=2):
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

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()