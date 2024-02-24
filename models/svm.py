from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.data_loader import DatasetLoader  # Import DataLoader class from the data loader module

# Initialize DataLoader instances for each dataset
training_set = DatasetLoader("folder1")
validation_set = DatasetLoader("folder2")
test_set = DatasetLoader("folder3")

# Load data from training, validation, and test sets
train_df = training_set.create_dataframe()
val_df = validation_set.create_dataframe()
test_df = test_set.create_dataframe()

# Extract features (X) and labels (y) from DataFrames
X_train, y_train = train_df['data'].tolist(), train_df['label'].tolist()
X_val, y_val = val_df['data'].tolist(), val_df['label'].tolist()
X_test, y_test = test_df['data'].tolist(), test_df['label'].tolist()

# Flatten the feature arrays
X_train_flat = [x.flatten() for x in X_train]
X_val_flat = [x.flatten() for x in X_val]
X_test_flat = [x.flatten() for x in X_test]

# Standardize pixel values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_val_scaled = scaler.transform(X_val_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Define kernel strategies and combinations
kernel_strategies = ['linear', 'rbf', 'poly']
degrees = [2, 3]  # Degrees for polynomial kernel

# Perform SVM classification with different kernel strategies and combinations
for kernel in kernel_strategies:
    if kernel == 'poly':
        for degree in degrees:
            model = SVC(kernel=kernel, degree=degree)
            model.fit(X_train_scaled, y_train)
            y_val_pred = model.predict(X_val_scaled)
            y_test_pred = model.predict(X_test_scaled)
            print(f"Kernel: {kernel}, Degree: {degree}")
            print("Validation Set Metrics:")
            print("Accuracy:", accuracy_score(y_val, y_val_pred))
            print("Precision:", precision_score(y_val, y_val_pred, average='weighted'))
            print("Recall:", recall_score(y_val, y_val_pred, average='weighted'))
            print("F1-score:", f1_score(y_val, y_val_pred, average='weighted'))
            print("\nTest Set Metrics:")
            print("Accuracy:", accuracy_score(y_test, y_test_pred))
            print("Precision:", precision_score(y_test, y_test_pred, average='weighted'))
            print("Recall:", recall_score(y_test, y_test_pred, average='weighted'))
            print("F1-score:", f1_score(y_test, y_test_pred, average='weighted'))
            print("--------------------------------------------")
    else:
        model = SVC(kernel=kernel)
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)
        print(f"Kernel: {kernel}")
        print("Validation Set Metrics:")
        print("Accuracy:", accuracy_score(y_val, y_val_pred))
        print("Precision:", precision_score(y_val, y_val_pred, average='weighted'))
        print("Recall:", recall_score(y_val, y_val_pred, average='weighted'))
        print("F1-score:", f1_score(y_val, y_val_pred, average='weighted'))
        print("\nTest Set Metrics:")
        print("Accuracy:", accuracy_score(y_test, y_test_pred))
        print("Precision:", precision_score(y_test, y_test_pred, average='weighted'))
        print("Recall:", recall_score(y_test, y_test_pred, average='weighted'))
        print("F1-score:", f1_score(y_test, y_test_pred, average='weighted'))
        print("--------------------------------------------")
