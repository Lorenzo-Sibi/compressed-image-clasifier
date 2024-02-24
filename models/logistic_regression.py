from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
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

# Create and train the LogisticRegression model
model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
model.fit(X_train_scaled, y_train)



y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

# Here the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print("Validation Set Metrics:")
print("Accuracy:", val_accuracy)
print("Precision:", val_precision)
print("Recall:", val_recall)
print("F1-score:", val_f1)

# Evaluation time! :D
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print("\nTest Set Metrics:")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1-score:", test_f1)
