from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

LABELS = {
    "": "",
    "": "",
}
DATA = 0

# Load your compressed image data (replace with your loading logic)
X = np.array([image_array for image_array in DATA])  # Assuming shape (800, height, width, channels)

# Reshape data (assuming channels as the last dimension)
X = X.reshape(X.shape[0], -1)  # Flattens to (800, features)

# Extract labels (replace with your labeling logic)
y = np.array([label for label in LABELS])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize pixel values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the SVM model
model = SVC(kernel='linear')  # Choose an appropriate kernel (e.g., 'rbf' for non-linear data)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate model performance (using metrics like accuracy, precision, recall, F1-score)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
