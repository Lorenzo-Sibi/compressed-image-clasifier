import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # For a nicer confusion matrix visualization

class ClassificationEvaluator:
    def __init__(self):
        # Initialize metrics
        self.metrics = {
            'accuracy': tf.keras.metrics.Accuracy(),
            'precision': tf.keras.metrics.Precision(),
            'recall': tf.keras.metrics.Recall(),
        }
        self.f1_score = None
        self.all_labels = []
        self.all_predictions = []

    def update_state(self, y_true, y_pred):
        # Update state for each metric and save all labels and predictions
        self.all_labels.extend(y_true.numpy())
        self.all_predictions.extend(y_pred)
        
        for metric in self.metrics.values():
            metric.update_state(y_true, y_pred)

        # Calculate F1 Score
        precision = self.metrics['precision'].result().numpy()
        recall = self.metrics['recall'].result().numpy()
        self.f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # TODO: fix evaluate method, cerca di eliminare se possibile il ciclo for, inoltre print_metrics() restituisce tuti valori 0.0

    def evaluate(self, test_dataset, model):
        self.set_model(model)
        for batch in test_dataset:
            feature, labels = batch
            predictions = model.predict(feature)
            self.update_state(labels, predictions)

        results = {name: metric.result().numpy() for name, metric in self.metrics.items()}
        results['f1_score'] = self.f1_score

        # Reset metrics after evaluation
        for metric in self.metrics.values():
            metric.reset_states()

        return results

    def set_model(self, model):
        self.model = model

    def print_metrics(self):
        print("Evaluation Metrics:")
        print(f"Accuracy: {self.metrics['accuracy'].result().numpy()}")
        print(f"Precision: {self.metrics['precision'].result().numpy()}")
        print(f"Recall: {self.metrics['recall'].result().numpy()}")
        print(f"F1 Score: {self.f1_score}")

    def plot_confusion_matrix(self, normalize=False):
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d")
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()