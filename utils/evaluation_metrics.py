import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)

class ClassificationEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred, average='macro')

    def recall(self):
        return recall_score(self.y_true, self.y_pred, average='macro')

    def f1_score(self):
        return f1_score(self.y_true, self.y_pred, average='macro')

    def roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        return fpr, tpr

    def auc_score(self):
        return auc(self.roc_curve()[0], self.roc_curve()[1])

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def classification_report(self):
        return classification_report(self.y_true, self.y_pred)

    def matthews_corrcoef(self):
        return matthews_corrcoef(self.y_true, self.y_pred)

    def plot_roc_curve(self, output_path=Path("./")):
        fpr, tpr = self.roc_curve()
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = {:.2f})'.format(self.auc_score()))
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)

    def plot_confusion_matrix(self, output_path=Path("./"), labels=None):
        cm = self.confusion_matrix()
        if labels is None:
            labels = np.unique(self.y_true)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.grid(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        plt.savefig(output_path)

    def print_metrics(self, title:str, output_path="./"):
        output_path = Path(output_path, title)
        print("\tAccuracy:", self.accuracy())
        print("\tPrecision:", self.precision())
        print("\tRecall:", self.recall())
        print("\tF1 Score:", self.f1_score())
        print("\tMatthews Correlation Coefficient:", self.matthews_corrcoef())
        print("\tClassification Report:")
        print(self.classification_report())
        print("\tConfusion Matrix:")
        print(self.confusion_matrix())
        print("\n")
        self.plot_confusion_matrix(output_path=Path(output_path, "confusion-matrix-", title))
        self.plot_roc_curve(output_path=Path(output_path, "roc-curve-", title))