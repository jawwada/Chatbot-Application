"""
Main example for training and evaluating a model.
This script trains a model on the ATIS dataset and evaluates it on the test set. It Also saves the model and tokenizer.
The model is a LSTM model with attention. The tokenizer is a singleton class that tokenizes text and encodes labels.
The structure of the code is as follows:
1. Load and preprocess the data
2. Create a tokenizer
3. Create DataLoaders
4. Define loss function and optimizer
5. Train the model
6. Evaluate the model
7. Save the model and tokenizer
8. Load the model and tokenizer
9. Serve the model
10. Predict on a query

The structure of the script abstracts the training and evaluation process in the same way that a machine learning library
like HuggingFace's Transformers does. This allows you to focus on the model architecture and hyperparameters without
worrying about the training and evaluation process. You can also easily switch between different models and datasets.
"""


import pandas as pd
import torch
import torch.nn as nn
from machine_learning.IntentTokenizer import IntentTokenizer
from machine_learning.model_utils import train, evaluate, predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "best_ICELSTMAmodel"


model_serve = torch.load(f"models/{model_name}.pth").to(device)
tokenizer = tokenizer = IntentTokenizer.load_state(IntentTokenizer,f"models/{model_name}_tokenizer.pickle", f"models/IntentClassifierLSTMWithAttention_le.pickle")
test_df = pd.read_csv('data/atis/test.tsv', sep='\t', header=None, names=["text", "label"])

# compute metrics like accuracy, precision, recall, F1 score
# Accuracy
y_test = test_df['label'].values
y_pred = predict(model_serve, test_df, tokenizer, device)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Precision, Recall, and F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Classification report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

# Binarize the labels for multi-class plots
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
y_pred_binarized = label_binarize(y_pred, classes=np.unique(y_test))

n_classes = y_test_binarized.shape[1]

# Plot Precision-Recall curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
    plt.plot(recall, precision, lw=2, label=f'class {tokenizer.le.classes_[i]}')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
plt.title("Precision vs. Recall curve")
plt.show()


from sklearn.metrics import roc_curve, auc

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {tokenizer.le.classes_[i]} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
