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
from machine_learning.IntentClassifierLSTMWithAttention import IntentClassifierLSTMWithAttention
from machine_learning.model_utils import train, evaluate, predict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the data
train_df = pd.read_csv('data/atis/train.tsv', sep='\t', header=None, names=["text", "label"])
test_df = pd.read_csv('data/atis/test.tsv', sep='\t', header=None, names=["text", "label"])
tokenizer = IntentTokenizer(train_df)
tokenizer.save_state("models/IntentClassifierLSTMWithAttention_tokenizer_example.pickle", "models/IntentClassifierLSTMWithAttention_le_example.pickle")

# Example usage
train_data = tokenizer.process_data(train_df,device=device)
test_data = tokenizer.process_data(test_df,device=device)
print("Number of training samples:", train_data.tensors[0].size())
print("Number of test samples:", test_data.tensors[0].size())

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
print("Number of training batches:", len(train_loader))
print("Number of test batches:", len(test_loader))

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
learning_rate = 0.01              # If you set this too high, it might explode. If too low, it might not learn
weight_decay = 1e-7               # Regularization strength
dropout_rate = 0.3                 # Dropout rate
embedding_dim = 64                # Size of each embedding vector
hidden_dim = 128                 # Number of features in the hidden state of the LSTM
batch_size = 32                  # Number of samples in each batch
output_dim = len(IntentTokenizer.le.classes_)  # Number of classes
num_epochs = 5            # Number of times to go through the entire dataset
vocab_size = tokenizer.max_vocab_size + 1  # The size of the vocabulary
# Create a string that summarizes these parameters
params_str = f"Vocab Size: {vocab_size}\n" \
             f"Embedding Dim: {embedding_dim}\n" \
             f"Hidden Dim: {hidden_dim}\n" \
             f"Output Dim: {output_dim}\n" \
             f"Dropout Rate: {dropout_rate}\n" \
             f"learning Rate: {learning_rate}\n" \
             f"epochs: {num_epochs}"
print(params_str)

# Pick the model and train it. Evaluate the model on the test set.
# choose model to train, uncomment the model you want to train and comment the other one
# IntentClassifierLSTM is a simple LSTM model. IntentClassifierLSTMWithAttention is a LSTM model with attention.
# The latter performs better.
# Difference in Accuracy between the two models is about 3%

# model = IntentClassifierLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate).to(device)
model = IntentClassifierLSTMWithAttention(vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train(model, optimizer, loss_function, train_loader, num_epochs)
evaluate(model, loss_function, test_loader)

# Save the model and tokenizer for serving.
model_name = "IntentClassifierLSTMWithAttention_main_example"
torch.save(model.to(torch.device("cpu")),f"models/{model_name}.pth")
tokenizer.save_state(f"models/{model_name}_tokenizer.pickle", f"models/{model_name}_le.pickle")

# Serve the model
device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_serve = torch.load(f"models/{model_name}.pth").to(device)


# Predict on a query
max_query_length = 50
query_text = "what airlines off from love field between 6 and 10 am on june sixth"
query = pd.DataFrame({"text": [query_text]})
prediction = predict(model_serve, query,tokenizer,device)
print(f"Predicted label: {prediction}")

max_query_length = 50
query_text = "I want to book a hotel near miami beach"
query = pd.DataFrame({"text": [query_text]})
prediction = predict(model_serve, query,tokenizer,device)
print(f"Predicted label: {prediction}")