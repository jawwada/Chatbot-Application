import pandas as pd
import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import random
import re
from machine_learning.IntentTokenizer import IntentTokenizer
from machine_learning.IntentClassifierLSTMWithAttention import IntentClassifierLSTMWithAttention
from machine_learning.model_utils import train, evaluate, predict
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

train_df = pd.read_csv('data/atis/train.tsv', sep='\t', header=None, names=["text", "label"])
test_df = pd.read_csv('data/atis/test.tsv', sep='\t', header=None, names=["text", "label"])
tokenizer = IntentTokenizer(train_df)
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

#model = IntentClassifierLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate).to(device)
model = IntentClassifierLSTMWithAttention(vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train(model, optimizer, loss_function, train_loader, num_epochs)
evaluate(model, loss_function, test_loader)
model_name = "IntentClassifierLSTMWithAttention"
torch.save(model.to(torch.device("cpu")),f"models/{model_name}.pth")
tokenizer.save_state(f"models/{model_name}_tokenizer.pickle")
tokenizer = IntentTokenizer.load_state(f"models/{model_name}_tokenizer.pickle")
model_serve = torch.load(f"models/{model_name}.pth").to(device)
max_query_length = 50
query_text = "what airlines off from love field between 6 and 10 am on june sixth"
query = pd.DataFrame({"text": [query_text]})
prediction = predict(model_serve, query,tokenizer,device)
print(f"Predicted label: {prediction}")