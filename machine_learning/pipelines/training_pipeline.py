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
from machine_learning.learners.IntentTokenizer import IntentTokenizer
from machine_learning.learners.IntentClassifierLSTMWithAttention import IntentClassifierLSTMWithAttention
from machine_learning.learners.model_utils import train, evaluate, predict
from machine_learning.pipelines.data_loaders import train_loader, test_loader, tokenizer, device

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
torch.save(model.to(torch.device("cpu")),f"data/models/{model_name}.pth")
tokenizer.save_state(f"data/models/{model_name}_tokenizer.pickle", f"data/models/{model_name}_le.pickle")

# Serve the model
model_serve = torch.load(f"data/models/{model_name}.pth").to(device)


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