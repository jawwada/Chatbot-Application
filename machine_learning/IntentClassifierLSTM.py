import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter

class IntentClassifierLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate):
        super(IntentClassifierLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # Batch normalization layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # Dropout layer
        dropped = self.dropout1(embedded)

        # LSTM layer
        lstm_out, (hidden, _) = self.lstm(dropped)
        # Take the output of the last time step
        hidden = hidden[-1]
        # Batch normalization
        normalized = self.batch_norm(hidden)

        # Fully connected layer
        out = self.fc(normalized)
        return out