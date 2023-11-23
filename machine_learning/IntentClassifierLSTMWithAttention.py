import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer
    Architecture:
        1. Linear transformation for Q, K, V
        2. Scaled dot-product attention
        3. Softmax
        4. Weighted sum of values
    Purpose:
        This layer is used to capture the most important information from the input sequence. It multiplies the values
        with attention weights to get the final output.
    """
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output
class IntentClassifierLSTMWithAttention(nn.Module):
    """
    Intent Classifier LSTM with attention
    Architecture:
        1. Embedding layer
        2. Dropout layer
        3. LSTM layer
        4. Self-attention layer
        5. Batch normalization layer
        6. Fully connected layer
    Purpose:
        This model is used to classify the intent of a sentence. It uses self-attention to capture the most important
        information from the input sequence.
    """
    def __init__(self, vocab_size=None, embedding_dim=None, hidden_dim=None, output_dim=None, dropout_rate=None):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        super(IntentClassifierLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttentionLayer(hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    @classmethod
    def from_config_file(cls, config_file_path):
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)

        return cls(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            dropout_rate=config['dropout_rate']
        )
    def load(self, model_file_path):
        self.load_state_dict(torch.load(model_file_path),map_location=torch.device('cpu'))
        return
    def save_config_file(self, config_file_path):
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate
        }

        with open(config_file_path, 'w') as config_file:
            json.dump(config, config_file)
        return

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)
        # Dropout layer
        dropped = self.dropout(embedded)
        # LSTM layer (returns output and last hidden state)
        lstm_out, _ = self.lstm(dropped)

        # Apply attention
        attn_out = self.attention(lstm_out)
        # Take the output of the last time step
        final_output= attn_out[:, -1, :]
        # Batch normalization
        normalized = self.batch_norm(final_output)

        out = self.fc(normalized)
        return out