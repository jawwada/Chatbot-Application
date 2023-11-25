import pandas as pd
import torch
from machine_learning.learners.IntentTokenizer import IntentTokenizer
from torch.utils.data import DataLoader


# Load data
train_df = pd.read_csv('data/input/atis/train.tsv', sep='\t', header=None, names=["text", "label"])
test_df = pd.read_csv('data/input/atis/test.tsv', sep='\t', header=None, names=["text", "label"])

# Instantiate the tokenizer
tokenizer = IntentTokenizer(train_df)

# define constants and hyperparameters
vocab_size = tokenizer.max_vocab_size
output_dim = len(tokenizer.le.classes_)
batch_size = 32
num_epochs = 100

#   Define device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# tokenize and process train and test data and create dataLoaders

train_data = tokenizer.process_data(train_df, device=device)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
print("Number of training samples:", train_data.tensors[0].size())
print("Number of training batches:", len(train_loader))

test_data = tokenizer.process_data(test_df, device=device)
print("Number of test samples:", test_data.tensors[0].size())
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
print("Number of test batches:", len(test_loader))