import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import numpy as np
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")
# Instantiate the model


train_file = 'data/atis/train.tsv'
test_file = 'data/atis/test.tsv'

# Load data
train = pd.read_csv(train_file, sep='\t', header=None)
test = pd.read_csv(test_file, sep='\t', header=None)


# Preprocess data
le = LabelEncoder()
train_labels = le.fit_transform(train[1])
test[1] = test[1].map(lambda s: '<unknown>' if s not in le.classes_ else s)
le.classes_ = np.append(le.classes_, '<unknown>')
test_labels = le.transform(test[1])
# logic to handle the unknown label 
#Print a list of classes in the data
print(le.classes_)
print(len(le.classes_))

output_dim = len(le.classes_) # Number of output classes


# Tokenize the text
# This function splits text into words. Feel free to customize it as needed.
def tokenize(text):
    return text.split()


word_counts = Counter()
for text in train[0]:
    word_counts.update(tokenize(text))

# Create a word to index mapping
word_to_index = {word: i + 1 for i, (word, _) in enumerate(word_counts.items())} # starting index from 1, 0 is usually for padding

'''Add a special token for unknown words
This line adds a special entry in the dictionary for the token "<UNK>", which stands for "unknown."
It assigns the index 0 to "<UNK>". This is used to represent words that are not found in the training vocabulary, 
especially when encountering new words in the test set or in real-world application.
'''
word_to_index["<UNK>"] = 0

vocab_size = len(word_to_index) # Number of unique words in the vocabulary
# You can also set a maximum vocabulary size if needed
# For example, to keep only the top 10,000 words:
# max_vocab_size = 10000
# word_to_index = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common(max_vocab_size))}

# Test the vocabulary
print("Vocabulary size:", len(word_to_index))
print("Index of the word 'flight':", word_to_index.get('flight', word_to_index['<UNK>']))



def text_to_indices(text, word_to_index):
    # Tokenize the text
    tokens = tokenize(text)
    # Convert tokens to indices
    indices = [word_to_index.get(word, word_to_index["<UNK>"]) for word in tokens]
    return indices

# Apply this function to the training and testing datasets
train_indices = [text_to_indices(text, word_to_index) for text in train[0]]
test_indices = [text_to_indices(text, word_to_index) for text in test[0]]


print(train_indices[42])
print(test_indices[131])


loss_function = nn.CrossEntropyLoss()
