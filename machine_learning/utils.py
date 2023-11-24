from collections import Counter
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn


def tokenize(text):

    """
    Args:
        text: text to be tokenized

    Returns:
        tokens: List of tokens
    """

    return text.split()


def build_vocabulary(texts, max_vocab_size=None):

    """
    Args:
        texts: List of sentences separated by space
        max_vocab_size: Maximum vocabulary size

    Returns:
        word_to_index: Word to index mapping
    """

    # Count the word frequencies
    word_counts = Counter()
    for text in texts:
        word_counts.update(tokenize(text))

    # Limit vocabulary size if max_vocab_size is set
    if max_vocab_size is not None and max_vocab_size > len(word_counts):
        word_counts = word_counts.most_common(max_vocab_size)

    # Create a word to index mapping
    word_to_index = {word: i + 1 for i, (word, _) in enumerate(word_counts)}
    word_to_index["<UNK>"] = 0  # Add a special token for unknown words (index 0)
    return word_to_index


def text_to_indices(text, word_to_index):

    """
    Args:
        text: List of sentences separated by space
        word_to_index: Word to index mapping

    Returns:
        indices: List of word indices
    """

    tokens = tokenize(text)
    indices = [word_to_index.get(word, word_to_index["<UNK>"]) for word in tokens]
    return indices


def encode_labels(train, test):

    """
    Function to encode the labels, i.e. convert string labels to integers. Also, it handles the unknown label.
    Args:
        train:dataframe with training data, format [text, label]
        test: dataframe with testing data, format [text, label]

    Returns:
        le: LabelEncoder object
    """

    # Preprocess data

    le = LabelEncoder()
    train_labels = le.fit_transform(train["label"])
    test["label"] = test["label"].map(lambda s: '<unknown>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<unknown>')
    test_labels = le.transform(test["label"])
    # logic to handle the unknown label
    return le


def compute_class_weights(labels):
    """
    Function to compute the class weights
    Args:
        labels: list of labels

    Returns:
        class_weights: dictionary with class weights
    """
    class_weights = {}
    unique_labels = np.unique(labels)
    counts = Counter(labels)
    for label in unique_labels:
        class_weights[label] = len(labels) / (counts[label] * len(unique_labels))
    return class_weights


def convert_and_pad_sequences(index_sequences, device, padding_value=0):
    """
    Convert lists of indices into padded PyTorch tensors.

    Args:
        index_sequences (list): List of lists, where each inner list is a sequence of word indices.
        device (torch.device): The device to move the tensors to (e.g., 'cuda' or 'cpu').
        padding_value (int): The value used for padding shorter sequences.

    Returns:
        torch.Tensor: A padded tensor of sequences.
    """
    sequences = [torch.tensor(seq).to(device) for seq in index_sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    return padded_sequences
