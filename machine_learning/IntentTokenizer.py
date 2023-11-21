import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os, pickle
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from utils import build_vocabulary, text_to_indices, convert_and_pad_sequences


class IntentTokenizer:
    """Singleton class to tokenize text and encode labels.
    Attributes:
        word2idx: Word to index mapping
        word_counts: Word frequencies
        le: Label encoder
        max_vocab_size: Maximum vocabulary size
    Purpose:
        This class is used to tokenize text and encode labels. It is a singleton class.
        The word2idx attribute is a dictionary that maps words to indices.
        The word_counts attribute is a Counter object that stores the word frequencies.
        The le attribute is a LabelEncoder object that encodes the labels.
        The max_vocab_size attribute is the maximum vocabulary size. It is set to 1000 by default.
        It is language independent and can be used for any text classification task.
    """

    _instance = None
    word2idx = None
    word_counts = None
    le = None
    max_vocab_size = None

    def __new__(cls, *args, **kwargs):
        # Singleton design pattern
        if not cls._instance:
            cls._instance = super(IntentTokenizer, cls).__new__(cls)
        return cls._instance

    def __init__(self, df, max_vocab_size=1000):
        # Initialize the tokenizer
        if not IntentTokenizer.word2idx:
            # Initialize the attributes
            # Counter object to store the word frequencies
            IntentTokenizer.word_counts = Counter()
            print("inside IntentTokenizer")
            # Build the vocabulary
            IntentTokenizer.word2idx = build_vocabulary(df["text"], max_vocab_size)
            print(f"Vocabulary Size: {len(IntentTokenizer.word2idx)}")
            # Maximum vocabulary size
            IntentTokenizer.max_vocab_size = len(IntentTokenizer.word2idx)
            # Label encoder
            IntentTokenizer.le=LabelEncoder()
            self.encode_labels(df)


    def process_data(self, df, device=torch.device("cpu")):
        """Process the data and return a TensorDataset."""

        index_sequences = [text_to_indices(text, IntentTokenizer.word2idx) for text in df["text"]]
        padded = convert_and_pad_sequences(index_sequences, device=device)
        labels = self.encode_labels(df)
        TCTensor = TensorDataset(padded.to(device), torch.tensor(labels).to(device))
        return TCTensor

    def encode_labels(self,df):
        """Encode the labels."""
        # Label encoder, transform the labels to integers if already fitted
        if hasattr(IntentTokenizer.le, 'classes_') and len(IntentTokenizer.le.classes_) > 0:
            """Get the labels from the given file."""
            df["label"] = df["label"].map(lambda s: '<unknown>' if s not in self.le.classes_ else s)
            labels = self.le.transform(df["label"])
        else:
            # Label encoder, fit and transform the labels to integers
            print("Encoding labels for the first time and adding unknown class.")
            labels = IntentTokenizer.le.fit_transform(df["label"])
            IntentTokenizer.le.classes_ = np.append(IntentTokenizer.le.classes_, '<unknown>')
            print("Label Encoding:", dict(zip(self.le.classes_, self.le.transform(self.le.classes_))))
        return labels
    def get_Inference_Tensor(self, df, device=torch.device("cpu")):
        """Process the data and return a TensorDataset."""
        if self.word2idx:
            # Initialize the attributes, index_sequences, padded, labels
            index_sequences = [text_to_indices(text,self.word2idx) for text in df["text"]]
            padded = convert_and_pad_sequences(index_sequences,device)
            return padded
        else:
            raise ValueError("Tokenizer not initialized with training data.")


    def save_state(self, file_path):
        """Save the tokenizer state to a file."""
        with open(file_path, 'wb') as file:
            pickle.dump(IntentTokenizer.word2idx, file)

    @classmethod
    def load_state(cls, file_path):
        """Load the tokenizer state from a file."""
        with open(file_path, 'rb') as file:
            cls.word2idx = pickle.load(file)
        cls._instance = IntentTokenizer.__new__(cls)
        return cls._instance

