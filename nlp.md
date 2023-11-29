NLTK (Natural Language Toolkit) is a powerful Python library for working with human language data (natural language processing). While NLTK itself does not directly provide functionality for generating word or sentence embeddings (like Word2Vec, GloVe, or BERT), it is often used for preprocessing steps such as tokenization, which is a prerequisite for using embeddings.

For word and sentence embeddings, you'll typically use libraries like Gensim for word embeddings (e.g., Word2Vec, FastText) 
and Transformers (by Hugging Face) for advanced sentence embeddings like BERT. I'll provide examples for both.

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Example text data (list of sentences)
documents = ["The quick brown fox jumps over the lazy dog.",
             "I love natural language processing.",
             "Word embeddings are amazing."]

# Tokenization
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# Get word vector for a specific word
word_vector = model.wv['language']
print(word_vector)

```

```python
from transformers import BertTokenizer, BertModel
import torch

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "Natural language processing with Transformers is fun."

# Encode text
input_ids = tokenizer.encode(sentence, return_tensors='pt')
with torch.no_grad():
    # Get last hidden states
    outputs = model(input_ids)
    last_hidden_states = outputs.last_hidden_state

# Pool the outputs into a single mean vector
sentence_embedding = torch.mean(last_hidden_states, dim=1)
print(sentence_embedding)
```
the same tokenizer can be used for both word and sentence embeddings, especially when using models like BERT from the Transformers library. BERT tokenizes input text into tokens, and each token receives its own embedding. Sentence embeddings can then be derived from these word embeddings, often by pooling (such as taking the mean, as in the previous example).

Why Use the Same Tokenizer for Both?
Consistency: Using the same tokenizer ensures consistency in how the text is split into tokens, which is crucial for models like BERT that have been trained on specific tokenization schemes.

Model Compatibility: Pre-trained models like BERT expect input to be tokenized in a specific way. Using the provided tokenizer ensures compatibility with the model's pre-training.

Why Take the Mean of Word Embeddings for Sentence Embeddings?
BERT provides contextual word embeddings – each word (token) embedding is influenced by the entire input sentence. To create a single vector representing the entire sentence, one common approach is to average these embeddings. This average (mean) serves as a simple but effective way to condense the information from all tokens into a single sentence-level representation.

Why is PyTorch Used in the Example?
PyTorch is used in the example because:

Model Implementation: The BERT model in the Hugging Face Transformers library is implemented using PyTorch (although TensorFlow implementations are also available).

Tensor Operations: PyTorch provides a convenient and efficient way to perform operations on tensors, which are the fundamental data structures used in deep learning models. In the example, operations like encoding the text and averaging the embeddings are done using PyTorch.

Example for Extracting Both Word and Sentence Embeddings from BERT
Here's how you can modify the previous example to extract both word and sentence embeddings:
```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentence = "Natural language processing with Transformers is fun."

# Tokenize and encode the sentence
input_ids = tokenizer.encode(sentence, return_tensors='pt')
with torch.no_grad():
    outputs = model(input_ids)

# Word embeddings (last hidden states)
word_embeddings = outputs.last_hidden_state.squeeze(0)

# Sentence embedding by mean pooling
sentence_embedding = torch.mean(word_embeddings, dim=0)

print("Word Embeddings:")
for token, embedding in zip(tokenizer.tokenize(sentence), word_embeddings):
    print(f"Token: {token}, Embedding: {embedding[:5]}...")  # Displaying first 5 dimensions for brevity

print("\nSentence Embedding:", sentence_embedding[:5])  # First 5 dimensions of sentence embedding
```
In this example, word_embeddings contains the embeddings for each token in the input sentence, and sentence_embedding is the mean of these embeddings, providing a representation for the entire sentence.

Conclusion
Tokenizers in models like BERT are versatile and can be used for generating both word-level and sentence-level embeddings. The choice of using mean pooling (or other methods) for sentence embeddings depends on the specific requirements of your application and the nature of the data you're working wit
###

Implementing a semantic search engine using BERT embeddings in a Siamese network architecture with triplet loss involves several steps. The goal here is to train a model that can effectively measure the semantic similarity between different text inputs, making it useful for tasks like finding the most semantically relevant documents given a query. Here's a high-level overview of how you can approach this:

1. Understanding the Components
BERT Embeddings: These provide contextual representations of text inputs.
Siamese Network Architecture: This involves two or more identical subnetworks with shared weights, used to learn similarity or relationships between different inputs.
Triplet Loss: This is a loss function used in machine learning to train a model to understand the relative similarity between inputs. A triplet consists of an anchor, a positive example (similar to the anchor), and a negative example (dissimilar from the anchor).
2. Data Preparation
Gather a dataset suitable for training a Siamese network. You'll need triplets consisting of an anchor text, a positive (semantically similar) text, and a negative (semantically dissimilar) text.
Preprocess the text data (tokenization, normalization, etc.).
3. Building the Siamese Network with BERT
Use a pre-trained BERT model as the base for each subnetwork in your Siamese network.
Ensure that all subnetworks share the same weights.
4. Implementing Triplet Loss
The triplet loss function is designed to minimize the distance between the anchor and the positive example and maximize the distance between the anchor and the negative example in the embedding space.
A common approach is to use cosine similarity or Euclidean distance as the measure of distance in the embedding space.
5. Training the Model
Train the model using your prepared dataset and the triplet loss function.
During training, ensure that the model learns to bring embeddings of similar texts closer and push embeddings of dissimilar texts further apart in the embedding space.
6. Building the Semantic Search Engine
Once the model is trained, use it to generate embeddings for the documents in your search engine's database.
For a given query, generate its embedding using the same model.
To find the most relevant documents, compare the query embedding with the document embeddings (e.g., using cosine similarity) and retrieve the documents with the highest similarity scores.
Example Code Outline
Here’s a simplified outline of what the code structure might look like (note that this is just an outline and not a complete implementation):

```python
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Siamese Network with BERT
class SiameseBERTNetwork(nn.Module):
    def __init__(self):
        super(SiameseBERTNetwork, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output

# Triplet Loss Function
def triplet_loss(anchor, positive, negative, margin=1.0):
    # Example using Euclidean distance
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()

# Instantiate the model, optimizer, etc.
model = SiameseBERTNetwork()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training Loop (simplified)
for epoch in range(num_epochs):
    for anchor, positive, negative in dataloader:  # Assume these are tokenized inputs
        optimizer.zero_grad()
        anchor_embedding = model(**anchor)
        positive_embedding = model(**positive)
        negative_embedding = model(**negative)
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()
`````
Important Considerations
Data Quality: The quality and diversity of your training data significantly impact the performance of the model.
Evaluation: Evaluate your model on a separate validation set to ensure it generalizes well.
Scalability: For a large-scale semantic search engine, consider scalability aspects, such as how to efficiently store and search through embeddings.
Fine-tuning: Depending on your specific use case, you may consider fine-tuning the BERT model on your dataset for better performance.
This approach leverages the power of BERT's embeddings and the effectiveness of Siamese networks with triplet loss to create a semantic understanding that can be applied in a search engine context.

### Encoder-Decoder Architecture
The encoder-decoder architecture is a neural network design pattern commonly used in natural language processing (NLP) for tasks like machine translation. It involves two subnetworks: an encoder and a decoder. The encoder takes an input sequence and encodes it into a fixed-length vector representation. The decoder then takes this vector representation and decodes it into a new sequence.
Encoder
An encoder is a component or model that transforms data from a high-dimensional space to a lower-dimensional space. In the context of machine learning and NLP:

Function: The encoder processes the input data and compresses the information into a more compact representation, often referred to as a "context vector" or "feature vector".
Output: This representation captures the essential information from the input data, reducing dimensionality and preparing it for further processing (like by a decoder in sequence-to-sequence models).
Usage: Encoders are used in tasks like feature extraction, data compression, and as the first half of autoencoder architectures.
Decoder
A decoder, conversely, takes the output from an encoder (or any condensed representation) and expands or reconstructs it back to a higher-dimensional space or into a different format.

Function: In sequence-to-sequence models (like machine translation), the decoder takes the context vector from the encoder and generates an output sequence, one element at a time.
Output: For example, in a translation task, the decoder generates a sequence of words in the target language.
Reconstruction: In autoencoders, the decoder reconstructs the input data from its encoded representation, aiming to minimize the difference between the original input and the reconstruction.
Differences Between Encoder and Decoder
Role: Encoders condense or compress information, while decoders expand or reconstruct information.
Direction of Data Flow: Data flows into the encoder (input) and out of the decoder (output).
Autoencoder
Autoencoders are a specific architecture in machine learning where an encoder and a decoder are used together.

Encoder Part: Compresses the input into a latent-space representation (encoding).
Decoder Part: Attempts to reconstruct the input from this latent representation.
Objective: The goal is often to learn a representation (encoding) for a set of data, typically for dimensionality reduction, denoising, or generative tasks.
Output: The output is the reconstructed version of the input.
Example in Context
Machine Translation: In a translation task, the encoder reads the input sentence (say, in English) and converts it into a context vector. The decoder then takes this vector and generates the translated sentence (say, in French).
Image Autoencoder: For an image autoencoder, the encoder compresses the image into a lower-dimensional representation, and the decoder reconstructs the image from this representation. The reconstructed image is usually compared to the original image to minimize reconstruction error.
Conclusion
Encoders and decoders are integral to various complex architectures in deep learning and NLP. They serve complementary roles – encoders for compressing or encoding information, and decoders for expanding or reconstructing information. In autoencoders, both are used in tandem to learn efficient data representations and for tasks like denoising or unsupervised learning.
et's consider an example of an autoencoder in PyTorch. An autoencoder is composed of an encoder and a decoder. The encoder compresses the input data into a lower-dimensional representation, and the decoder reconstructs the data back from this compressed representation. We will use a simple dataset like the MNIST dataset of handwritten digits for this example.

Dataset: MNIST
The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0 through 9).

Autoencoder Architecture
Our autoencoder will have:

An Encoder: Compresses the 28x28 images into a lower-dimensional latent space.
A Decoder: Reconstructs the images from the latent space representation.
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)  # Compressed representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Using Sigmoid because the output values are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training Complete")

```
latent space. The decoder then reconstructs the image back to 784 dimensions.
Dataset and DataLoader: We use the MNIST dataset, transform the images to tensors, and normalize them. The DataLoader provides batches of images for training.
Loss Function: We use Mean Squared Error (MSE) loss, which measures the reconstruction error between the output and the original image.
Training Loop: For each epoch, we pass the images through the autoencoder to get the reconstructed images, calculate the loss, and update the model's weights.
Output
After training, the model should have learned to compress and reconstruct the MNIST images. The loss value printed at each epoch represents how well the model is reconstructing the images - lower values indicate better reconstruction.
you can visualize the reconstructed images using matplotlib:
```python
import matplotlib.pyplot as plt

```
**NLP Concept and Data Example**
Imagine we want to build a model that translates English sentences to French. Our input (source) is an English sentence, and our output (target) is the corresponding French sentence.


Input Data (English Sentence): "Hello world"
Output Data (French Translation): "Bonjour le monde"
In a seq2seq model, the encoder will process the English sentence and create a context vector, a condensed representation of the sentence's meaning. The decoder will then use this vector to generate the French translation.
```
import torch
import torch.nn as nn
import torch.optim as optim

# Assume we have a simple vocabulary
word_to_ix = {"hello": 0, "world": 1}
ix_to_word = {0: "bonjour", 1: "le", 2: "monde"}

# Hyperparameters
embedding_dim = 256
hidden_dim = 256
vocab_size = len(word_to_ix)
output_size = len(ix_to_word)

# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Initialize models
encoder = EncoderRNN(vocab_size, hidden_dim)
decoder = DecoderRNN(hidden_dim, output_size)

# Example input
input_tensor = torch.tensor([word_to_ix["hello"], word_to_ix["world"]], dtype=torch.long)

# Encoding
encoder_hidden = encoder.initHidden()
for i in range(input_tensor.size()[0]):
    encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)

# Decoding
decoder_input = torch.tensor([[0]], dtype=torch.long)  # SOS token
decoder_hidden = encoder_hidden

decoded_words = []
for i in range(3):  # Let's limit the output length for simplicity
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    topv, topi = decoder_output.topk(1)
    decoded_words.append(ix_to_word[topi.item()])
    decoder_input = topi.squeeze().detach()

print("Translated:", " ".join(decoded_words))


```
Explanation:
Encoder: The EncoderRNN takes the input words (in this case, "hello" and "world"), embeds them into a higher-dimensional space, and then processes them with a GRU (Gated Recurrent Unit) to produce the context vector (encoder_hidden).

Decoder: The DecoderRNN starts with a special start-of-sequence (SOS) token and the context vector. It then generates one word at a time, using its own GRU and a linear layer to select the most likely next word.

Translation: The model generates a sequence of French words based on the context vector. In this simplified example, the translation is done word-by-word without considering the actual grammar or sentence structure.

Note:
This example is highly simplified and does not represent a practical translation model. Real-world applications would require:

A much larger and more complex vocabulary.
Handling of varying input and output lengths.
More sophisticated handling of sentence structure and context.
Training the model on a large dataset of parallel sentences in both languages



### BERT (Bidirectional Encoder Representations from Transformers) is a Transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google. As of 2019, Google has been leveraging BERT to better understand user searches. BERT is a method of pre-training language representations, meaning that we train a general-purpose "language understanding" model on a large text corpus (like Wikipedia), and then use that model for downstream NLP tasks that we care about (like question answering). BERT outperforms previous methods because it is the first unsupervised, deeply bidirectional system for pre-training NLP.
**Key Components of BERT's Architecture:**
Transformer Model:

BERT is based on the Transformer architecture, which relies heavily on an attention mechanism to draw global dependencies between input and output.
The Transformer includes two separate mechanisms: an encoder that reads the text input and a decoder that produces a prediction for the task. BERT only uses the Transformer's encoder mechanism.
Attention Mechanism:

At the heart of BERT (and the Transformer) is the attention mechanism, specifically "multi-head self-attention".
This mechanism allows the model to weigh the importance of different words in a sentence in the context of all other words, leading to a rich, contextualized word representation.
Bidirectionality:

Traditional language models were either left-to-right or right-to-left, but BERT is designed to be deeply bidirectional.
This bidirectionality is achieved through a novel pre-training task called "Masked Language Model" (MLM), where some percentage of the input tokens are masked at random, and the model is trained to predict these masked tokens.
Pre-training and Fine-tuning:

BERT involves two stages: pre-training and fine-tuning.
During pre-training, the model is trained on a large corpus of text (like Wikipedia and BooksCorpus) on two tasks: MLM and Next Sentence Prediction (NSP). This stage allows BERT to learn a general understanding of language.
During fine-tuning, BERT is further trained on a smaller, task-specific dataset.
Embedding Layers:

BERT uses a combination of token embeddings, segment embeddings, and position embeddings.
Token embeddings represent the individual words.
Segment embeddings allow the model to distinguish between different sentences (useful for tasks that require understanding the relationship between sentences).
Position embeddings provide the model with positional information about the tokens, as the Transformer does not have a recurrent structure to naturally understand sequence order.
Layer and Model Variants:

BERT comes in different sizes. The base model (BERT-Base) has 12 layers (transformer blocks), 12 attention heads, and 110 million parameters. BERT-Large has 24 layers, 16 attention heads, and 340 million parameters.
Example of BERT Processing a Sentence:
Input Representation:

A sentence is tokenized into words or subwords (WordPiece tokenization). Special tokens like [CLS] (used at the start of every sequence for classification tasks) and [SEP] (used to separate sentences) are added.
Passing Through Layers:

The input tokens pass through each layer of the Transformer. In each layer, the self-attention mechanism allows the model to consider other words in the sentence as it processes each word.
The output of each layer is the input to the next layer.
Output:

The final layer's output can be used for various NLP tasks. The [CLS] token's representation, in particular, is often used in classification tasks as it contains the aggregate information of the entire sequence.

### Mastering Transformers , summary

**Evolution of NLP toward Transformers**

We have seen profound changes in NLP over the last 20 years. 
During this period, we experienced different paradigms and finally entered a new era dominated mostly by 
magical Transformer architecture. This architecture did not come out of nowhere. 
Starting with the help of various neural-based NLP approaches, it gradually evolved to an attention-based encoder-decoder 
type architecture and still keeps evolving. The architecture and its variants have been successful thanks to the following 
developments in the last decade: 

1. Contextual word embeddings.
2. Better subword tokenization algorithms for handling unseen words or 
3. rare words Injecting additional memory tokens into sentences, such as Paragraph ID in Doc2vec or a Classification (CLS) token in Bidirectional Encoder Representations from Transformers (BERT) 
4. Attention mechanisms, which overcome the problem of forcing input sentences to encode all information into one context vector 
5. Multi-head self-attention 
6. Positional encoding to case word order 
7. Parallelizable architectures that make for faster training and 
8. fine-tuning Model compression (distillation, quantization, and so on) 
9. TL (cross-lingual, multitask learning) 

For many years, we used traditional NLP approaches such as n-gram language models, TF-IDF-based information retrieval models,
and one-hot encoded document-term matrices. 
All these approaches have contributed a lot to the solution of many NLP problems such 
as sequence classification, language generation, language understanding, and so forth. 
On the other hand, **these traditional NLP methods have their own weaknesses—for instance, 
falling short in solving the problems of sparsity, unseen words representation, 
tracking long-term dependencies**, and others. In order to cope with these weaknesses, 
we developed DL-based approaches such as the following: RNNs CNNs

**Bow, ngrams, tf-idf**
In a BoW approach, words and documents are represented with a one-hot encoding as a sparse way of representation, also known as the Vector Space Model (VSM). Text classification, word similarity, semantic relation extraction, word-sense disambiguation, and many other NLP problems have been solved by these one-hot encoding techniques for years. On the other hand, n-gram language models assign probabilities to sequences of words so that we can either compute the probability that a sequence belongs to a corpus or generate a random sequence based on a given corpus.
A BoW is a representation technique for documents by counting the words in them. The main data structure of the technique is a document-term matrix. Let's see a simple implementation of BoW with Python. The following piece of code illustrates how to build a document-term matrix with the Python sklearn library for a toy corpus of three sentences: 
```
from sklearn.feature_extraction.text 
import TfidfVectorizer 
import numpy as np 
import pandas as pd 
toy_corpus= ["the fat cat sat on the mat", "the big cat slept", "the dog chased a cat"] 
vectorizer=TfidfVectorizer() 
corpus_tfidf=vectorizer.fit_transform(toy_corpus) 
print(f"The vocabulary size is \ {len(vectorizer.vocabulary_.keys())} ") 
print(f"The document-term matrix shape is\ {corpus_tfidf.shape}") 
df=pd.DataFrame(np.round(corpus_tfidf.toarray(),2)) 
df.columns=vectorizer.get_feature_names() T
```
he output of the code is a document-term matrix, as shown in the following screenshot. 
The size is (3 x 10), but in a realistic scenario the matrix size can grow to much larger numbers such as 10K x 10M\


The table indicates a count-based mathematical matrix where the cell values are transformed by a 
Term Frequency-Inverse Document Frequency (TF-IDF) weighting schema. 
This approach does not care about the position of words. **Since the word order strongly determines the meaning, 
ignoring it leads to a loss of meaning. This is a common problem in a BoW method, which is finally solved by a recursion mechanism 
in RNN and positional encoding in Transformers.**
Each column in the matrix stands for the vector of a word in the vocabulary, and each row stands for the vector of a document.
Semantic similarity metrics can be applied to compute the similarity or dissimilarity of the words as well as documents. 
Most of the time, we use **bigrams** such as cat_sat and the_street to enrich the document representation. 
For instance, as the parameter ngram_range=(1,2) is passed to TfidfVectorizer, it builds a vector space containing 
both unigrams (big, cat, dog) and bigrams (big_cat, big_dog). 
Thus, such models are also called bag-of-n-grams, which is a natural extension of BoW. 
If a word is commonly used in each document, it can be considered to be high-frequency, such as and the. 
Conversely, some words hardly appear in documents, called low-frequency (or rare) words. 
As high-frequency and low-frequency words may prevent the model from working properly, 
TF-IDF, which is one of the most important and well-known weighting mechanisms, is used here as a solution. 
Inverse Document Frequency (IDF) is a statistical weight to measure the importance of a word in a document—for example, 
while the word the has no discriminative power, chased can be highly informative and give clues about the subject of the text. 
This is because high-frequency words (stopwords, functional words) have little discriminating power in understanding the documents. 
The discriminativeness of the terms also depends on the domain—for instance, 
a list of DL articles is most likely to have the word network in almost every document. 
IDF can scale down the weights of all terms by using their Document Frequency (DF), 
where the DF of a word is computed by the number of documents in which a term appears. 
Term Frequency (TF) is the raw count of a term (word) in a document.

**dimensionality problem**
The dimensionality problem is a common problem in BoW and n-gram models. it is a sparse representation of the documents.
for example, the big matrices are not suitable for training a model.

**Word2vec**, sorted out the dimensionality problem by producing short and dense representations of the words, called word embeddings. 
This early model managed to produce fast and efficient static word embeddings. 
It transformed unsupervised textual data into supervised data (self-supervised learning) by either predicting the 
target word using context or predicting neighbor words based on a sliding window. 
The following piece of code illustrates how to train word vectors for the sentences of the play Macbeth: 
```
from gensim.models import Word2vec 
model = Word2vec(sentences=macbeth, size=100, window= 4, min_count=10, workers=4, iter=10) 
```
The code trains the word embeddings with a vector size of 100 by a sliding 5-length context window. To visualize the words embeddings, we need to reduce the dimension to 3 by applying Principal Component Analysis (PCA) as shown in the following code snippet: 
```
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
import random np.random.seed(42) 
words=list([e for e in model.wv.vocab if len(e)>4]) 
random.shuffle(words) 
words3d = PCA(n_components=3,random_state=42).fit_transform(model.wv[words[:100]]) 
def plotWords3D(vecs, words, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2], c='b', marker='o')
    for word, x, y, z in zip(words, vecs[:, 0], vecs[:, 1], vecs[:, 2]):
        ax.text(x, y, z, word)

plotWords3D(words3d, words, "Visualizing Word2vec Word Embeddings using PCA")
```
**GloVe**, another widely used and popular model, argued that count-based models can be better than neural models. 
It leverages both global and local statistics of a corpus to learn embeddings based on word-word co-occurrence statistics. 
It performed well on some syntactic and semantic tasks, as shown in the following screenshot. 
The screenshot tells us that the embeddings offsets between the terms help to apply vector-oriented reasoning. 
We can learn the generalization of gender relations, which is a semantic relation from the offset between 
man and woman (man-> woman). Then, we can arithmetically estimate the vector of actress by adding the vector of 
the term actor and the offset calculated before. Likewise, we can learn syntactic relations such as word plural forms. 
For instance, if the vectors of Actor, Actors, and Actress are given, we can estimate the vector of Actresses.

### Idea of subword tokenization
The Word2vec-like models learn word embeddings by employing a prediction-based neural architecture. 
They employ gradient descent on some objective functions and nearby word predictions. 
While traditional approaches apply a count-based method, neural models design a prediction-based architecture for distributional semantics. 
Are count-based methods or prediction-based methods the best for distributional word representations?
The GloVe approach addressed this problem and argued that these two approaches are not dramatically different. 
Jeffrey Penington et al. even supported the idea that the count-based methods could be more successful by capturing global statistics. 
They stated that GloVe outperformed other neural network language models on word analogy, word similarity, and Named Entity Recognition (NER) tasks. 
These two paradigms, however, did not provide a helpful solution for **unseen words and word-sense problems. 
They do not exploit subword information, and therefore cannot learn the embeddings of rare and unseen words.**

FastText, another widely used model, proposed a new enriched approach using subword information, 
where each word is represented as a bag of character n-grams. The model sets a constant vector to each character n-gram and 
represents words as the sum of their sub-vectors, which is an idea that was first introduced by Hinrich Schütze (Word Space, 1993). 
The model can compute word representations even for unseen words and learn the internal structure of words such as suffixes/affixes, 
which is especially important with morphologically rich languages such as Finnish, Hungarian, Turkish, Mongolian, Korean, Japanese, Indonesian, and so forth. 
Currently, **modern Transformer architectures use a variety of subword tokenization methods such as WordPiece, SentencePiece, or Byte-Pair Encoding**

**ELMo**, a deep contextualized word representation model, was the first model to introduce the concept of contextual word embeddings.
It was a deep bidirectional language model that was trained on a large corpus of text. 
****


