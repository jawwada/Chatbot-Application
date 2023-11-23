This repository contains a detailed implementation of a machine learning application for intent classification, specifically designed to work with the ATIS dataset. 
Here's a summary of the key components and their functionalities:
1. Model Training.ipynb => This notebook contains the training of the basic models, LSTM and LSTM with Attention
2. Hyperparameter Optimization.ipynb => This notebook contains the hyperparameter optimization of the LSTM with Attention model 
3. Model Evaluation.ipynb => This notebook contains the evaluation of the LSTM with Attention model , and strategies to improve the model 
4. Model Abstraction, Transformers .ipynb => This notebook contains the abstraction of the model to a transformer model 

:db => This file contains the database file for the application. It is a sqlite database for optuna 

Dockerfile => This file contains the dockerfile for the application.  

README.md => This file contains the instructions for the application 

config => This folder contains the configuration files for the application 
└── IntentClassifierLSTMWithAttention.json => This file contains the configuration for the LSTM with Attention model 

config_bert.py => This file contains the configuration for the BERT model 

data  => This folder contains the data for the application 
└── atis => This folder contains the atis dataset 

docker-compose.yml => This file contains the docker-compose file for the application 

intent_classifier.py    => This file contains the intent classifier class for the application 

machine_learning => This folder contains the machine learning files for the application  

IntentClassifierLSTM.py => This file contains the LSTM model for the application  

IntentClassifierLSTMWithAttention.py => This file contains the LSTM with Attention model for the application  IntentTokenizer.py => This file contains the tokenizer class for the application  compute_metric.py  => This file contains the metric computation for the application. It contains the confusion matrix, roc curve, and other metrics.  hyperparameter_optimization.py => Executable in itself. This file contains the hyperparameter optimization for the application  main_example.py => Executable in itself. This file contains the main example for the application 
└── model_utils.py => This file contains the model utilities for the application 

main.py => Executable in itself. This file contains the main file for the application 

model_state_dict.pth => This file contains the state dict for the application

models => This folder contains the models for the application. It contains different models, that go as an input to flask server or can be used evaluation by the user through compute_metric.py  E-LSTMAmodel.pth   IntentClassifierLSTMWithAttention.pth  

plot_parallel_coordinate.png => This file contains the parallel coordinate plot for the hyperparameter optimization 

plotly_contours.png => This file contains the plotly contours for the hyperparameter optimization 

requirements.txt => This file contains the requirements for the application 

server.py => Executable in itself. This file contains the server for the application 

└── utils.py => This file contains the utility functions for the application 

A brief description of the files is as follows:

` Server Implementation (server.py): This file sets up a Flask server with Swagger documentation.
It defines routes for basic API checks (/api/example, /ready) and for performing intent classification (/intent).
The intent classification route handles POST requests, validates the input, and uses a loaded model for prediction.
Error handling is included for bad requests and internal server errors.

Configuration and Data Preprocessing (config_bert.py): This script handles the configuration for a machine learning model, particularly focusing on data preprocessing.
It includes loading and preprocessing of the ATIS dataset, label encoding, tokenization, and setting up vocabulary.
The script is designed to work with PyTorch and includes configurations for various neural network parameters.

Training and Evaluation (machine_learning/compute_metric.py): Focuses on training a model on the ATIS dataset, evaluating it, and visualizing metrics like accuracy, precision, recall, and F1 score.
It includes the creation of confusion matrices and ROC curves.
The script follows a structured approach to machine learning, similar to frameworks like HuggingFace Transformers.

Model Utilities (model_utils.py): Contains utility functions for training, evaluating, and predicting with models.
It also includes MLflow integration for experiment logging and managing hyperparameters.

Hyperparameter Optimization (machine_learning/hyperparameter_optimization.py): Uses Optuna for optimizing the hyperparameters of an intent classification model.
The script defines the objective function for the hyperparameters, search space, and integrates KFold cross-validation.
It logs the best parameters and model performance metrics to MLflow.

Model building Example (machine_learning/main_example.py): Demonstrates the complete workflow of training and evaluating an LSTM model with attention for intent classification.
This script includes data loading, preprocessing, model training, evaluation, and saving the model for future use.

Tokenizer Implementation (IntentTokenizer.py): Defines a tokenizer class for text processing, including building a vocabulary and encoding labels.
The class provides methods for processing data into a format suitable for machine learning models.

Model Definition Classes: 1. IntentClassifierLSTM.py: Defines a basic LSTM model for intent classification.

Model Definition Classes: 2. IntentClassifierLSTMWithAttention.py: Enhances the LSTM model with a self-attention mechanism for improved performance. 

Intent Classifier Wrapper (intent_classifier.py): Provides a high-level interface for the intent classification model, handling model loading, prediction, and processing.
Utility Functions (utils.py): Includes functions for tokenization, vocabulary building, label encoding, and sequence padding.

The purpose of this repo is to present a robust intent classification system with a focus on modularity, allowing for easy experimentation and extension. The integration of MLflow and Optuna for experiment tracking and hyperparameter optimization showcases a professional approach to machine learning development