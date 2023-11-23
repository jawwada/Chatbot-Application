# Atis Intent Classification

TODO: Add an Architecture Diagram of the project
the architecture diagram is added in the architecture folder. It is a high level architecture diagram. It included the User, Flask Server, interacting with User with Swagger UI. 
The models is cached flask_cache to Redis Server. There is a the CI/CD pipeline as well. Git Repo, webhooks, deployment to kubernetes or app service.
The Machine Learning Part and Server Part are decoupled. The Machine Learning Part can be deployed as a separate service, via MLFlow/Azure ML/Amazon SageMaker/Vertex AI. 
The modeler updates the model parameters through json files and sends it to the server. The server loads the model and serves the model.
The Logging can be done via Python Logging Module, Splunk or ELK. Add Logging in the architecture diagram.

TODO: Add an example CI/CD Pipeline (complete yaml file) of the project
Git Repo, webhooks, jenkins-to build and ship deployment to kubernetes or app service.

TODO: Add swagger work and a Swagger Documentation Screenshot of the project
the Swagger Documentation Screenshot is added in the architecture folder. It shows the Swagger Documentation for the project.

TODO: Add pytest, a Unit Testing for Model and Flask Server 

TODO: Add a Logging Mechanism for Flask Server. Python logging module in combination with Flask_logging, Splunk or ELK can be used for logging.

## Table of Contents


## Project Overview
This project focuses on Intent Classification, leveraging a pytorch based machine learning model for analysis and inference. Additionally, it showcases model deployment via a Docker-based Flask web server, enabling request handling and server interaction. 
The purpose of this project is to show case an initial but robust frame work to reserach, experiment, manager and deploy machine learning models.
1. Model Experimentation and Tracking (MLflow): The project includes MLflow integration for experiment tracking and management. This allows for easy experimentation and comparison of different models and hyperparameters. Logging the best results and models for deployment
2. Hyperparameter Optimization (Optuna): The project includes Optuna integration for hyperparameter optimization. This allows for efficient tuning of model hyperparameters, improving model performance.
3. Model Deployment (Docker + Flask): The project includes a Docker-based Flask web server for model deployment. This allows for easy interaction with the model via HTTP requests.
4. CI/CD Pipeline (Azure DevOps): The project includes a CI/CD pipeline file for Azure Devops Trigger, for continuous integration and deployment. This allows for automated building and deployment of the model, improving efficiency and reducing manual effort.
5. Swagger Documentation (Flask + Swagger): The project includes Swagger documentation for the Flask server. This allows for easy interaction with the server and model via a web interface.
6. Unit Testing (Pytest): The project includes unit tests for the model and server. This allows for efficient testing of the codebase, ensuring code quality and functionality.
7. Logging (Python Logging Module): The project includes logging for the model and server. This allows for efficient tracking and debugging of the codebase.
8. Flask_cache (Flask_cache): The project includes flask_cache for the server. This allows for efficient caching of the model, improving performance.
9. Model Abstraction (Transformers): The project includes abstraction of the model to a transformer model. This allows for easy adaptation to other datasets and generalization for serving to Flask.


Key features and processes within the project include:

1. **Data Preprocessing:** Implementation of a tokenizer and agnostic vocabulary builder.
2. **Model Training:** Development of a sequence based algorithm (LSTM + Embeddings) for Intent Classification. Refinement of the model using hyperparameter optimization. Addition of a self-attention mechanism for improved performance.
3. **Model Inference:** Application of the trained model to test data. Evaluation of model performance using metrics like confidence, accuracy, precision, recall, F1 score, lift, Roc curve etc..
4. **Model Deployment:** Deployment of the model using a Docker container and Flask web server.
5. **CI/CD Pipeline:** Integration of Continuous Integration and Continuous Deployment using a yaml file. That can be extended to provide webhooks to gitlab and deploy the application to kubernetes or app service.
6. **Unit Testing:** Ensuring code quality and functionality with Pytest.
7. **Logging:** Utilization of the Python Logging Module for efficient tracking and debugging.
8. **API Interaction:** Demonstration of sending requests to the Flask server and receiving responses.


## Requirements
- Python 3.9.10
- Docker (for Docker-based setup)
- PyCharm or Visual Studio Code (for IDE-based setup)

## Quick Start

To run the project from shell/locally , navigate to the project directory and set the PYTHONPATH:
```bash
cd ultimate-ai-challenge
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```
Note all the executable scripts from the project root as the python path is set to project root

There is a data folder in the project root, which contains the data files for the project. I have added two data files to the original data. This is machine generated Out-of-Sample data, which will helps us to test the model on unseen data. And also improves the model performance.


Install the required packages:
```bash
pip install -r requirements.txt
```

## Alternative Setups
### 1. Local Setup

Running the main script:
```bash
python src/machine_learning/main_example.py
python src/machine_learning/hyperparameter_optimization.py
python src/machine_learning/compute_metric.py
```
Running the tests:
```bash
pytest
```
Running the Flask server:
```bash
python server.py --model=IntentClassifierLSTMWithAttention --port=8080
```
Keep server running, open another prompt, go to project root, setup python path like above and make example request for model inference:
```bash
curl -X POST http://localhost:8080/intent \                                                                                                                                                  ─╯
-H "Content-Type: application/json" \
-d '{"text": "find me a flight to miami"}'
```

### 2. Docker Setup
To build and run the project using Docker:
```bash
docker build -t ultimate-ai-challenge .
docker run -p 8080:8080 ultimate-ai-challenge
```
Keep server running, open another prompt, go to project root, setup python path like above and make example request for model inference:


### 3. IDE Setup
To run the project using an IDE, first set the working directory to project root
then run any of the files in the `src` directory.
1. to run the main script, set the working directory to project root and run `main.py` in the `src` directory.
2. run app.py in the app directory to start the flask server.
3. run request.py in the request directory to make a request to the flask server.

## CI/CD Pipeline
The CI/CD pipeline is implemented using Azure DevOps. The pipeline is triggered when a new tag is pushed to the repository. The pipeline builds the Docker image and pushes it to the Azure Container Registry. The pipeline is defined in the `sensor_fault_detection.yaml` file.
process given below:
1. Create a new tag in git repository. A webhook will be triggered
2. Pipeline will be triggered and build the docker image
3. It will push the docker image to a container registry
4. Kubernetes or app service will be configured, deployed with the latest image

## Machine Learning Engineering:


## Project Structure

This repository contains a detailed implementation of a machine learning application for intent classification, specifically designed to work with the ATIS dataset. 
Here's a summary of the key components and their functionalities:
1. Model Training.ipynb => This notebook contains the training of the basic models, LSTM and LSTM with Attention
2. Hyperparameter Optimization.ipynb => This notebook contains the hyperparameter optimization of the LSTM with Attention model 
3. Model Evaluation.ipynb => This notebook contains the evaluation of the LSTM with Attention model , and strategies to improve the model 
4. Model Abstraction, Transformers .ipynb => This notebook contains the abstraction of the model to a transformer model 

***
These Notebooks present the complete workflow of training and evaluating an LSTM model with attention for intent classification. They include data loading, preprocessing, model training, evaluation, and saving the model for future use.
They also demonstrate the complete workflow of hyperparameter optimization for an intent classification model. They also include a detailed discussion on how to improve the model performance during production. An additional notebook is added to showcase the abstraction of the model to a transformer model.
Model generalization for serving to Flask , and easy adaptation to other datasets is also discussed in the notebooks.
***

### Project Files

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