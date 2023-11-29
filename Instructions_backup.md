# Atis Intent Classification
TODO: Main things to add in the project
 mainly swagger, yaml, architecure diagram (swagger, flask, redis, machine learning experiment management, logging of user interactions for future model learnings, pytest




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
1. Setup logging
brew install grafana
brew services start grafana

brew install prometheus


2. Run Prometheus
Start Prometheus using the configuration file where you specified your Flask app as a target.

```bash
prometheus --config.file=prometheus.yml
```
```
Example logs maintained in logs/flask-logs.log
2023-11-24 00:55:21,985 - server_full - ERROR - Server encountered an internal error: 500 Internal Server Error: The server encountered an internal error and was unable to complete your request. Either the server is overloaded or there is an error in the application.
2023-11-24 00:56:26,057 - server_full - INFO - Loading model.IntentClassifierLSTMWithAttention
2023-11-24 00:56:26,080 - server_full - INFO - Application started successfully
2023-11-24 00:56:29,128 - server_full - INFO - User entry: find me a flight to miami
2023-11-24 00:56:29,128 - server_full - INFO - Predicted intent: [{'label': 'flight', 'confidence': 0.9681052565574646}, {'label': 'flight_time', 'confidence': 0.006745603866875172}, {'label': 'airport', 'confidence': 0.0047215186059474945}]
2023-11-24 00:58:02,219 - server_full - INFO - Loading model.IntentClassifierLSTMWithAttention
2023-11-24 00:58:02,242 - server_full - INFO - Application started successfully
2023-11-24 00:58:22,076 - server_full - INFO - User entry: find me a flight to miami
2023-11-24 00:58:22,076 - server_full - INFO - Predicted intent: [{'label': 'flight', 'confidence': 0.9681052565574646}, {'label': 'flight_time', 'confidence': 0.006745603866875172}, {'label': 'airport', 'confidence': 0.0047215186059474945}]
2023-11-24 00:58:34,544 - server_full - INFO - User entry: find me a flight to miami
2023-11-24 00:58:34,544 - server_full - INFO - Predicted intent: [{'label': 'flight', 'confidence': 0.9681052565574646}, {'label': 'flight_time', 'confidence': 0.006745603866875172}, {'label': 'airport', 'confidence': 0.0047215186059474945}]
2023-11-24 01:06:49,831 - server_full - INFO - User entry: set an alarm for 7 AM
```
to send thses logs to splunk or elk, install filebeat and configure it to send logs to elk server or splunk server


3. Run Grafana
4. Run Flask Server
5. Configure Prometheus
6. Verify Metrics Collection
7. Setup Grafana
8. Make 100 requests to flask server
9. How to store intents requests and responses in a database




Now, Prometheus will scrape metrics from your Flask application at the specified intervals.

6. Verify Metrics Collection
Open your browser and navigate to http://localhost:9090 (or wherever your Prometheus instance is running). Use the Prometheus UI to query and verify that metrics from your Flask app are being collected.

Also, you can check the /metrics endpoint of your Flask app directly by navigating to http://localhost:5000/metrics (adjust the port if necessary) to see the raw metrics being exposed.
Setup Grafana;
go to http://localhost:3000
set up data source as prometheus, and add the prometheus url as http://localhost:9090

Make 100 requests to flask server
python make_requests.py


*****

Swagger is also done
http://localhost:8090/apidocs/
Just check the port name, it is 8090
I have added the swagger documentation screenshot in the architecture folder
it is implemented in server0.2.py. change the port to 8080 there and run the server and go to http://localhost:8080/apidocs/
Also look at the api testing and if there is an error

*****

Only architecture diagram is left



How to store intents requests and responses in a database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///requests.db'
db = SQLAlchemy(app)
class Request(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    request = db.Column(db.String(200), nullable=False)
    response = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Request %r>' % self.id
Browse Database and share screenshots
Share screen shots of prometheus, grafana, database, swagger.




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

Todo: incorporate swagger in the docker setup, include the swagger documentation in the docker image, also swagfrom code structure sall be used to generate the swagger documentation
and not the comments structure

```bash
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


edge caes to handle:
Out-of-Scope Queries:

Users may ask questions or make requests that are unrelated to the bot's intended purpose. Handling such queries gracefully is important.
Ambiguous Queries:

Users may provide incomplete or ambiguous input that the bot struggles to understand. The bot should seek clarification or provide a helpful response.
Spelling and Typing Errors:

Users may make typographical errors or misspell words. The bot should attempt to correct or interpret these errors to provide relevant responses.
Slang and Abbreviations:

Users often use slang, abbreviations, or informal language. The bot should be capable of understanding and responding appropriately.
Multilingual and Multidialectal Input:

Users may communicate in different languages or dialects. A multilingual bot should handle these variations and provide responses in the user's preferred language.
User Frustration and Impatience:

Users may become frustrated or impatient if they don't receive immediate assistance. The bot should detect user frustration and respond empathetically.
Long and Complex Queries:

Some users may pose lengthy or complex questions. The bot should manage long conversations and provide concise, relevant answers.
Non-Text Input:

Users might send images, voice messages, or other non-textual content. The bot should be able to process and respond to different media types if necessary.
User Privacy and Security:

Users might share sensitive information inadvertently. The bot should handle such data with care and follow privacy and security best practices.
Repetitive Queries:

Users may ask the same question repeatedly, expecting a different answer. The bot should recognize these patterns and respond appropriately.
User Misunderstanding:

Users might misinterpret the bot's responses or intentions. The bot should be prepared to provide clarifications and avoid misunderstandings.
Handling User Criticism:

Users may express criticism or negative feedback. The bot should handle criticism professionally and attempt to address user concerns.
Humor and Casual Conversation:

Users might engage in casual or humorous conversations. While maintaining professionalism, the bot can engage in light banter if appropriate.
Inappropriate or Offensive Content:

Users may use offensive language or engage in inappropriate behavior. The bot should have mechanisms to detect and respond to such content or report it.
Integration with Third-Party Services:

If the bot integrates with external services or APIs, it should handle cases where those services are temporarily unavailable or return errors.
Fallback Mechanisms:

When the bot cannot understand or process a user's input, it should have a robust fallback mechanism to gracefully handle the situation and potentially transfer the conversation to a human agent.
Context Switching:

Users might switch topics or contexts within a conversation. The bot should be capable of maintaining context and switching smoothly between topics.
User Goodbyes:

Users may abruptly end a conversation. The bot should acknowledge the user's intention to leave and provide a pleasant closing message.
Understanding Emotions:

Detecting and responding to user emotions, such as frustration or sadness, can enhance the user experience.
Legal and Ethical Considerations:

Ensure the chatbot complies with legal and ethical guidelines, such as privacy laws and regulations related to data handling and user consent.
Designing a chatbot to handle these edge cases requires a combination of natural language processing (NLP) capabilities, robust dialogue management, and careful consideration of user experience principles. Regular testing and user feedback are crucial for identifying and addressing these edge cases in a chatbot's design.

User
What are the edge cases for flask server
ChatGPT
When designing and developing a Flask server or any web application, it's important to consider various edge cases to ensure that your server behaves correctly and robustly. Here are some common edge cases to consider when working with a Flask server:

Invalid HTTP Methods:

Ensure that your server handles HTTP methods like GET, POST, PUT, DELETE, etc., appropriately and returns the correct HTTP status codes for unsupported methods.
HTTP Status Codes:

Handle and return the appropriate HTTP status codes for various scenarios, such as success, errors, redirections, and client errors.
404 Not Found:

Implement a custom error handler for 404 errors to provide helpful responses for URLs that do not exist.
Timeouts:

Consider how your server handles requests that take too long to process, and implement appropriate timeouts to prevent resource exhaustion.
Large Requests and Responses:

Test how your server behaves when handling large payloads or responses, and implement appropriate handling and validation.
Security Vulnerabilities:

Address security concerns, such as SQL injection, Cross-Site Scripting (XSS), Cross-Site Request Forgery (CSRF), and other vulnerabilities.
Authentication and Authorization:

Ensure that your server correctly handles authentication and authorization, including edge cases like expired tokens or unauthorized access.
Concurrency and Race Conditions:

Consider how your server handles multiple simultaneous requests and whether it's susceptible to race conditions or resource conflicts.
CORS (Cross-Origin Resource Sharing):

Implement proper CORS handling to allow or restrict access to your server's resources from different origins.
Content Negotiation:

Handle content negotiation correctly, allowing clients to specify the desired response format (e.g., JSON, XML) using request headers.
Unicode and Character Encoding:

Ensure your server correctly handles Unicode characters and character encodings, preventing issues with character set mismatches.
Request Headers and Query Parameters:

Validate and handle request headers and query parameters properly, taking into account edge cases like missing or malformed headers/parameters.
Request and Response Size Limits:

Set reasonable limits on request and response sizes to prevent denial of service attacks or excessive resource consumption.
Server Crashes and Failures:

Implement error handling and recovery mechanisms to gracefully handle server crashes and failures, minimizing downtime.
Data Serialization and Deserialization:

Handle data serialization (e.g., JSON) and deserialization correctly, considering edge cases related to data format errors or invalid input.
File Uploads:

If your server allows file uploads, validate and handle uploaded files securely to prevent security risks like file overwrites or execution vulnerabilities.
Robust Logging:

Ensure that your server logs important events, errors, and exceptions, making it easier to diagnose issues in production.
Testing Edge Cases:

Develop thorough unit tests and integration tests that cover various edge cases to catch issues early in development.
Deployment Considerations:

Consider deployment-specific edge cases, such as load balancing, containerization, and scaling, to ensure your server behaves as expected in a production environment.
Graceful Shutdown:

Implement a graceful shutdown mechanism to allow your server to finish ongoing requests and clean up resources when it's time to stop the application.