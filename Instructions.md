# Intent Classification

## Project Overview
This project focuses on Text Classification for text data, leveraging a Neural Networks based machine learning model. Additionally, it showcases model deployment via a Docker-based Flask web server, enabling request handling and server interaction. 
Key features and processes within the project include:

1. **Data Preprocessing:** Implementation of a tokenizer and vocabulary builder.
2. **Model Training:** Development of a sequence based algorithm (LSTM + Embeddings) for Intent Classification. Refinement of the model using hyperparameter optimization. Addition of a self-attention mechanism for improved performance.
3. **Model Inference:** Application of the trained model to test data. Evaluation of model performance using metrics like confidence, accuracy, precision, recall, F1 score, lift, Roc curve etc..
4. **Model Deployment:** Deployment of the model using a Docker container and Flask web server.
5. **CI/CD Pipeline:** Integration of Continuous Integration and Continuous Deployment Using Docker (other services, for example git tag push and automatic deployment assumed).
6. **Unit Testing:** A basic testing of flask functionality through pytest.
7. **Logging and Health Checking** Utilization of the Python Logging Module for efficient tracking and debugging. Integration of Prometheus and Grafana for server health monitoring.
8. **Swagger:** Demonstration of sending requests to the Flask server and receiving responses.


## Requirements
- Python 3.9.10
- Docker (for Docker-based setup)
- PyCharm or Visual Studio Code (for IDE-based setup)
- Prometheus/ Grafana
- see requirements.txt for module versions

## Quick Start

To run the project from shell/locally , navigate to the project directory and set the PYTHONPATH:
```bash
cd ultimate-ai-challenge
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Create a new project environment with python 3.9 or 3.9.10
conda create -n py39 python=3.9

Install the required packages:

```bash
pip install -r requirements.txt
```

Running the pipeline script:
```bash
python machine_learning/pipelines/training_pipeline.py
python machine_learning/pipelines/hyperparameter_optimization_pipelines.py
python machine_learning/pipelines/evaluation_pipeline.py
```

Running the tests:
```bash
pytest
```

Running the Flask server:
```bash
python app/server.py --model=IntentClassifierLSTMWithAttention --port=8080
```
Keep server running, open another prompt, go to project root, setup python path like above and make example request for model inference:
```bash
python app/make_requests.py
```

### 2. Docker Setup
To build and run the project using Docker:
```bash
docker compose build
docker run -p 8080:8080 ultimate-a:ic-1.0
```

you can make requests by running python app/make_request.py

## CI/CD Pipeline
The CI/CD pipeline's basica component docker is implemented. Creating webhooks for automatic build and deployment to a deployment tool akin to kubernetes is not done. 
process could be assume below:
1. Create a new tag in git repo. Configure a webhook
2. Pipeline will be triggered, build the docker image and push the docker image to a container registry
3. Kubernetes or app service is configures to be deployed with the latest image



## Project Structure
The project is structured as follows:
```bash
.
├── Dockerfile
├── Instructions.md
├── Instructions_backup.md
├── README.md
├── app
│   ├── make_requests.py
│   └── server.py
├── config
│   ├── config_bert.py
│   ├── log_handler.py
│   └── model_initialization
│       └── IntentClassifierLSTMWithAttention.json
├── docker-compose.yml
├── machine_learning
│   ├── learners
│   │   ├── IntentClassifierLSTM.py
│   │   ├── IntentClassifierLSTMWithAttention.py
│   │   ├── IntentTokenizer.py
│   │   ├── intent_classifier.py
│   │   ├── model_utils.py
│   │   └── utils.py
│   ├── notebooks
│   │   ├── 1. Model Training.ipynb
│   │   ├── 2. Hyperparameter Optimization.ipynb
│   │   ├── 3. Model Evaluation.ipynb
│   │   ├── 4. Model Abstraction, Transformers .ipynb
│   │   ├── plot_parallel_coordinate.png
│   │   └── plotly_contours.png
│   └── pipelines
│       ├── evaluation_pipeline.py
│       ├── hyperparameter_optimization_pipeline.py
│       └── training_pipeline.py
├── main.py
├── prometheus.yml
├── requirements.txt
└── tests
    ├── conftest.py
    └── test_flask.py
```
## Project Components

The project folder has the following structure:
- **Dockerfile**: Contains the Dockerfile for the project.
- **Instructions.md**: Contains the project overview and setup instructions.
- **requirements.txt**: Contains the required packages for the project.
- **prometheus.yaml**: contains the prometheus configuration.
- **config**: Contains the configuration files for the project. Configuration files for application, e.g. logging, and model initialization JSON files for loading at runtime
- **data**: Contains the most of the learning data, blobs, database, trained model state dictionaries, and results.
- **docker-compose.yml**: Contains the docker-compose file for the project.
- **app**: Contains the Flask web server for handling requests.
  - **make_requests.py**: the script for making requests to the Flask server.
  - **server.py**: Contains the server script for the flask.
- **machine_learning**: Contains the machine learning code for the project.
  - **notebooks**: Contains the Jupyter notebook used for model training.
    - 1.Model Training.ipynb => This notebook contains the training of the basic models, LSTM and LSTM with Attention
    - 2. Hyperparameter Optimization.ipynb => This notebook contains the hyperparameter optimization of the LSTM with Attention model 
    - 3. Model Evaluation.ipynb => This notebook contains the evaluation of the LSTM with Attention model , and strategies to improve the model 
    - 4. Model Abstraction, Transformers .ipynb => This notebook contains the abstraction of the model to a transformer model 
  - **pipelines**
    - **inference.py**: Contains the code for making inferences using the trained model.
    - **model.py**: Contains the code for training the model.
    - **data_preprocessing.py**: Contains the code for preprocessing the data.


The project has logging for the main model building and inference tasks implemented using the Python logging module. The logs are stored in the `logs` directory which is created once the main script is run.


**Important Note:**
Whenever there is an error for project path or module not found, please set the PYTHONPATH to the project root directory
```
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```
***

**Improvements:**
1. Many free parameters and fixed values are used in the code, which can be moved to the json config files.
2. 