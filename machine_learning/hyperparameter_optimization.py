"""
This script uses Optuna to optimize the hyperparameters of an intent classification model. It uses the
IntentClassifierLSTMWithAttention model as an example. You can use this script as a template to optimize the
hyperparameters of your own model.

Following are the steps to use this script:
1. Define the objective function. This function defines the objective function for Optuna. It trains the model using
the hyperparameters suggested by Optuna and returns the average validation accuracy. The average validation accuracy
is used as the objective function for Optuna to optimize.
2. Define the search space. This is done by using the suggest functions of the trial object. For example, the
following line of code suggests a learning rate between 1e-3 and 1e-1 on a log scale:
lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
3. Define the model, loss function, and optimizer. This is done inside the objective function. You can also define
them outside the objective function and pass them as arguments to the objective function.
4. Define the KFold cross validation. This is done inside the objective function. You can also define it outside the
objective function and pass it as an argument to the objective function.
5. Define the number of trials. This is done in the main function. You can also define it outside the main function
and pass it as an argument to the main function.
6. Define the number of epochs. This is done inside the objective function. You can also define it outside the
objective function and pass it as an argument to the objective function.
7. Define the number of folds. This is done inside the objective function. You can also define it outside the
objective function and pass it as an argument to the objective function.
8. Define the number of epochs in the KFold cross validation. This is done inside the objective function. You can
also define it outside the objective function and pass it as an argument to the objective function.
9. Define the optimizer and the loss function. This is done inside the objective function. You can also define them
outside the objective function and pass them as arguments to the objective function.
10. Define the model. This is done inside the objective function. You can also define it outside the objective
function and pass it as an argument to the objective function.
"""

# Description: Hyperparameter optimization using Optuna
# This script uses Optuna to optimize the hyperparameters of an intent classification model.
# It uses the IntentClassifierLSTMWithAttention model as an example.
#

import warnings

import mlflow
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# project imports
from machine_learning.IntentClassifierLSTMWithAttention import IntentClassifierLSTMWithAttention
from machine_learning.IntentTokenizer import IntentTokenizer
from machine_learning.model_utils import evaluate
from machine_learning.model_utils import get_or_create_experiment
from machine_learning.model_utils import train
from machine_learning.model_utils import log_hyperparameters
from machine_learning.model_utils import log_metrics

# Suppress warnings and set verbosity
warnings.filterwarnings("ignore", category=UserWarning, message=".*setuptools.*")
optuna.logging.set_verbosity(optuna.logging.ERROR)

# Load data
train_df = pd.read_csv('data/atis/train.tsv', sep='\t', header=None, names=["text", "label"])
test_df = pd.read_csv('data/atis/test.tsv', sep='\t', header=None, names=["text", "label"])

# Instantiate the tokenizer
tokenizer = IntentTokenizer(train_df)

# define constants and hyperparameters
vocab_size = tokenizer.max_vocab_size
output_dim = len(tokenizer.le.classes_)
batch_size = 32
num_epochs = 10

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


def objective(trial):
    """
    This function defines the objective function for Optuna. It trains the model using the hyperparameters suggested
    by Optuna and returns the average validation accuracy. The average validation accuracy is used as the objective
    function for Optuna to optimize.

    :param trial:   Optuna trial object
    :return:    Accuracy
    """
    with mlflow.start_run():
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        criterion = nn.CrossEntropyLoss()
        log_hyperparameters(trial)

        # Pick the model and train it. Evaluate the model on the test set.
        # Model, loss, and optimizer
        # model = IntentClassifierLSTM(cfg.vocab_size, embedding_dim, hidden_dim, cfg.output_dim,dropout_rate).to(device)
        model = IntentClassifierLSTMWithAttention(vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate).to(
            device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train the model using KFold cross validation for K=5
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_val_acc = []

        # For each fold, train the model and evaluate it on the validation set
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
            # Prepare fold data
            train_data_subset = tokenizer.process_data(train_df.loc[train_idx, :], device=device)
            val_data_subset = tokenizer.process_data(train_df.loc[val_idx, :], device=device)
            train_subset_loader = DataLoader(train_data_subset, batch_size=batch_size, shuffle=True)
            val_subset_loader = DataLoader(val_data_subset, batch_size=batch_size, shuffle=False)

            # Train the model
            fold_loss = train(model, optimizer, criterion, train_subset_loader, num_epochs)
            # Evaluate the model on the validation set
            val_accuracy = evaluate(model, criterion, val_subset_loader, data_type="Validation")
            print(f'Fold: {fold + 1}, Training Loss: {fold_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            fold_val_acc.append(val_accuracy)
        average_val_acc = sum(fold_val_acc) / len(fold_val_acc)
        print(f'Average validation accuracy: {average_val_acc:.4f}')
        log_metrics(trial, average_val_acc)
    return average_val_acc


if __name__ == "__main__":
    """
    Main function. This code below defines the experiment and runs Optuna to optimize the hyperparameters. It then
    trains the model using the best hyperparameters and logs the model and the hyperparameters to MLflow. It also
    prints the best hyperparameters and the best validation accuracy. It also plots the optimization history and
    parallel coordinate plots. 
    """
model_class_name = "IntentClassifierLSTMWithAttention"
experiment_id = get_or_create_experiment(model_class_name)

mlflow.set_experiment(experiment_id=experiment_id)
storage = optuna.storages.RDBStorage(url=f"sqlite:///{model_class_name}.db")
study = optuna.create_study(storage=storage, direction="maximize")

study.study_name = model_class_name
study.optimize(objective, n_trials=2)
best_trial = study.best_trial

with mlflow.start_run(experiment_id=experiment_id):
    # Log the best parameters
    mlflow.log_params(best_trial.params)

    # Train the model using best parameters
    model = IntentClassifierLSTMWithAttention(
        vocab_size,
        best_trial.params['embedding_dim'],
        best_trial.params['hidden_dim'],
        output_dim,
        best_trial.params['dropout_rate']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_trial.params['lr'],
                           weight_decay=best_trial.params['weight_decay'])
    train_loss = train(model, optimizer, nn.CrossEntropyLoss(), train_loader, 5)
    test_accuracy = evaluate(model, nn.CrossEntropyLoss(), test_loader, data_type="Test")
    print(f'Test Accuracy: {test_accuracy:.4f}')
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("train_loss", train_loss)
    mlflow.pytorch.log_model(model, f"best_model_{study.study_name}")

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Assuming 'study' is your Optuna study object
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()
