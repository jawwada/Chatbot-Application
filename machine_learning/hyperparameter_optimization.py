# Description: Hyperparameter optimization using Optuna
# This script uses Optuna to optimize the hyperparameters of an intent classification model.
# It uses the IntentClassifierLSTMWithAttention model as an example.
#

from sklearn.model_selection import KFold
import torch.nn as nn
from machine_learning.IntentClassifierLSTMWithAttention import IntentClassifierLSTMWithAttention
from machine_learning.IntentTokenizer import IntentTokenizer
import torch.optim as optim
import pandas as pd
import torch
from torch.utils.data import DataLoader
from machine_learning.model_utils import train, evaluate, predict,get_or_create_experiment
import optuna
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*setuptools.*")
import mlflow
from optuna.visualization import plot_optimization_history
import matplotlib.pyplot as plt



optuna.logging.set_verbosity(optuna.logging.ERROR)

# Load data
train_df = pd.read_csv('data/atis/train.tsv', sep='\t', header=None, names=["text", "label"])
test_df = pd.read_csv('data/atis/test.tsv', sep='\t', header=None, names=["text", "label"])

# Instantiate the tokenizer
tokenizer = IntentTokenizer(train_df)

# define constants and hyperparameters
vocab_size=tokenizer.max_vocab_size+1
output_dim=len(tokenizer.le.classes_)
batch_size = 32
num_epochs = 3

#   Define device
device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create DataLoaders
train_data = tokenizer.process_data(train_df, device=device)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
print("Number of training samples:", train_data.tensors[0].size())
print("Number of training batches:", len(train_loader))


test_data = tokenizer.process_data(test_df, device=device)
print("Number of test samples:", test_data.tensors[0].size())
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
print("Number of test batches:", len(test_loader))

def log_hyperparameters(trial):

    # Log hyperparameters
    mlflow.log_param("lr", trial.params["lr"])
    mlflow.log_param("hidden_dim", trial.params["hidden_dim"])
    mlflow.log_param("embedding_dim", trial.params["embedding_dim"])
    mlflow.log_param("dropout_rate", trial.params["dropout_rate"])
    mlflow.log_param("weight_decay", trial.params["weight_decay"])
    return

def log_metrics(trial, accuracy):
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    return

def objective(trial):
    with mlflow.start_run():
        # Suggest hyperparameters

        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256, 512])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        criterion = nn.CrossEntropyLoss()
        log_hyperparameters(trial)
        # Model, loss, and optimizer
        # model = IntentClassifierLSTM(cfg.vocab_size, embedding_dim, hidden_dim, cfg.output_dim,dropout_rate).to(device)
        model = IntentClassifierLSTMWithAttention(vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        kfold = KFold(n_splits=2, shuffle=True, random_state=42)
        fold_val_acc = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
            # Prepare fold data
            train_data_subset = tokenizer.process_data(train_df.loc[train_idx,:], device=device)
            val_data_subset = tokenizer.process_data(train_df.loc[val_idx,:], device=device)
            train_subset_loader = DataLoader(train_data_subset, batch_size=batch_size, shuffle=True)
            val_subset_loader = DataLoader(val_data_subset, batch_size=batch_size, shuffle=False)
            fold_loss = train(model, optimizer, criterion, train_subset_loader, num_epochs)
            val_accuracy = evaluate(model,  criterion, val_subset_loader, data_type="Validation")
            print(f'Fold: {fold + 1}, Training Loss: {fold_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            fold_val_acc.append(val_accuracy)
        average_val_acc = sum(fold_val_acc) / len(fold_val_acc)
        print(f'Average validation accuracy: {average_val_acc:.4f}')
        log_metrics(trial, average_val_acc)
    return average_val_acc

if __name__ == "__main__":

    experiment_id = get_or_create_experiment("IntentClassifierLSTMWithAttention")

    mlflow.set_experiment(experiment_id=experiment_id)
    storage = optuna.storages.RDBStorage(url="sqlite:///:db")
    study = optuna.create_study(storage=storage,direction="maximize")

    study.study_name = "IntentClassifierLSTMWithAttention"
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