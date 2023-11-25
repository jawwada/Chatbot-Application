"""
This module contains utility functions for training and evaluating models. You can use these functions to train and
evaluate any model, not just the ones in this project. You can also use them to train and evaluate models in other
projects. This module also contains a function for predicting the label of a query using a trained model.
Additionally, this module contains a function for retrieving the ID of an existing MLflow experiment or creating a new
one if it doesn't exist. This is useful for logging metrics and artifacts to MLflow.
"""
import mlflow
import torch
import torch.nn as nn
from machine_learning.pipelines.data_loaders import test_loader, device
import pandas as pd
from machine_learning.learners.IntentTokenizer import IntentTokenizer
import torch.optim as optim
import mlflow
from torch.utils.data import DataLoader
from machine_learning.pipelines.data_loaders import test_loader, tokenizer, train_df, batch_size, device, output_dim, vocab_size


def train(model, optimizer, loss_function, train_loader, num_epochs=30):
    """
    Train the model for the given number of epochs.
    :param model:  The model to train
    :param optimizer:   The optimizer to use
    :param loss_function:   The loss function to use
    :param train_loader:  The training data loader
    :param num_epochs:  The number of epochs to train for
    :return:    The final training loss
    """

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            x, y = batch  # Move batch to device

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            y_hat = model(x).to(x.device)

            # Compute loss
            loss = loss_function(y_hat, y)

            # Update train loss
            train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Step
            optimizer.step()

  
            # Compute accuracy
            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == y).sum().item()
            total+=y.size(0)

        # Compute average losses and accuracy
        train_loss /= len(train_loader)
        acc = correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {acc:.4f}')
        mlflow.log_metric("train_loss", train_loss)
        test_accuracy = evaluate(model, nn.CrossEntropyLoss(), test_loader, data_type="Test")
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.pytorch.log_model(model, f"{model.__class__.__name__}_{epoch}")
        if test_accuracy > 0.98:
            model.save_config_file(f"config/model_initialization/best_model_ELSTAMA.json")
            torch.save(model.state_dict(), f"data/models/best_model_ELSTAMA_state_dict.pth")
       # early stopping
        if epoch > 1 and train_loss < 0.1 and acc > 0.99:
            print("Early stopping")
            return train_loss
            break

        # Print epoch summary
    return train_loss

# Example usage


def evaluate(model, loss_function, test_loader, data_type="Test"):
    """
    Evaluate the model on the given dataset.
    :param model:   The model to evaluate
    :param loss_function:   The loss function to use
    :param test_loader:     The test data loader
    :param data_type:   The type of data (e.g. "Test", "Validation")
    :return:    The final test accuracy. Will be 0 if test_loader is None. We are not returning the test loss because
                we are not using it for anything. If you want to return it, you can.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0.0
    total=0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch in test_loader:
            x, y = batch

            # Forward pass
            y_hat = model(x).to(x.device)

            # Compute loss
            loss = loss_function(y_hat, y)
            test_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == y).sum().item()
            total+=y.size(0)

    # Compute average losses and accuracy
    test_loss /= len(test_loader)
    acc = correct / total

    # Log metrics

    # Print test results
    print(f"{data_type} Loss: {test_loss:.4f}")
    print(f"{data_type} Accuracy: {acc:.4f}")

    return acc

# Example usage
# Assuming model, loss_function, test_loader are already defined


def predict(model, query, tokenizer, device):
    """
    Predict the label of the given query.
    :param model:   The model to use
    :param query:   The query to predict
    :param tokenizer:   The tokenizer to use
    :param device:  The device to use
    :return:    The predicted label
    """
    # Tokenize query
    input = tokenizer.get_Inference_Tensor(query, device=device)
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input)
        _, predicted = torch.max(outputs, 1)

    # Convert prediction to label
    return tokenizer.le.inverse_transform(predicted.data.cpu().numpy())

def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)



# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def log_hyperparameters(trial):
    # Log hyperparameters

    mlflow.log_param("lr", trial.params["lr"])
    mlflow.log_param("hidden_dim", trial.params["hidden_dim"])
    mlflow.log_param("embedding_dim", trial.params["embedding_dim"])
    mlflow.log_param("dropout_rate", trial.params["dropout_rate"])
    mlflow.log_param("weight_decay", trial.params["weight_decay"])
    print(
        f'lr: {trial.params["lr"]}, hidden_dim: {trial.params["hidden_dim"]}, embedding_dim: {trial.params["embedding_dim"]}, dropout_rate: {trial.params["dropout_rate"]}, weight_decay: {trial.params["weight_decay"]}')

    return

#Define Custom Tokenizers Model Combinations and Save them
def save_modelname_tokenizer(model,modelname="best_ICELSTMAmodel",tokenizer=None):
    model.save_config_file(f"config/model_initialization/{modelname}.json")
    torch.save(model.state_dict(),f"data/models/{modelname}_state_dict.pth")
    tokenizer.save_state(f"data/models/{modelname}_tokenizer.pickle", f"data/models/{modelname}_le.pickle")
    return

#Query any model and get the prediction
def serve_modelname_query(modelname, query_text, model_query_length=100):
    model_serve = torch.load(f"data/models/{modelname}.pth").to(device)
    tokenizer = IntentTokenizer.load_state(IntentTokenizer,f"data/models/{modelname}_tokenizer.pickle", f"data/models/{modelname}_le.pickle")
    query = pd.DataFrame({"text": [query_text]})
    prediction = predict(model_serve, query,tokenizer,device)
    print(f"Predicted label: {prediction}")
    return prediction



def objective_ELSTMA(trial):
    from machine_learning.learners.IntentClassifierLSTMWithAttention import IntentClassifierLSTMWithAttention
    from sklearn.model_selection import KFold

    with mlflow.start_run():
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 256])
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        criterion = nn.CrossEntropyLoss()
        log_hyperparameters(trial)
        # Model, loss, and optimizer
        # model = IntentClassifierLSTM(cfg.vocab_size, embedding_dim, hidden_dim, cfg.output_dim,dropout_rate).to(device)
        model = IntentClassifierLSTMWithAttention(vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_val_acc = []
        num_epochs = 10
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

        # Log metrics
        mlflow.log_metric("train_loss", fold_loss)
        print(f'Foldloss: {fold_loss:.4f}')
        mlflow.log_metric("accuracy", average_val_acc)
        print(f'Average validation accuracy: {average_val_acc:.4f}')
        test_accuracy = evaluate(model, nn.CrossEntropyLoss(), test_loader, data_type="Test")
        print(f'Test Accuracy: {test_accuracy:.4f}')
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.pytorch.log_model(model, f"best_model_{trial.study.study_name}")
        if test_accuracy>0.97:
            mlflow.pytorch.log_model(model, f"best_model_{trial.study.study_name}_test_accuracy_{test_accuracy}")
    return average_val_acc