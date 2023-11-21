import torch
import pandas as pd
import torch
import mlflow
import optuna
from torch.utils.data import DataLoader
from machine_learning.IntentTokenizer import IntentTokenizer


def train(model, optimizer, loss_function, train_loader, num_epochs=30):

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        correct = 0
        total=0

        for batch in train_loader:
            x, y = batch  # Move batch to device

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            y_hat = model(x)

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

        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", acc, step=epoch)

        # Print epoch summary
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {acc:.4f}')
    return train_loss

# Example usage


def evaluate(model, loss_function, test_loader, data_type="Test"):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0.0
    total=0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch in test_loader:
            x, y = batch

            # Forward pass
            y_hat = model(x)

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
    mlflow.log_metric("eval_loss", test_loss)
    mlflow.log_metric("eval_accuracy", acc)

    # Print test results
    print(f"{data_type} Loss: {test_loss:.4f}")
    print(f"{data_type} Accuracy: {acc:.4f}")

    # Optionally log the model
    mlflow.pytorch.log_model(model, "model")
    return acc

# Example usage
# Assuming model, loss_function, test_loader are already defined


def predict(model, query, tokenizer, device):
    # Tokenize query
    input = tokenizer.get_Inference_Tensor(query, device=device)
    print(input)
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