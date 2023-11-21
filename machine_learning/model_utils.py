"""
This module contains utility functions for training and evaluating models. You can use these functions to train and
evaluate any model, not just the ones in this project. You can also use them to train and evaluate models in other
projects. This module also contains a function for predicting the label of a query using a trained model.
Additionally, this module contains a function for retrieving the ID of an existing MLflow experiment or creating a new
one if it doesn't exist. This is useful for logging metrics and artifacts to MLflow.
"""
import mlflow
import torch


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


def log_hyperparameters(trial):
    """
    Log hyperparameters to MLflow.
    :param trial:   The trial to log
    :return:    None
    """
    # Log hyperparameters
    mlflow.log_param("lr", trial.params["lr"])
    mlflow.log_param("hidden_dim", trial.params["hidden_dim"])
    mlflow.log_param("embedding_dim", trial.params["embedding_dim"])
    mlflow.log_param("dropout_rate", trial.params["dropout_rate"])
    mlflow.log_param("weight_decay", trial.params["weight_decay"])
    return


def log_metrics(trial, accuracy):
    """
    Log metrics to MLflow.
    :param trial:   The trial to log
    :param accuracy:    The accuracy to log
    :return:    None
    """
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    return