"""
Extract Best Optuna study and play with the parameters
"""
import mlflow
import optuna
import torch.optim as optim
from machine_learning.learners.IntentClassifierLSTMWithAttention import IntentClassifierLSTMWithAttention
from machine_learning.learners.model_utils import train, evaluate
from machine_learning.learners.model_utils import get_or_create_experiment
from config.paramaters import vocab_size
model_class_name = "IntentClassifierLSTMWithAttention"
experiment_id = get_or_create_experiment(model_class_name)
import torch.nn as nn

from machine_learning.pipelines.data_loaders import train_loader, test_loader
from machine_learning.pipelines.data_loaders import num_epochs, device, output_dim, vocab_size


mlflow.set_experiment(experiment_id=experiment_id)
storage = optuna.storages.RDBStorage(url=f"sqlite:///data/db/{model_class_name}.db")
study = optuna.create_study(study_name = f"model_class_name_test" ,storage=storage, direction="maximize", load_if_exists=True)

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
    train_loss = train(model, optimizer, nn.CrossEntropyLoss(), train_loader, num_epochs)
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

fig = optuna.visualization.plot_slice(study)
fig.show()

fig = optuna.visualization.plot_contour(study)
fig.show()