import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from modules.model.loss import MAPE_Loss
from modules.model.model import Forecaster as Model
from modules.model.train import train_one_epoch, evaluate_model

from modules.data_processing.input_matrix import create_input_matrix
# Assume get_model, get_dataloaders, train_one_epoch, evaluate_model are defined elsewhere

def objective(trial, training_data, validation_data):
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    num_epochs = trial.suggest_int("num_epochs", 10, 500)
    hidden_dim = trial.suggest_int("hidden_dim", 16, 256, log=True)
    input_days = trial.suggest_int("time steps (days)", 0, 7)

    # 2. Build Model, Optimizer, etc.
    model = Model(hidden_layers=num_layers, hidden_layer_nodes=hidden_dim, input_days=input_days, learning_rate=lr, epochs=num_epochs)

    loss_function = torch.nn.SmoothL1Loss()
    optimizer_class = torch.optim.Adam 
    optimizer = optimizer_class(model.parameters(), lr=lr)

    time_steps = model.time_steps
    X_train, Y_train, _ = create_input_matrix(training_data, time_steps)
    X_val, Y_val, _ = create_input_matrix(validation_data, time_steps)

    # 3. Training Loop with Pruning
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, X_train, Y_train, loss_function, optimizer)
        validation_error = evaluate_model(model, X_val, Y_val, loss_function)

        # 5. Report intermediate results for pruning
        trial.report(validation_error, epoch)

        # Handle pruning based on intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 4. Return Final Objective Value
    final_validation_accuracy = evaluate_model(model, X_val, Y_val, loss_function)
    return final_validation_accuracy # Optuna maximizes by default if not specified

# 6. Create and Run Study
def create_and_run_study(training_data, validation_data) :

    study = optuna.create_study(
        direction="minimize", # Minimize validation error
        pruner=optuna.pruners.MedianPruner() # Example pruner
    )
    study.optimize(lambda trial: objective(trial, training_data, validation_data), n_trials=1000) # Run 1000 trials

    # 7. Analyze Results
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    #TODO make JSON here
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
