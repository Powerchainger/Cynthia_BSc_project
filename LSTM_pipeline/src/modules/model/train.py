import torch
import optuna
import json
import os
import pandas as pd

from modules.io.save import save_model
from modules.io.model_params import Model_params
from modules.model.model import Forecaster as Model

from modules.data_processing.input_matrix import create_input_matrix
from modules.data_processing.pre_processing import pre_process

#TODO doc, type suggestions

def train_model(model_params, training_data, min_max_vals, out_dir, file_name, appliances=[]):
    
    #initialize the model
    model = Model(*(model_params.to_model_args()))
    
    # create the input, target output
    time_steps = model.time_steps
    X_train, Y_train, _ = create_input_matrix(training_data, appliances, min_max_vals, time_steps)

    # initialize optimizer and loss function
    loss_function = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

    epochs = model.epochs
    for epoch in range(epochs):

        error = train_one_epoch(model, X_train, Y_train, loss_function, optimizer)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1:3}/{epochs:3}], Loss:{error:.5f}')

    
    save_model(model, out_dir, file_name)
    return model

def train_one_epoch(model, X, Y, loss_function, optimizer):

    model.train()
    model.zero_grad()

    with torch.set_grad_enabled(True):
        Y_pred = model(X)

    # compute loss
    loss = loss_function(Y_pred, Y)
    error = float(loss.item())

    # backwards pass
    loss.backward()
    optimizer.step()

    return error 

def evaluate_model(model, X, Y, loss_function):

    model.eval()
    with torch.set_grad_enabled(False) :
        Y_pred = model(X)

    # compute loss
    loss = loss_function(Y_pred, Y)

    return loss.item()

def objective(trial, training_data, validation_data, min_max_vals, appliances):
    
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden_layers = trial.suggest_int('hidden_layers', 2, 5)
    nodes_per_layer = trial.suggest_int('nodes_per_layer', 16, 256)
    epochs = trial.suggest_int('epochs', 10, 500, log)
    time_steps = trial.suggest_int('time_steps', 10, 10 + 6*24, step=24)

    model = Model(
            nodes_per_layer=nodes_per_layer,
            hidden_layers=hidden_layers,
            time_steps=time_steps,
            lr=lr,
            epochs=epochs,
            min_y= min_max_vals['main'][0],
            max_y= min_max_vals['main'][1],
            appliance_amount=len(appliances)
    )
    
    loss_function = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    
    X_train, Y_train, _ = create_input_matrix(training_data, appliances, min_max_vals, time_steps)
    X_val, Y_val, _ = create_input_matrix(validation_data, appliances, min_max_vals, time_steps)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, X_train, Y_train, loss_function, optimizer)
        validation_error = evaluate_model(model, X_val, Y_val, loss_function)

        trial.report(validation_error, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    final_validation_accuracy = evaluate_model(model, X_val, Y_val, loss_function)
    return final_validation_accuracy

def hyper_parameter_tuning(training_data, validation_data, min_max_vals, out_dir, file_name, appliances=[]):

    study = optuna.create_study(
            direction='minimize', # minimize the validation error
            pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(lambda trial: objective(trial, training_data, validation_data, min_max_vals, appliances), n_trials=500)

    model_param_dict = dict(study.best_trial.params)
    model_param_dict['min_y'] = min_max_vals['main'][0]
    model_param_dict['max_y'] = min_max_vals['main'][1]
    model_param_dict['appliance_amount'] = len(appliances)

    dir_path = out_dir + '/model_parameters/'
    os.makedirs(dir_path, exist_ok=True)
    file_path = dir_path + file_name + '.json'

    with open(file_path, 'w') as f:
        json.dump(model_param_dict, f)

    return Model_params(file_path)
