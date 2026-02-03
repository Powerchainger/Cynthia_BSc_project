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

def train_model(model_params, training_data, validation_data, min_max_dict, out_dir, file_name, appliances=[]):
    
    #initialize the model
    model = Model(*(model_params.to_model_args()))
    
    # create the input, target output
    time_steps = model.time_steps
    X_train, Y_train, _ = create_input_matrix(training_data, appliances, min_max_dict, time_steps)
    X_val, Y_val, _ = create_input_matrix(validation_data, appliances, min_max_dict, time_steps)

    # initialize optimizer and loss function
    loss_function = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)  

    print(f'size of X_train:{len(X_train)}')
    print(f'size of X_val:{len(X_val)}')

    running_validation = []
    epochs = model.epochs
    for epoch in range(epochs):
        train_error = train_one_epoch(model, X_train, Y_train, loss_function, optimizer)
        val_error = evaluate_model(model,X_val,Y_val, loss_function) 
        running_validation.append(val_error)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(f'Epoch [{epoch + 1:3}/{epochs:3}], training loss:{train_error:.5f}, validation loss:{val_error:.5f}')

    print(f'\nDone training model {file_name}, final training loss:{train_error:.5f}, final validation loss:{val_error:.5f}')    
    save_model(model, out_dir, file_name)
    return model, running_validation

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
    error = float(loss.item())

    return error 

def objective(trial, training_data, validation_data, time_steps, min_max_dict, appliances):
   
    params = {
        'nodes_per_layer' : trial.suggest_int('nodes_per_layer', 150, 250),
        'hidden_layers'   : trial.suggest_int('hidden_layers', 2, 4),
        'time_steps'      : time_steps,                # somewhat fixed, taken from data analysis
        'lr'              : trial.suggest_float('lr', 1e-4, 1e-2),
        'epochs'          : trial.suggest_int('epochs', 100, 200),
        'min_y'           : min_max_dict['main'][0],   # fixed parameter
        'max_y'           : min_max_dict['main'][1],   # fixed parameter
        'appliance_amount': len(appliances)           # fixed parameter
    }

    model = Model(**params)
    loss_function = torch.nn.SmoothL1Loss()                         
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)   
    
    X_train, Y_train, _ = create_input_matrix(training_data, appliances, min_max_dict, time_steps)
    X_val, Y_val, _ = create_input_matrix(validation_data, appliances, min_max_dict, time_steps)

    epochs = model.epochs
    for epoch in range(epochs):
        train_error = train_one_epoch(model, X_train, Y_train, loss_function, optimizer)
        validation_error = evaluate_model(model, X_val, Y_val, loss_function)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(f'Epoch [{epoch + 1:3}/{epochs:3}], training loss:{train_error:.5f}, validation loss:{validation_error:.5f}')

        trial.report(validation_error, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    final_validation_accuracy = evaluate_model(model, X_val, Y_val, loss_function)
    return final_validation_accuracy

def hyper_parameter_tuning(training_data, validation_data, min_max_dict, out_dir, file_name, time_steps, appliances=[]):

    study = optuna.create_study(
            direction='minimize', # minimize the validation error
            pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(
        lambda trial: objective(
            trial,
            training_data,
            validation_data,
            time_steps,
            min_max_dict,
            appliances
        ),
        n_trials=10
    )

    model_param_dict = dict(study.best_trial.params)
    model_param_dict['min_y'] = min_max_dict['main'][0]
    model_param_dict['max_y'] = min_max_dict['main'][1]
    model_param_dict['appliance_amount'] = len(appliances)
    model_param_dict['time_steps'] = time_steps

    dir_path = out_dir + '/model_parameters/'
    os.makedirs(dir_path, exist_ok=True)
    file_path = dir_path + file_name + '.json'

    with open(file_path, 'w') as f:
        json.dump(model_param_dict, f)

    return Model_params(file_path)
