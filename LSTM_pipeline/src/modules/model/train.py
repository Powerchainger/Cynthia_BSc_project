import torch
import optuna
import json
import os
import pandas as pd

from modules.io.save import save_model
from modules.io.json_loaders import JSON_model_params_loader as Model_params
from modules.model.forecaster import Forecaster as Model

from modules.data_processing.input_tensors import create_input_tensors


def train_model(
    model_params,
    training_data,
    validation_data,
    min_max_dict,
    out_dir,
    file_name,
    appliances=[]
):
    """Trains a model using given params on given training data
    
    First a model will be initialized using the given params.
    Then the input tensors are created.
    Then the model will be trained for a given amount of epochs on the
    input tensors.
    Lastly, the weights of the model are saved to a file

    Keyword arguments:
    model_params -- the hyper-parameters of the model
    training_data -- the data to train the model on
    validation_data -- not directly used, but the validation loss is 
    displayed while training
    min_max_dict -- needed for the input
    out_dir -- directory to save the model weights to, if it does not 
    exist it will be created
    file_name -- name of the file that contains the model weights
    appliances -- list of appliances that are to be included in the
    input (default = [])

    Returns:
    The trained model and the validation loss throughout the training
    process
    """

    #initialize the model
    model = Model(*(model_params.to_tuple()))
    
    # create the input, target output
    time_steps = model.time_steps
    X_train, Y_train, _ = create_input_tensors(
        training_data,
        appliances,
        min_max_dict,
        time_steps
    )
    X_val, Y_val, _ = create_input_tensors(
        validation_data, 
        appliances,
        min_max_dict,
        time_steps
    )

    # initialize optimizer and loss function
    loss_function = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)  

    print(f'size of X_train:{len(X_train)}')
    print(f'size of X_val:{len(X_val)}')

    running_validation = []
    epochs = model.epochs
    for epoch in range(epochs):
        train_error = train_one_epoch(
            model,
            X_train,
            Y_train,
            loss_function,
            optimizer
        )
        val_error = evaluate_model(model,X_val,Y_val, loss_function) 
        running_validation.append(val_error)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(
                f'Epoch [{epoch + 1:3}/{epochs:3}], training loss:{train_error:.5f}, validation loss:{val_error:.5f}'
            )

    print(
        f'\nDone training model {file_name}\n final training loss:{train_error:.5f}, final validation loss:{val_error:.5f}'
    )    
    save_model(model, min_max_dict, out_dir, file_name)
    return model, running_validation

def train_one_epoch(model, X, Y, loss_function, optimizer):
    """Trains a model for a single epoch """
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
    """Evaluates the loss of the model"""
    model.eval()
    with torch.set_grad_enabled(False) :
        Y_pred = model(X)

    # compute loss
    loss = loss_function(Y_pred, Y)
    error = float(loss.item())

    return error 

def objective(
    trial,
    training_data,
    validation_data,
    time_steps,
    min_max_dict,
    appliances
):
    """ Meat of the hyper-parameter tuning of the optuna library
    
    First the search space of the hyper-parameters is defined 
    in params.
    Then, input and output tensors are created for the testing and
    validation data.
    Then, the model is then trained for <epochs> amount of epochs.
    The validation loss is reported at the end of every epoch, if it is
    too low to be a candidate of the optimal hyper-parameters then the 
    trial is pruned.

    Keyword arguments:
    trial -- the trial that is currently being performed
    training_data -- the data the model is trained on
    validation_data -- the data the mode is evaluated on 
    time_steps -- the length of the input sequences
    min_max_dict -- needed for the input
    appliances -- list of appliances that need to be in the input

    Returns:
    The final validation accuracy
    """
    params = {
        'nodes_per_layer' : trial.suggest_int('nodes_per_layer', 150, 250),
        'hidden_layers'   : trial.suggest_int('hidden_layers', 2, 4),
        'time_steps'      : time_steps, # somewhat fixed, taken from acf 
        'lr'              : trial.suggest_float('lr', 1e-4, 1e-2),
        'epochs'          : trial.suggest_int('epochs', 100, 200),
        'min_y'           : min_max_dict['main'][0],    # fixed parameter
        'max_y'           : min_max_dict['main'][1],    # fixed parameter
        'appliance_amount': len(appliances)             # fixed parameter
    }

    model = Model(**params)
    loss_function = torch.nn.SmoothL1Loss()                         
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)   
    
    X_train, Y_train, _ = create_input_tensors(
        training_data,
        appliances,
        min_max_dict,
        time_steps
    )
    X_val, Y_val, _ = create_input_tensors(
        validation_data,
        appliances,
        min_max_dict,
        time_steps
    )

    epochs = model.epochs
    for epoch in range(epochs):
        train_error = train_one_epoch(
            model,
            X_train,
            Y_train,
            loss_function,
            optimizer
        )
        validation_error = evaluate_model(model, X_val, Y_val, loss_function)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(
                f'Epoch [{epoch + 1:3}/{epochs:3}], training loss:{train_error:.5f}, validation loss:{validation_error:.5f}'
            )

        trial.report(validation_error, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    final_validation_accuracy = evaluate_model(
        model,
        X_val,
        Y_val,
        loss_function
    )
    return final_validation_accuracy

def hyper_parameter_tuning(
    training_data,
    validation_data,
    min_max_dict,
    out_dir,
    file_name,
    time_steps,
    appliances=[]
):
    """ Performs hyper-parameter tuning using the optuna library
    
    First the study is defined, the hyper-parameters that need to be
    tuned are defined, and the amount of trials to run is also defined.
    Then the trials will be ran, the final resutls will be stored in a
    JSON file.

    Keyword arguments:
    training_data -- the training data used for hyper-param tuning
    validation_data -- the validation data used for hyper-param tuning
    min_max_dict -- needed for the input
    out_dir -- directory to store results in after tuning
    file_name -- file that contains the best hyper-params found in the
    trials
    time_steps -- length of the input sequence
    appliances -- the appliances that need to be in the input 
    (default == [])inferenn
    
    Returns:
        The optimal hyper-params in the form of a Model_params object
    """
    #TODO update the modelparams to its new name
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
