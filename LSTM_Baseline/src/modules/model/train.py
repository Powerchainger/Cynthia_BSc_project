import torch

from modules.model.loss import MAPE_Loss
from modules.model.model import Forecaster as Model

from modules.data_processing.input_matrix import create_input_matrix
from modules.data_processing.pre_processing import pre_process

from modules.data_processing.post_processing import scale_tensor

# Function that trains a model initialized with model_params on the given 
# training data and validation data for a given amount of epochs   
# it returns the trained model
def train_model(model_params, training_data, validation_data) :
    # Initialize the model
    model = Model(*model_params) 
    
    # create input matrix from preprocessed data
    time_steps = model.time_steps
    X_train, Y_train, _ = create_input_matrix(training_data, time_steps)
    X_val, Y_val, _ = create_input_matrix(validation_data, time_steps)

    # un normalize the Y values
    #Y_train = un_normalize(Y_train, train_min, train_max)
    #Y_val = un_normalize(Y_val, val_min, val_max)

    # initialize optimizer and loss function
    loss_function = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

    epochs = model.epochs
    for epoch in range(epochs) :
        for phase in ['train', 'validate'] :
            if phase == 'train' :

                model.train()
                X = X_train
                Y = Y_train

                # clear accumulated gradients
                model.zero_grad()

            else :
                model.eval()
                X = X_val
                Y = Y_val

            # only track gradients if we are training
            with torch.set_grad_enabled(phase == 'train') :
                # forward pass
                Y_pred = model(X)

                # post process the predicted values
                Y_pred = scale_tensor(Y_pred, torch.min(Y), torch.max(Y))

                # compute loss
                loss = loss_function(Y_pred, Y)
                
                # backward pass only neccesary when training 
                if phase == 'train' :
                    loss.backward()
                    optimizer.step()

            error = loss.item()  
            
            # log training metrics
            if (epoch + 1) % 10 == 0 :
                print(f'Epoch [{epoch + 1:3}/{epochs:3}], Loss:{loss.item():.5f}, phase:{phase}')

    return model

def train_one_epoch(model, X, Y, loss_function, optimizer) :

    model.train()
    model.zero_grad()

    Y_pred = model(X)
    Y_pred = scale_tensor(Y_pred, torch.min(Y), torch.max(Y))

    # compute loss
    loss = loss_function(Y_pred, Y)

    # backwards pass
    loss.backward()
    optimizer.step()

    return loss.item() 

def evaluate_model(model, X, Y, loss_function) :

    model.eval()
    with torch.set_grad_enabled(False) :
        Y_pred = model(X)

    # post process the predicted values
    Y_pred = scale_tensor(Y_pred, torch.min(Y), torch.max(Y))

    # compute loss
    loss = loss_function(Y_pred, Y)

    return loss.item()
