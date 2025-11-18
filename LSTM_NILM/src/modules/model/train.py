import torch
#from sklearn.metrics import mean_absolute_percentage_error as MAPE

from modules.model.loss import MAPE_Loss
from modules.model.model import Forecaster as Model

from modules.data_processing.input_matrix import prepare_input_matrix
from modules.data_processing.pre_processing import pre_process

from modules.data_processing.post_processing import un_normalize

# Function that trains a model initialized with model_params on the given 
# training data and validation data for a given amount of epochs   
# it returns the trained model
def train_model(model_params, training_data_raw, validation_data_raw, epochs) :
    # Initialize the model
    model = Model(*model_params) 

    # first preprocess
    training_data, train_min, train_max = pre_process(training_data_raw)
    validation_data, val_min, val_max = pre_process(validation_data_raw)
    
    # prepare input matrix from preprocessed data
    time_steps = model.time_steps
    X_train, Y_train = prepare_input_matrix(training_data, time_steps)
    X_val, Y_val = prepare_input_matrix(validation_data, time_steps)

    # un normalize the Y values
    Y_train = un_normalize(Y_train, train_min, train_max)
    Y_val = un_normalize(Y_val, val_min, val_max)

    # initialize optimizer and loss function
    loss_function = MAPE_Loss()
    optimizer = torch.optim.Adam(model.parameters())

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
                #if(phase == 'train') :
                #    Y_pred = un_normalize(Y_pred, train_min, train_max)
                #else :
                #   Y_pred = un_normalize(Y_pred, val_min, val_max)

                loss = loss_function(Y_pred, Y)
                
                # backward pass only neccesary when training 
                if phase == 'train' :
                    loss.backward()
                    optimizer.step()

            error = loss.item()  
            
            # log training metrics
            if (epoch + 1) % 10 == 0 :
                print(f'Epoch [{epoch + 1}/{epochs}], Loss(MAPE):{loss.item():.2f}%, phase:{phase}')

    return model
