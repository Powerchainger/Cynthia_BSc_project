import torch
from sklearn.metrics import mean_absolute_percentage_error as MAPE

from modules.model.loss import MAPE_Loss
from modules.model.model import Forecaster as Model 

from modules.data_processing.input_matrix import prepare_input_matrix
from modules.data_processing.pre_processing import pre_process

from modules.data_processing.post_processing import un_normalize

# Function that receives: a model and testing data
# It will run the model on the testing data
# and then print the MAPE to stdout
def test_model(model, testing_data_raw) :
   
    # pre process the data
    testing_data, test_min, test_max = pre_process(testing_data_raw) 

    # prepare data for input
    time_steps = model.time_steps
    X, Y = prepare_input_matrix(testing_data, time_steps) 
    
    Y = un_normalize(Y, test_min, test_max)

    loss_function = MAPE_Loss()
    # eval the model on the data
    model.eval()
    with torch.no_grad() :
        Y_pred = model(X)
 
    loss = loss_function(Y_pred, Y)

    error = loss.item() 
    # log the metrics
    print(f'MAPE:{error:.2f}%') 
