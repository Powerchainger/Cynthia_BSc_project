import torch
from sklearn.metrics import mean_absolute_percentage_error as MAPE

from modules.model.loss import MAPE_Loss
from modules.model.model import Forecaster as Model 

from modules.data_processing.input_matrix import create_input_matrix
from modules.data_processing.pre_processing import pre_process

from modules.data_processing.post_processing import scale_tensor

from modules.metrics.plot import plot_cumalitive_sum, plot_all_weekday_error, save_all_results 

# Function that receives: a model and testing data
# It will run the model on the testing data
# and then print the MAPE to stdout
def test_model(model, testing_data, results_path) :

    # prepare data for input
    time_steps = model.time_steps
    X, Y, dates = create_input_matrix(testing_data, time_steps) 

    loss_function = MAPE_Loss()
    # eval the model on the data
    model.eval()
    with torch.no_grad() :
        Y_pred = model(X)

    # post process the predicted values
    Y_pred = scale_tensor(Y_pred, torch.min(Y), torch.max(Y))

    loss = loss_function(Y_pred, Y)
    error = loss.item() 

    # log the metrics
    print(f'MAPE:{error:.2f}%') 

    plot_cumalitive_sum(Y_pred, Y, dates, results_path)
    plot_all_weekday_error(Y_pred, Y, dates, results_path)
    save_all_results(Y_pred, Y, dates, results_path) 

