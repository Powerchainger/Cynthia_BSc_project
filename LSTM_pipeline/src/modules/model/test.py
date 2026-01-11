import torch
import os

from modules.model.model import Forecaster as Model 

from modules.data_processing.input_matrix import create_input_matrix
from modules.data_processing.pre_processing import pre_process

from modules.metrics.plot import create_plot_per_day
from modules.metrics.error import compute_value_metrics
from modules.metrics.partial_dependence import plot_partial_dependence
from modules.metrics.cumulative_sum import plot_cumulative_sum_results, create_weekday_plots


def test_model(model_params, model, training_data, testing_data, min_max_dict, out_dir, results_name, appliances=[]):
    
    #prepare input, 
    time_steps = model.time_steps
    X, Y, dates = create_input_matrix(testing_data, appliances, min_max_dict, time_steps)

    model.eval()
    with torch.no_grad():
        Y_pred = model(X)
    
    # 1. create subfolder for results
    results_path = out_dir + '/' + results_name + '/' 
    os.makedirs(results_path, exist_ok=True)
    # 2. create cum_sum 
    plot_cumulative_sum_results(Y_pred, Y, dates, results_path) 
    # 3. create pdp,
    plot_partial_dependence(model, testing_data, appliances, min_max_dict, results_path) 
    # 4. create all plots,
    create_plot_per_day(Y_pred, Y, dates, results_path)
    # 5. create plots for every weekday,
    create_weekday_plots(Y_pred, Y, dates, results_path) 
    # 6. RMSE, MAPE,
    compute_value_metrics(Y_pred, Y, results_path)

