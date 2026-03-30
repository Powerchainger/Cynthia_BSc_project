import torch
import os

from modules.model.forecaster import Forecaster as Model 

from modules.io.save import save_values

from modules.data_processing.input_tensors import create_input_tensors

#from modules.metrics.plot import create_plot_per_day, create_EVO_plots
#from modules.metrics.error import compute_value_metrics
from modules.metrics.partial_dependence import plot_partial_dependence
#from modules.metrics.cumulative_sum import plot_cumulative_sum_results, create_weekday_plots
from modules.metrics.perm_feature_importance import permutation_feature_importance

def test_model(
    model,
    testing_data,
    min_max_dict,
    out_dir,
    results_name,
    appliances=[]
    ):
    """ Performs model inference, saves the results and targets values

    First the input, output, and dates are created for the testing data
    Then model inference is performed, the results are saved to a csv
    that contains 3 columns:
        1. date-time
        2. inferred values
        3. target values

    Keyword arguments:
    model -- model to be used for inference
    testing_data -- data to perform inference with
    min_max_dict -- needed for the input
    out_dir -- directory to save results to
    results_name -- name of the dir that stores the results file, if
    it doesn't exist it will be created
    appliances -- the appliances which are used for the input 
    (default = [])
    """
    #prepare input, 
    time_steps = model.time_steps
    X, Y, dates = create_input_tensors(testing_data, appliances, min_max_dict, time_steps)

    model.eval()
    with torch.no_grad():
        Y_pred = model(X)
    
    #has partially been moved to results_analysis
    # 1. create subfolder for results
    results_path = out_dir + '/' + results_name + '/' 
    os.makedirs(results_path, exist_ok=True)
    # 2. create cum_sum 
    #plot_cumulative_sum_results(Y_pred, Y, dates, results_path)  
    # 3. create pdp,
    plot_partial_dependence(model, testing_data, appliances, min_max_dict, results_path) 
    # 4. create all plots,
    #create_plot_per_day(Y_pred, Y, dates, results_path) 
    # 5. create plots for every weekday,
    #create_weekday_plots(Y_pred, Y, dates, results_path) 
    # 6. RMSE, MAPE, MAE
    #compute_value_metrics(Y_pred, Y, results_path)
    # 7. save the actual values for later use
    save_values(Y_pred, Y, dates, results_path)
    # 8. permutation feature importance
    permutation_feature_importance(model, testing_data, appliances, min_max_dict, results_path)

# Old code, might not work anymore
#def EVO_pilot_test_model(model_params, model, testing_data, min_max_dict, out_dir, results_name):
#    
#    time_steps = model.time_steps
#    X, Y, dates = create_input_matrix(testing_data, [], min_max_dict, time_steps)
#
#    model.eval()
#    with torch.no_grad():
#        Y_pred = model(X)
#    
#    results_path = out_dir + '/' + results_name + '/'
#    os.makedirs(results_path, exist_ok=True)
#
#    create_EVO_plots(Y_pred, Y, dates, results_path)
#    compute_value_metrics(Y_pred, Y, results_path) 
#    save_values(Y_pred, Y, dates, results_path)
