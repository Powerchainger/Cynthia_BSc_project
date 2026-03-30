import torch
import os
import numpy as np
import pandas as pd

from modules.data_processing.pre_processing import pre_process
from modules.model.forecaster import Forecaster as model

def run_model(
    model,
    input,
    min_max_dict,
    out_dir,
    file_name = 'results',
    appliances = []
):
    """Runs a model on a input and prints results to stdout

    Keyword arguments:
    model -- the model to be used for inference
    input -- a pd dataframe of length 24
    min_max_dict -- needed for input
    appliances -- needed for input, (default = [])
    file_name -- name for the output file (default = 'results')
    out_dir -- directory to save the output to

    returns:
    A size 24 list consisting of the predictions
    """

    time_steps = model.time_steps
    X = _create_single_input_tensor(input, appliances, min_max_dict)

    model.eval()
    with torch.no_grad():
        Y_pred = model(X)

    results_path = out_dir + '/' + file_name + '.csv'
    df = pd.DataFrame({
        'hour' : list(range(0,24)),
        'predicted' : Y_pred[0]
    })

    df.to_csv(results_path, index=False)
    print(Y_pred)
    return list(Y_pred)

def _create_single_input_tensor(raw, appliances, min_max_dict):
    """Creates a single input tensor from the raw input data"""
    samples = pre_process(raw, appliances, min_max_dict)
    return torch.tensor(np.array([samples[:24]]), dtype=torch.float32)
