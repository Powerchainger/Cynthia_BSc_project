import torch
import numpy as np
import pandas as pd

from modules.data_processing.pre_processing import pre_process


DAY_LEN = 24
DAY_CUTOFF = 10

def create_input_tensors(df, appliances, min_max_dict, time_steps):
    """ Creates the input tensors for the model from the raw dataframe

    The function first pre-processes the dataframe to encode the raw 
    data, then it checks for each moment in time if an output can be 
    made, if so it checks if the input can be made. Once all inputs
    and outputs are gathered they are converted to tensors and returned

    Keyword arguments:
    df -- the raw data for which to create the input and targets
    appliances -- list of appliances that need to be added to the input
    min_max_dict -- dictionary of min max values that needs to be 
    passed to pre_process for min max normalization of the input
    time_steps -- length that the inputs must be

    Returns:
    A tuple, containing a tensor that contains the inputs,  
    another tensor of the same length that consists of the correct
    corresponding outputs, and a list that contains the dates and time
    corresponding to the outputs.

    Example:
    the input tensor looks like the following for a sequence of length
    5, containing 3 inputs:
    [[x, x, x, x, x], [x, x, x, x, x,], [x, x, x, x, x]]
    the corresponding output tensor looks like the following:
    [[y_0, y_1, ..., y_24],[y_0, ... y_24],[y_0, ... y_)24]]
    """
    targets = np.array(df['main'])
    dates = df['time']
    samples = pre_process(df, appliances, min_max_dict)

    idx = 0
    X, Y, Y_dates = [], [], []
    while(idx < len(targets)):
        
        if _can_make_input(targets, idx, time_steps):
            input_samples, target_samples, target_dates = _make_input(
                samples,
                targets,
                dates,
                time_steps,
                idx
            )
                
            X.append(input_samples)
            Y.append(target_samples)
            Y_dates.append(target_dates)
            
        idx = idx + 1

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    
    return X, Y, Y_dates


def _can_make_input(targets, idx, time_steps):
    """Checks if the output is complete, then if the input is complete 

    Helper function for create_input_tensor
    
    Keyword arguments:
    targets -- the available samples for the sequence
    idx -- the index to check if the input and output can be made for
    time_steps -- the length of the input sequence

    returns:
    boolean, true if the input sequence is complete and the target
    sequence is complete. Else it is false
    """ 
    # the input range
    X_start = idx
    X_end = idx + time_steps

    # the target range
    Y_start = X_end + (DAY_LEN - DAY_CUTOFF) 
    Y_end = Y_start + DAY_LEN 
   
    # check bounds
    if(X_start < 0 or Y_end > len(targets)):
        return False

    X = targets[X_start:X_end]
    Y = targets[Y_start:Y_end]

    # check if we are in bounds and if there are no nans
    if(Y_end > len(targets)) :
        return False
    else:
        return not pd.isna(X).any() and not pd.isna(Y).any() 

def _make_input(samples, targets, dates, time_steps, idx) : 
    """ creates the input from the samples """
    assert(len(samples) == len(targets) and len(samples) == len(dates))

    # make input, target, target_dates
    X_start = idx
    X_end = idx + time_steps

    Y_start = X_end + (DAY_LEN - DAY_CUTOFF)
    Y_end = Y_start + DAY_LEN

    # we need to be in bounds
    assert(X_start >= 0 and X_end < len(samples))
    assert(Y_start >= 0 and Y_end <= len(samples))

    X = np.array(samples[X_start:X_end])
    Y = np.array(targets[Y_start:Y_end])
    # list because I don't want to deal with np.datetime64 types
    Y_dates = list(dates[Y_start:Y_end]) 

    return X, Y, Y_dates
