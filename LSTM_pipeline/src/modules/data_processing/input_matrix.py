import torch
import numpy as np
import pandas as pd

from modules.data_processing.pre_processing import pre_process

DAY_LEN = 24
DAY_CUTOFF = 10

#TODO: type suggestions
def create_input_matrix(df, appliances, min_max_dict, time_steps):

    targets = np.array(df['main'])
    dates = df['time']
    samples = pre_process(df, appliances, min_max_dict)

    idx = 0
    X, Y, Y_dates = [], [], []
    while(idx < len(targets)):
        
        if _can_make_input(targets, idx, time_steps):
            input_samples, target_samples, target_dates = _make_input(samples, targets, dates, time_steps, idx)
                
            X.append(input_samples)
            Y.append(target_samples)
            Y_dates.append(target_dates)
            
        idx = idx + 1

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    
    return X, Y, Y_dates

def _can_make_input(targets, idx, time_steps):
    
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
    Y_dates = list(dates[Y_start:Y_end]) # list because I don't want to deal with np.datetime64 types

    return X, Y, Y_dates
