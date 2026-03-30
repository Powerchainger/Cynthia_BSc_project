import torch
import os
import json
import pandas as pd

from modules.misc.utils import find_next_day_from_results

def save_model(model, min_max_dict, out_dir, file_name) :
    """ Creates a dir to save the model into as a .pt file"""
    dir_path = out_dir + '/models/'
    os.makedirs(dir_path, exist_ok=True)

    model_file_path = dir_path + file_name + '.pt'
    torch.save(model.state_dict(), model_file_path)

    dict_file_path = dir_path + file_name + '_min_max_dict.JSON'
    with open(dict_file_path, 'w') as f:
        json.dump(min_max_dict, f)


def save_values(Y_pred, Y, dates, out_dir, file_name='values'):
    """ Saves results to a csv file 

    First since the results usually overlap, the values for every day
    are isolated and then saved to a csv.

    Keyword arguments:
    Y_pred -- the results of model inference
    Y -- true values that the model tried to infer
    out_dir -- directory to save the file into
    file_name -- name of the csv file (default = 'values'
    """
    Y_pred, Y, dates = _filter_overlap(Y_pred, Y, dates) 
    df = pd.DataFrame(
            { 'time' : dates,
              'predicted' : Y_pred,
              'true' : Y
            }
    )
    save_path = out_dir + '/' + file_name + '.csv'
    df.to_csv(save_path, index=False)

def _filter_overlap(Y_pred, Y, dates):
    """ Helper function that filters overlap from results """
    assert(len(Y_pred) == len(Y))       # Something has gone wrong
    assert(len(Y_pred) == len(dates))   # BIG time if these are not 
    assert(len(Y) == len(dates))        # evaluated to true 

    new_Y_pred = []
    new_Y = []
    new_dates = []

    idx = find_next_day_from_results(dates)
    while idx < len(Y_pred):
        new_Y_pred = new_Y_pred + Y_pred[idx].tolist()
        new_Y = new_Y + Y[idx].tolist()
        new_dates = new_dates + dates[idx]
        idx = find_next_day_from_results(dates, idx + 1)

    return (new_Y_pred, new_Y, new_dates)    
