import torch
import os
import pandas as pd

from modules.misc.utils import find_next_day_from_results
# Function that saves a model to the given path.
# This is done by saving the weights of the model.
def save_model(model, out_dir, file_name) :
    dir_path = out_dir + '/models/'
    os.makedirs(dir_path, exist_ok=True)

    file_path = dir_path + file_name + '.pt'
    torch.save(model.state_dict(), file_path)

def save_values(Y_pred, Y, dates, out_dir, file_name='values'):
    #format: day, predicted, true, problem = overlap, solution = only values of days we save anyway
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
    assert(len(Y_pred) == len(Y))
    assert(len(Y_pred) == len(dates))
    assert(len(Y) == len(dates))

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

