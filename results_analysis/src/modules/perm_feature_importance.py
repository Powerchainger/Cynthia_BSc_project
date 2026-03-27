import pandas as pd
import numpy as np
import os
import torch

from sklearn.metrics import mean_absolute_error

from modules.data_processing.input_matrix import create_input_matrix

def _measure_error(model, df, appliances, min_max_dict):
    
    X, Y, _ = create_input_matrix(df, appliances, min_max_dict, model.time_steps)
    model.eval()
    with torch.no_grad():
        Y_pred = model(X)

    return mean_absolute_error(Y, Y_pred)

def _permute_hour_importance(model, df, appliances, min_max_dict, perm_amt):

    avg_error = 0.0

    original_years = df['time'].dt.year
    original_months = df['time'].dt.month
    original_days = df['time'].dt.day
    for _ in range(0,perm_amt):

        shuffled_time = df['time'].sample(frac=1).reset_index(drop=True)
        temp_df = pd.DataFrame({
            'years' : original_years, 
            'months': original_months,
            'days'  : original_days,
            'hours' : shuffled_time
        })

        # applies to the shuffled time the original dates 
        df['time'] = temp_df.apply(
            lambda x: x.hours.replace(
                year=x.years,
                month=x.months,
                day=x.days
            ),
            axis=1
        )

        avg_error = avg_error + _measure_error(model, df, appliances, min_max_dict)

    avg_error = avg_error / perm_amt
    return avg_error

def _permute_weekday_importance(model, df, appliances, min_max_dict, perm_amt):

    avg_error = 0.0 

    original_hours = df['time'].dt.hour
    for _ in range(0,perm_amt):

        shuffled_dates = df['time'].sample(frac=1).reset_index(drop=True)
        temp_df = pd.DataFrame({'dates' : shuffled_dates, 'hours' : original_hours})
        # applies to the shuffled date and time the original time
        df['time'] = temp_df.apply(lambda x: x.dates.replace(hour=x.hours), axis=1)

        avg_error = avg_error + _measure_error(model, df, appliances, min_max_dict)

    avg_error = avg_error / perm_amt
    return avg_error
   
    # copy the hours
    # shuffle the dates
    # for each index on the time replace the hour with the hour of the old series

def _permute_numerical_importance(feature, model, df, appliances, min_max_dict, perm_amt):
    
    avg_error = 0.0
   
    for _ in range(0, perm_amt):

        df[feature] = df[feature].sample(frac=1).reset_index(drop=True)
        avg_error = avg_error + _measure_error(model, df, appliances, min_max_dict)

    avg_error = avg_error / perm_amt
    return avg_error

def permutation_feature_importance(model, df, appliances, min_max_dict, out_dir, perm_amt=10):

    df = df.dropna().reset_index(drop=True)
    base_error = _measure_error(model, df, appliances, min_max_dict) 

    error_dict = {}
    for feature in ['main'] + appliances :
        copy = df.copy()
        error = _permute_numerical_importance(feature, model, copy, appliances, min_max_dict, perm_amt)
        error_dict[feature] = error
    
    copy = df.copy()
    error_dict['weekday'] = _permute_weekday_importance(model, copy, appliances, min_max_dict, perm_amt)
    copy = df.copy()
    error_dict['hour'] = _permute_hour_importance(model, copy, appliances, min_max_dict, perm_amt)

    for key, val in error_dict.items():
        diff = val - base_error
        error_dict[key] = diff

    save_path = out_dir + 'permutation_feature_importance.txt' 
    with open(save_path, 'w') as f:
        for key, val in error_dict.items():
            print(f'{key} : {val}', file=f)

