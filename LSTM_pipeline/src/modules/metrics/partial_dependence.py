import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from modules.data_processing.input_tensors import create_input_tensors

def plot_partial_dependence(model, data, appliances, min_max_dict, out_dir):
    plot_path = out_dir + '/' + 'partial_dependence' + '/'
    os.makedirs(plot_path, exist_ok=True)

    # pdp for day of the week
    plot_pdp_categorical(
        feature_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        pdp = weekday_pdp(model, data, appliances, min_max_dict),
        feature_name = 'Weekday',
        out_dir = plot_path 
    ) 

    # pdp for the hour of the day
    plot_pdp_categorical(
        feature_labels = list(range(0,24)),
        pdp = hour_pdp(model, data, appliances, min_max_dict),
        feature_name = 'Hour of the day',
        out_dir = plot_path
    )
                        
    for feature in ['main'] + appliances:
        n = 50 
        range_to_test = np.arange(
            start=data[feature].min(),
            stop= data[feature].max(),
            step=(data[feature].max() - data[feature].min()) / n
        )

        plot_pdp_numerical(
                feature_range = range_to_test,
                pdp = usage_pdp(model, feature, range_to_test, data, appliances, min_max_dict),
                feature_name = feature,
                out_dir = plot_path
        )

def plot_pdp_numerical(feature_range, pdp, feature_name, out_dir):
    fig, ax = plt.subplots()    
    ax.plot(feature_range, pdp)

    ax.set_xlabel('consumption [kwh]')
    ax.set_ylabel('cumulative sum')
    ax.set_title('Partial dependence for: ' + feature_name)
    
    save_path = out_dir + 'pdp_' + feature_name + '.png'
    plt.savefig(save_path, dpi=300)

def plot_pdp_categorical(feature_labels, pdp, feature_name, out_dir):

    fig, ax = plt.subplots()
    ax.bar(feature_labels, pdp)
    
    ax.set_ylabel('cumulative sum')
    ax.set_title('Partial depenedence for: ' + feature_name)

    save_path = out_dir + 'pdp_' + feature_name + '.png'
    plt.savefig(save_path, dpi=300)

# numerical features
def usage_pdp(model, feature, feature_range, data, appliances, min_max_dict):
    pdp = []
    copy = data.copy().dropna().reset_index(drop=True)
    for val in feature_range:
        copy[feature] = val
        
        X, _, _ = create_input_tensors(copy, appliances, min_max_dict, model.time_steps)
        model.eval()
        with torch.no_grad():
            Y_pred = model(X)
    
        pdp.append(Y_pred.sum() / len(Y_pred))

    return pdp

# categorical feature
def hour_pdp(model, data, appliances, min_max_dict):
    pdp = []
    for hour in range(0,24):
        copy = data.copy()
        time_to_copy = copy[copy['time'].dt.hour == hour]['time'].min()
        copy['time'] = copy['time'].map(lambda x: x.replace(hour=time_to_copy.hour))

        X, _, _ = create_input_tensors(copy, appliances, min_max_dict, model.time_steps)
        model.eval()
        with torch.no_grad():
            Y_pred = model(X)

        pdp.append(Y_pred.sum() / len(Y_pred))

    return pdp

# categorical feature
def weekday_pdp(model, data, appliances, min_max_dict):
    pdp = []
    for weekday in range(0,7):
        
        copy = data.copy()
        date_to_copy = copy[copy['time'].dt.weekday == weekday]['time'].min()
        year = date_to_copy.year
        month = date_to_copy.month
        day = date_to_copy.day

        copy['time'] = copy['time'].map(lambda x: x.replace(year=year, month=month, day=day))
        
        X, _, _ = create_input_tensors(copy, appliances, min_max_dict, model.time_steps)
        model.eval()
        with torch.no_grad():
            Y_pred = model(X)
        
        pdp.append(Y_pred.sum() / len(Y_pred))

    return pdp
