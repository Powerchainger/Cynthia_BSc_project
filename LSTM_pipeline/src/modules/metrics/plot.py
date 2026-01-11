import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd

def find_start(dates):

    idx = 0
    while idx < len(dates) and dates[idx][0].time() != pd.to_datetime('00:00').time():
        idx = idx + 1
    
    return idx

def create_plot_per_day(Y_pred, Y, dates, out_dir):

    save_path = out_dir + '/daily_plots/'
    os.makedirs(save_path, exist_ok=True)

    assert(len(Y_pred) == len(Y))
    assert(len(Y_pred) == len(dates))
    assert(len(Y) == len(dates))
    length = len(Y_pred) 

    idx = find_start(dates) 
    while idx < length: 
        head = dates[idx][0]
        _plot_day(Y_pred[idx], Y[idx], head, save_path)
        idx = idx + 24

def _plot_day(Y_pred, Y, date, save_path) : 

    fig, ax = plt.subplots()

    x_labels = list(range(0,24)) 
    x = np.arange(len(x_labels))

    weekday_labels = [ 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', ]
    weekday = weekday_labels[date.weekday()]
    date_str = date.strftime('%Y-%m-%d')

    ax.plot(x, Y_pred, 'o-', color='r', label='Predicted consumption')
    ax.plot(x, Y, 'o-', color='b', label='True consumption')
    ax.set_xlabel('hour of the day')
    ax.set_ylabel('energy consumed [kwh]')
    ax.set_xticks(x, x_labels)
    ax.legend()

    title = weekday + ' ' + date_str  
    plt.suptitle(date_str)

    save_path = save_path + date_str + '_' + weekday + '.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

