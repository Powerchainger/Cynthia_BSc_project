import os
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

#TODO type suggestions

def _filter_weekdays(Y_pred, Y, dates):
    # the results, the true values, and the datetime correspoding must be of equal length 
    assert(len(Y_pred) == len(Y))
    assert(len(Y_pred) == len(dates))
    assert(len(Y) == len(dates))
    length = len(Y_pred)

    weekdays = [ [] for x in range(0,7) ]
    idx = 0
    while idx < length and dates[idx][0].time() != pd.to_datetime('00:00').time():
    
        idx = idx + 1

    # idx now points to first midnight
    while idx < length:
        head = dates[idx][0]
        weekday = head.weekday() 
        next_day = idx + 24
        
        day_pred = Y_pred[idx]
        day = Y[idx]
        weekdays[weekday].append((day_pred, day))

        idx = next_day

    return weekdays

def _compute_cumulative_sum(weekdays):

    total, total_pred = [], [] 
    avg, avg_pred = [], []

    for weekday in weekdays:
        
        Y_pred_total, Y_total = 0, 0
        for Y_pred, Y in weekday:
            Y_pred_total = Y_pred_total + sum(Y_pred)
            Y_total = Y_total + sum(Y)
          
        total.append(Y_total)
        total_pred.append(Y_pred_total)
        if (len(weekday) > 0):
            avg.append(Y_total / len(weekday))
            avg_pred.append(Y_pred_total / len(weekday))
        else:
            avg.append(0.0)
            avg_pred.append(0.0)

    return (total_pred, total), (avg_pred, avg)

def _plot_cumulative_sum(cum_sum_pred, cum_sum, path, plot_name):
    
    weekday_labels = [ 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', ]
    x = np.arange(len(weekday_labels))
    x_width = 0.5  
    x_offset = x_width / 2

    fig, ax = plt.subplots()
    ax.bar(x-x_offset, cum_sum_pred, x_width, label='predicted', color='r')
    ax.bar(x+x_offset, cum_sum, x_width, label='true', color='b')
    ax.set_xlabel('day of the week')
    ax.set_ylabel('cumulative sum')
    ax.set_xticks(x, weekday_labels)
    ax.legend()

    save_path = path + plot_name + '.png'
    plt.savefig(save_path, dpi=300)

def plot_cumulative_sum_results(Y_pred, Y, dates, out_dir):
    # 1. create the directory to save our results to
    save_path = out_dir + '/cumulative_sum/' 
    os.makedirs(save_path, exist_ok=True)
    
    # 2. filter the weekdays
    weekdays = _filter_weekdays(Y_pred, Y, dates)
    # 3. compute cumalitive sum
    total_cum_sum, avg_cum_sum = _compute_cumulative_sum(weekdays)
    # 4. save values, plot values
    #save_values(total_cum_sum, save_path, 'total')
    #save_values(avg_cum_sum, save_path, 'average')

    _plot_cumulative_sum(*total_cum_sum, save_path, 'total')
    _plot_cumulative_sum(*avg_cum_sum, save_path, 'average')


def plot_weekday(Y_weekday, weekday_idx, save_path):
   
    fig, ax = plt.subplots() 

    weekday_labels = [ 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', ]
    weekday_str = weekday_labels[weekday_idx]
    x_labels = list(range(0,24)) 
    x = np.arange(len(x_labels))
    
    for day_pred, day in Y_weekday:
        ax.plot(x, day_pred, '-', color='r', alpha=0.25)
        ax.plot(x, day, '-', color='b', alpha=0.25)

    ax.set_xlabel('Hour of the day')
    ax.set_ylabel('energy consumed [kwh]')
    ax.set_xticks(x, x_labels)
    desc = [ 'Predicted consumption', 'True consumption' ]
    ax.legend(desc)
    
    plt.suptitle(weekday_str)
    
    save_path = save_path + weekday_str + '.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_weekday_plots(Y_pred, Y, dates, out_dir):
    save_path = out_dir + '/weekday_plots/'
    os.makedirs(save_path, exist_ok=True)

    weekdays = _filter_weekdays(Y_pred, Y, dates)
    for idx, weekday in enumerate(weekdays):
        plot_weekday(weekday, idx, save_path)
