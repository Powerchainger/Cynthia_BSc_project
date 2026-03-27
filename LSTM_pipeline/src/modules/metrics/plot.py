import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd

from modules.misc.utils import find_next_day_from_results


#unused 
def create_EVO_plots(Y_pred, Y, dates, out_dir):
    # find first day
    assert(len(Y_pred) == len(Y))
    assert(len(Y_pred) == len(dates))
    assert(len(Y) == len(dates))
        
    save_path = out_dir + '/plots/'
    os.makedirs(save_path, exist_ok=True)

    idx = find_next_day_from_results(dates)
    while idx < len(Y_pred):
        day_Y_pred = Y_pred[idx]
        day_Y = Y[idx]
        date = dates[idx][0]

        _plot_day(day_Y_pred, day_Y, date, save_path)
        idx = find_next_day_from_results(dates, idx+1)

def running_loss_plot(baseline_loss, NILM_loss, out_dir):
    save_path = out_dir + '/loss_plot.png'

    fig, ax = plt.subplots()

    ax.plot(baseline_loss, '-', color='r', label='Baseline loss')
    ax.plot(NILM_loss, '-', color='b', label='NILM loss')
    ax.legend()

    plt.suptitle('loss')
    plt.savefig(save_path, dpi=300)
    plt.close()
