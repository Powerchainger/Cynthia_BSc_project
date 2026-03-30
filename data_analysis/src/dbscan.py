import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN


def load_daily_profiles(df):
    """Function that loads the daily consumption profile from dataframe

    The function does two things, first is to turn the dataframe into
    a list containing the daily consumption, which is also a list of 
    length 24. Secondly, it computes the daily average consumption, 
    which is needed for dbscan.

    Keyword arguments:
    df -- the dataframe to compute the average daily consumption from
    """


    start_date = df['time'].min()
    end_date = df['time'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    daily_profiles = []
    total_consumption = 0.0

    samples = []
    for date in date_range:
        day_readings = df.loc[(df['time'].dt.date == date.date()), 'main']

        if (len(day_readings) != 24 or pd.isna(day_readings).any()):
            continue

        day_sample = day_readings.to_list()
        total_consumption = total_consumption + sum(day_sample)
        samples.append(day_sample)

    daily_avg_consumption = total_consumption / len(date_range)
    
    return (daily_avg_consumption, samples)

def plot_clusters(clusters, data, name, out_dir):
    """Plots dbscan results, the outliers are plotted in black
    
    Keyword arguments:
    clusters -- the results from dbscan, contains the labels
    data -- the data corresponding to the clusters
    name -- the name of the file containing the plots to be saved
    out_dir -- the directory to save the plots to
    """
    labels = clusters.labels_
    unique_labels = set(labels)
    n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)

    colors = [plt.cm.hsv(each)
        for each in np.linspace(0, 1, len(unique_labels))]

    desc = []
    for label in unique_labels:
        desc.append('cluster_%d' % label if label != -1 else 'outliers')

    lines = []
    for label in unique_labels:
        if(label != -1):
            lines.append(Line2D([0], [0], color=colors[label], lw=1))
        else:
            lines.append(Line2D([0], [0], color=[0, 0, 0, 1], lw=1))
    
    fig, ax = plt.subplots()
    fig.set_figwidth(9.6)
    for i in range(0, len(data)):
        day = data[i]
        label = labels[i]

        if(label == -1):
            color = [0, 0, 0, 1]
        else:
            color = colors[label]
        
        x = list(range(0, 24))
        y = day

        ax.plot(x, y,
            color = tuple(color),
            alpha = 0.2)

    ax.legend(lines, desc, loc='upper right')
    title = 'DBSCAN for ' + name
    plt.title(title)
    plt.xlabel('Time index')
    plt.ylabel('Energy Consumed [kWh]')

    save_path = out_dir + '/' + name + '.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

def dbscan(df, out_dir, file_out):
    """Runs dbscan given a pandas dataframe
    
    First the average consumption per day is computed and the days 
    are turned into a 24 lenght list by daily_avg_consumption
    Then dbscan is ran and plotted

    Keyword arguments:
    df -- the dataframe to run dbscan on
    out_dir -- the directory to save the results to
    file_out -- the name of the file containing the results, to be saved
    to out_dir
    """
    daily_avg_consumption, samples = load_daily_profiles(df)

    clusters = DBSCAN(
        eps=0.1*daily_avg_consumption, min_samples=2,
        metric='euclidean').fit(samples)
    plot_clusters(clusters, samples, file_out, out_dir)


def main():
    """Runs dbscan given a csv file, and an output directory

    Program arguments:
    csv_path -- first arg, path to the csv file to run dbscan on
    out_dir -- second arg, path to the directory to save the results to
    name_out -- name of the file containing dbscan plots, which is saved
    in out_dir
    """
    if (len(sys.argv) < 4):
        print('Error: please give path to csv and output dir as arg, and file name')
        exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2]
    name_out = sys.argv[3]

    df = pd.read_csv(csv_path, parse_dates=['time'])
    dbscan(df, out_dir, name_out)

if __name__ == '__main__':
    main()
