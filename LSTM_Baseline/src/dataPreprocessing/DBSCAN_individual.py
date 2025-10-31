import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sample_size = 48

# TODO:
# make it so the scan outputs can be fed into the LSTM
def DBSCAN_individual(filepath) :
    # load the daily profiles,
    # the profiles are a list containing:
    #   A list where the first item is the CUSTOMER_ID
    #   and the second item is a list containing samples corresponding to each day
    daily_profiles = load_daily_profiles(filepath)

    for [ customer_id, daily_avg_consumption, readings]  in daily_profiles :
        clusters = DBSCAN(
            eps=0.1*daily_avg_consumption, min_samples=2,
            metric='euclidean').fit(readings)
        plot_clusters(clusters, readings, customer_id)

def load_daily_profiles(filepath) :
    csv_df = pd.read_csv(filepath, parse_dates=['READING_DATETIME'])

    # CUSTOMER_IDS
    customer_ids = csv_df['CUSTOMER_ID'].unique().tolist()

    # range of the dates in the csv
    start_date = csv_df['READING_DATETIME'].min()
    end_date = csv_df['READING_DATETIME'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    daily_profiles = []
    total_consumption = 0.0 
    for customer_id in customer_ids :
        # all the entries in the csv corresponding to this customer
        customer_readings = csv_df[csv_df['CUSTOMER_ID'] == customer_id]

        samples = []
        print('loading data for: %d' % customer_id)
        customer_consumption = 0
        for date in date_range:
            day_readings = customer_readings[
                    (customer_readings['READING_DATETIME'].dt.date == date.date())]
 
            # we have the readings for the current date, first we assert that
            # the data for this day has 48 samples
            # if it has not we drop this day for DBSCAN

            if(day_readings.shape[0] != sample_size) :
                print('for customer %d, skip loading day: ' % customer_id)
                print(date.date())
                continue
            
            # now we turn the daily readings into a list of len=sample_size
            day_load = day_readings[' GENERAL_SUPPLY_KWH']
            day_sample = day_load.to_list()
            customer_consumption = customer_consumption + day_load.sum()
            samples.append(day_sample)

        daily_avg_consumption = customer_consumption / len(date_range)
        daily_profiles.append([ customer_id, daily_avg_consumption, samples ])

    return daily_profiles

def plot_clusters(clusters, data, customer_id) :

    labels = clusters.labels_
    unique_labels = set(labels)
    n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)

    colors = [plt.cm.hsv(each)
        for each in np.linspace(0, 1, len(unique_labels))]
    
    # description for the legend
    desc = []
    for label in unique_labels:
        desc.append('cluster_%d' % label if label != -1 else 'outliers')

    # line for the legend
    lines = []
    for label in unique_labels:
        if(label != -1):
            lines.append(Line2D([0], [0], color=colors[label], lw=1))
        else:
            lines.append(Line2D([0], [0], color=[0, 0, 0, 1], lw=1))

    # for the information while it's running
    print('Plotting readings for customer: %d' % customer_id)
    print('Amount of clusters: %d' % n_clusters_)

    fig, ax = plt.subplots()
    fig.set_figwidth(9.6)
    for i in range(0,len(data)) :
        day = data[i]
        label = labels[i]
        
        if(label == -1) :
            #outlier day
            color = [0, 0, 0, 1]
        else :
            color = colors[label]
        
        x = list(range(1, 49))
        y = day

        ax.plot(x, y,
            color=tuple(color),
            alpha=0.2)


    ax.legend(lines, desc, loc='upper right')
    plt.title('daily profiles for Customer %d' % customer_id)
    plt.xlabel('Time Index')
    plt.ylabel('Energy Consumed [kWh]')

    plt.savefig('../../figures/customer%d.png' % customer_id)
    plt.close()

#load_daily_profiles('../../dataset/training_data.csv')
DBSCAN_individual('../../dataset/all_data.csv')

