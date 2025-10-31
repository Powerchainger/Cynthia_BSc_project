import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from DBSCAN_individual import plot_clusters
#perform DBSCAN on the datafiles

sample_size = 48

def DBSCAN_aggregate(filepath):

    daily_avg_consumption, readings = load_aggregate_profiles(filepath)
    clusters= DBSCAN(
        eps=0.1*daily_avg_consumption,
        metric='euclidean').fit(readings)
    plot_clusters(clusters, readings)

def load_aggregate_profiles(filepath):
    csv_df = pd.read_csv(filepath, parse_dates=['READING_DATETIME'])

    start_date = csv_df['READING_DATETIME'].min()
    end_date = csv_df['READING_DATETIME'].max()
    date_range= pd.date_range(start=start_date, end=end_date, freq='D')

    total_consumption = 0.0
    aggregate_daily_profiles = []
    for day in date_range:
        #collect aggregate load for current day
        time_range = pd.date_range(start=day, periods = sample_size, freq='30min')
        day_readings = csv_df[csv_df['READING_DATETIME'].dt.date == day.date()]

        day_profile = []
        for time in time_range:
            # every 30 mins is a reading and every 48 readings makes a day
            all_current_readings = day_readings[
                day_readings['READING_DATETIME'] == time]
            aggregate_load = all_current_readings[' GENERAL_SUPPLY_KWH'].sum()

            day_profile.append(aggregate_load)
            total_consumption = total_consumption + aggregate_load
        
        aggregate_daily_profiles.append(day_profile)

    daily_avg_consumption = total_consumption / len(date_range)
    
    return (daily_avg_consumption, aggregate_daily_profiles)

#DBSCAN_aggregate('../../dataset/all_data.csv')
        

