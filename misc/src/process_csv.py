import sys
import pandas as pd
import numpy as np

# function that processes a csv from powerchainger, turns the readings from 
# every second into hourly average consumption for main and all appliances
def process(df) :

    # create a new df with the same columns as the previous one
    new_df = pd.DataFrame(columns=df.columns)

    # time series by hour, assumes complete data 
    time_begin = df['time'].min().round(freq='h')
    time_end = df['time'].max().round(freq='h')
    time_range = pd.date_range(start=time_begin,
                               end=time_end,
                               freq='h')

    for hr in time_range : 
        # we grab the average consumption at hr, meaning we grab the total
        # consumption starting at hour ending at hour+1, and average that
        begin = hr 
        end = hr + pd.Timedelta(hours=1) 
        time_mask = (df['time'] >= begin) & (df['time'] < end)
        
        # all the readings from hr - 30 mins to hr + 30 mins
        readings = df[time_mask].loc[:, df.columns != 'time']
        # drop the readings with nan in the main 
        readings = readings.dropna(subset = ['main'])

        # compute the avg consumption in kwh for all appliances
        avg = readings.sum() / len(readings)
        avg_kwh = avg / 1000

        # the row consists of the time + the average usage in kwh for the hour
        row = [hr] + avg_kwh.tolist()
        new_df.loc[len(new_df)] = row   

    return new_df 

def main() :
    # need a csv to read, and a path to write new csv to
    if (len(sys.argv) < 3) :
        print('Error: too few arguments to run, 2 args needed')
        exit(1)

    file_path_read = sys.argv[1]
    file_path_write = sys.argv[2]

    try :
       csv_df = pd.read_csv(file_path_read, parse_dates=['time'], na_values='')
    except :
        print(f'Error: could not read csv:\'{file_path_read}\'')
        exit(1)

    # process the dataframe 
    df_out = process(csv_df)

    try :
        df_out.to_csv(file_path_write, index=False)
    except :
        print(f'Error: could not write csv to path:\'{file_path_write}\'')
        exit(1)

if __name__ == '__main__' :
    main()
