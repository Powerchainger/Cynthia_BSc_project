import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


houses = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21 ]
path = '../data/REFIT/avg_kwh_hourly/uncropped/house_'
file_extension = '.csv'

save_path = 'plots/time_intervals.png'

def house_interval(house):
    df = pd.read_csv(path + str(house) + file_extension, parse_dates['time'])

    return (df['time'].min().date(), df['time'].max().date())

def main():
    """I don't think this piece of code is particularly useful, but it 
    specifically was used to compute the window of time for which all
    the houses in refit had available data
    """
    fig, ax = plt.subplots()

    y_pos = np.arange(len(houses))
    y_height = 0.4

    for house, y_pos in zip(houses, y_pos):

        start_date, end_date = house_interval(house)
        
    
        ax.broken_barh


    ax.invert_yaxis()
    plt.save_fig(save_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
