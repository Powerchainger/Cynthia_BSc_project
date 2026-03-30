import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


def run_acf(df, out_dir):
    """computes the autocorrelation function for all the columns in a df 
    
    This function computes the acf for all the columns in a dataframe,
    except for time. Furthermore, the max lags in a weeks time is also
    computed and saved to a .txt file. The max lags were used to 
    determine the timesteps of the models.

    Keyword arguments:
    df -- the dataframe to compute the acf for
    out_dir -- the directory to save the results to 
    """
    path = out_dir + '/acf_plots/'
    os.makedirs(path, exist_ok=True)
    
    cols = [ col for col in df.columns if col != 'time' ]
    weeklen = 24 * 7

    max_lags = {}
    for col in cols:

        fig = plot_acf(df[col], lags=weeklen, missing='drop')

        fig.suptitle(col)

        plot_path = path + col
        plt.savefig(plot_path, dpi=300)
        plt.close()

        result = acf(df[col], nlags=weeklen, missing='drop')
        max_lags[col] = 24 + pd.Series(result[24:]).idxmax()

    save_path = path + 'max_lags.txt'
    with open(save_path, 'w') as f:
        for key, val in max_lags.items():
            print(f'{key}: {val}', file=f)

def run_pacf(df, out_dir):
    """Runs the partial autocorrelation function"""
    path = out_dir + '/pacf_plots/'
    os.makedirs(path, exist_ok=True)
    
    cols = [ col for col in df.columns if col != 'time' ]
    weeklen = 24 * 7


    for col in cols:

        df.loc[pd.isna(df[col]), col] = 0
        fig = plot_pacf(df[col], lags=weeklen)

        fig.suptitle(col)

        plot_path = path + col
        plt.savefig(plot_path, dpi=300)
        plt.close()

def load_df(path):
    """Loads a csv file into a pandas dataframe"""
    df = pd.read_csv(path, parse_dates=['time'])
    
    return df

def main():
    """Computes acf and pacf for a csv file
    
    Program arguments:
    csv_path -- first arg, path to the csv for which to run acf and pacf
    out_dir -- second arg, path to the output directory
    """ 

    if (len(sys.argv) < 3):
        print('Error: please give path to csv and output dir as arg')
        exit(1)
    
    csv_path = sys.argv[1]
    out_dir = sys.argv[2]
    
    df = load_df(csv_path)
    run_acf(df, out_dir) 
    run_pacf(df, out_dir)

if __name__ == '__main__':
    main()
