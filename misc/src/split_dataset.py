import sys
import pandas as pd
import numpy as np

# function that splits a dataset by dates, DOES NOT CHECK FOR ANY NANS
# 80% will be used for the training set
# 10% will be used for the validation set
# 10% will be used for the testing set
def split(df) :

    # calculate the lengths for an approximate 80%, 10%, 10% split
    total_len = len(df)
    training_len = int(total_len * 0.8) # 80%
    validation_len = int((total_len - training_len) / 2) # 10%
    testing_len = total_len - training_len - validation_len # 10%

    training = df.iloc[0:training_len]
    validation = df.iloc[training_len:training_len+validation_len]
    testing = df.iloc[training_len+validation_len:
        training_len+validation_len+testing_len]
    
    return (training, validation, testing)

def main() :
    # need a csv to read, and a dir to write new csv's to
    if (len(sys.argv) < 3) :
        print('Error: too few arguments to run, 2 args needed')
        exit(1)

    file_path_read = sys.argv[1]
    dir_path_write = sys.argv[2]

    try :
       csv_df = pd.read_csv(file_path_read, parse_dates=['time'])
    except :
        print(f'Error: could not read csv:\'{file_path}\'')
        exit(1)

    # process the dataframe 
    (training, validation, testing) = split(csv_df)

    try :
       training.to_csv(dir_path_write + 'training.csv', index=False)
       validation.to_csv(dir_path_write + 'validation.csv', index=False)
       testing.to_csv(dir_path_write + 'testing.csv', index=False)
    except :
        print(f'Error: could not write csv\'s to dir:\'{dir_path_write}\'')
        exit(1)

if __name__ == '__main__' :
    main()
