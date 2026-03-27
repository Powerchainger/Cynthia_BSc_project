import torch
import pandas as pd
import numpy as np

from modules.io.json_loaders import JSON_model_params_loader 
from modules.model.forecaster import Forecaster as Model


def load_model(file_path, model_params):
    """Function that loads a model
    
    Keyword arguments:
    file_path -- path to the .pt file which contains model weights
    params -- hyper-params of the model

    Returns:
    The model initialized by the params, containing the weigths from
    the .pt file
    """

    model = Model(*(model_params.to_tuple()))
    model.load_state_dict(torch.load(file_path, weights_only=True))
    return model

def load_data(file_path):
    """Function that loads training, validation, testing data from csv 

    Keyword arguments:
    file_path -- path to the .csv file which contains the data

    Returns:
    a tuple containing the training, validation, testing data and the 
    list of appliances in the csv and a a dictionary with the minimum 
    and maximum values for each numerical feature that the models can 
    use (main, appliances)
    """ 
    csv_df = pd.read_csv(file_path, parse_dates=['time'])
    appliances = [ 
        name 
        for name 
        in csv_df.columns 
        if name != 'time' and name != 'main' 
    ]

    return (
        split_dataset(csv_df),
        appliances,
        _create_min_max_dictionary(csv_df)
    )

def split_dataset(df, ratio=0.6): 
    """Function that splits dataframe into 3 dataframes 
    
    This is a helper function for load_data, it splits the dataframe 
    into training, validation, and testing dataframes.

    Keyword arguments:
    df -- the dataframe to split into training,validation,testing
    ratio -- the ratio in which the data will be split, for example
    0.6 gives a ratio of 0.6 training data, 0.2 validation data, and
    0.2 testing data (default = 0.6)

    Returns:
    a tuple (training, validation, testing), containing the split data
    """
    assert(ratio > 0 and ratio < 1) # can't split if this is false

    data_len = len(df)
    training_end = int(data_len*ratio)
    validation_end = training_end + int((data_len - training_end) / 2)
    testing_end = data_len

    training = df.iloc[0:training_end]
    training = training.reset_index(drop=True)

    validation = df.iloc[training_end:validation_end]
    validation = validation.reset_index(drop=True)

    testing = df.iloc[validation_end:testing_end]
    testing = testing.reset_index(drop=True)

    return (training, validation, testing)

def _create_min_max_dictionary(df):
    """ Creates a dictionary of min_max values 

    For every column other than time, it will create a dictionary of 
    the minimum and maximum values of the columns, which is to be used
    later in training/testing/running of the models.

    Keyword arguments:
    df -- dataframe to create a min_max dictionary from

    Returns:
    min_max_dict -- the min max dictionary
    """
    min_max_dict = {}

    columns = [ name for name in df.columns if name != 'time' ]

    for col in columns:
        col_min = np.nanmin(df[col])
        col_max = np.nanmax(df[col])
        
        min_max_dict[col] = (col_min, col_max)
    
    return min_max_dict


# Unused, alternative way to load data from csv's

#def load_data_unused(file_path: str):
#
#    train_path = file_path + '/' + 'training.csv'
#    validation_path = file_path + '/' + 'validation.csv'
#    testing_path = file_path + '/' + 'testing.csv'
#
#    train_df = pd.read_csv(train_path, parse_dates=['time'])
#    val_df = pd.read_csv(validation_path, parse_dates=['time'])
#    test_df = pd.read_csv(testing_path, parse_dates=['time'])
#
#    appliances = [
#        name
#        for name
#        in train_df.columns
#        if name != 'time' and name != 'main'
#    ]
#
#    return (
#        (train_df, val_df, test_df),
#        appliances,
#        _create_min_max_dictionary(train_df)
#    )
