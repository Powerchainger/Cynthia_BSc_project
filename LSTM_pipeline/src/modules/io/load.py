import torch
import pandas as pd
import numpy as np

from modules.io.model_params import Model_params
from modules.model.model import Forecaster as Model

def load_model(file_path: str, params: Model_params) -> Model:
    """ Function that initializes the model according to the Model_params given, then loads the weights from the file_path given, then returns the model."""

    model = Model(*(params.to_model_args()))
    model.load_state_dict(torch.load(file_path, weights_only=True))
    return model

type DF = pandas.DataFrame
def load_data(file_path: str) -> tuple[DF, DF, DF, [str]]:
    """ Function that loads the data from a csv, then splits it into training, validation, testing data.""" 
    csv_df = pd.read_csv(file_path, parse_dates=['time'])
    appliances = [ name for name in csv_df.columns if name != 'time' and name != 'main' ]

    return (split_dataset(csv_df), appliances, _create_min_max_dictionary(csv_df))

def split_dataset(df: DF, ratio: float=0.8) -> tuple[DF, DF, DF]:
    """helper function for load_data, splits the dataframe into an 80, 10, 10 split for the training, validation, and testing data respectively."""
    assert(ratio > 0 and ratio < 1)

    data_len = len(df)
    training_end = int(data_len*ratio)
    validation_end = training_end + int((data_len - training_end) / 2)
    testing_end = data_len

    training = df.iloc[0:training_end]
    validation = df.iloc[training_end:validation_end]
    testing = df.iloc[validation_end:testing_end]

    return (training, validation, testing)

#TODO type suggestions
def _create_min_max_dictionary(df):
    min_max_dict = {}

    columns = [ name for name in df.columns if name != 'time' ]

    for col in columns:
        col_min = np.nanmin(df[col])
        col_max = np.nanmax(df[col])
        
        min_max_dict[col] = (col_min, col_max)

    return min_max_dict


