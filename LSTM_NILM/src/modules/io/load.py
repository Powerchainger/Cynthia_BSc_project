import torch
import json
import pandas as pd
import numpy as np

from modules.model.model import Forecaster as Model

# Function that loads the params for a model from a JSON file
# it returns a touple containing the params with which a model
# can be initialized
# 
# The touple contains:
#  (hidden_layer_nodes, hidden_layers, time_steps) 
def load_model_params(file_path) :

    #try :
        # open the file 
    with open(file_path, 'r') as file: 
            # read the JSON
        data = json.load(file)
    #except :
    #    print(f'Could not read model params from file:{file_path}')
    #    exit(1)
   
    # grab the params from the JSON
    hidden_layer_nodes = data['hidden_layer_nodes']
    hidden_layers = data['hidden_layers']
    time_steps = data['time_steps']
    appliances = data['appliances'] # array

    # return the params as a touple
    return (hidden_layer_nodes, hidden_layers, time_steps), appliances

# Function that loads initializes a model according to params,
# then it loads the weights for said model from file_path.
# Lastly it returns the model
def load_model(file_path, params) :
    print(f'Loading model with params:{params} and weights from:\'{file_path}\'')
        
    #initialize the model according to the params
    print('Initializing model....')
    model = Model(*params)
    print('Done....')

    # load the weights of the model, stored in file_path
    # this fails if the weights saved belong to a model of different params
    print('Loading weights....')
    try :
        model.load_state_dict(torch.load(file_path, weights_only=True))
    except : 
        print(f'Error: could not load weights for model with params:{params} from file{file_path}')
        exit(1)
    print('Done....')

    return model

# Function that loads data from a CSV file
# Then formats it so it can be used as input for a model 
# 
# Note, the CSV file must be formatted according to the following:
#   column 1: Time, named time
#   column 2: Load in KWH, named main
#
# output format:
#   A list containing samples, where a sample is a list of
#       1. the current load 
#       2. the current month of the year    from 1 to 12
#       3. the current day of the week      from 1 to 7
#       4. the current hour of the day      from 0 to 23 
def load_data(file_path, appliances) :
    print(f'Loading data from CSV:\'{file_path}\'') 
    # open the csv
    try :
        csv_df = pd.read_csv(file_path, parse_dates=['time'])
    except :
        print(f'Error could not read CSV:\'{file_path}\'')

    # check if the appliances are in the index of the csv
    for appliance in appliances :
        if appliance not in csv_df.columns :
            print(f'{appliance} not in the index of csv. aborting....')
            exit(1)

    readings = []
    appliance_readings = [ [] for appliance in appliances ] 
    months = []
    weekdays = []
    hours = []
    # process row by row
    for _, row in csv_df.iterrows() :
        # get the time and load from the csv
        time = row['time']
        reading = np.nan if pd.isna(row['main']) else row['main']

        # get the month, day, and hour from the time
        month = time.month - 1 # index month starts from 1 to 1 
        day = time.dayofweek
        hour = time.hour

        # get the NILM data for the appliances and add it to the reading
        # lists
        for idx, appliance in enumerate(appliances) :
            appliance_reading = np.nan if pd.isna(row[appliance]) else row[appliance]    
            appliance_readings[idx].append(appliance_reading)
                
        # add sample to our lists
        readings.append(reading)
        months.append(month)
        weekdays.append(day)
        hours.append(hour)

    print('Done....') 
    return [ readings, appliance_readings, months, weekdays, hours ]  
