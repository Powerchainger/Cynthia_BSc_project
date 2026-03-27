import numpy as np
from sklearn.preprocessing import MinMaxScaler


def _as_samples(readings, appliances_readings, weekdays, hours):
    """ Combines individual encoded features into a list of samples"""
    samples = [
        np.concatenate((main, day, hour))
        for (main, day, hour)
        in zip (readings, weekdays, hours) 
    ]
    
    for appliance_readings in appliances_readings :
        samples = [
            np.concatenate((sample, appliance))
            for (sample, appliance)
            in zip(samples, appliance_readings)
        ]

    return samples

def pre_process(df, appliances, min_max_dict): 
    """ Encodes the input for the models

    Every sample consists of the following features:
    [ M, D, H, A_0 ... A_b ]
    where:
    M   -- min-max normalized 'main'
    D   -- one-hot encoded weekday
    H   -- one-hot encoded hour of the day
    A_i -- min-max normalized 'appliance' for appliance i

    Keyword arguments: 
    df -- the raw data to encode
    appliance -- list of appliances to also encode
    min_max_dict -- dictionary containing min max values to be used for
    min-max normalization

    returns:
    A list of samples that can be used to create input tensors 
    """
    readings = df['main']
    dates = df['time']
    appliance_readings = [
        np.array(df[appliance])
        for appliance
        in appliances
    ] 
    
    readings = _min_max_normalize(readings, *min_max_dict['main'])
    appliance_readings = [
        _min_max_normalize(readings, *min_max_dict[appliance])
        for appliance, readings 
        in zip(appliances, appliance_readings) 
    ]

    hours, weekdays = _encode_dates(dates)
  
    return _as_samples(readings, appliance_readings, hours, weekdays)


def _encode_dates(dates):
    """Encodes the datetime objects for hour of the week and weekday"""
    weekdays = [] 
    hours = []

    for date in dates :
        # get the weekday, and hour from the date
        weekday_raw = date.dayofweek
        hour_raw = date.hour
        
        weekday = _one_hot_encoder(weekday_raw, 7)
        hour = _one_hot_encoder(hour_raw, 24)

        weekdays.append(weekday)
        hours.append(hour)

    return [ hours, weekdays ] 

def _one_hot_encoder(element, cardinality):
    """One-hot encodes a given element with given cardinality"""
    vector = np.zeros(cardinality)
    vector[element] = 1

    return vector

# normalizer that uses min-max scaling to fit the data within the range [0..1]
def _min_max_normalize(data, min_x, max_x):
    """ Min-max normalizes a pandas series object"""
    scaled_data = (np.array(data) - min_x) / (max_x - min_x)

    scaled_data = scaled_data.reshape(-1, 1)
    return scaled_data
