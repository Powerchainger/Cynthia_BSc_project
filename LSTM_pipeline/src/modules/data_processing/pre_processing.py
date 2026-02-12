import numpy as np
from sklearn.preprocessing import MinMaxScaler

type DF = pandas.DataFrame
type arr = numpy.ndarray

#TODO: document and type suggestions

def _as_samples(readings, appliances_readings, weekdays, hours):
    
    samples = [ np.concatenate((main, day, hour)) for (main, day, hour) in zip (readings, weekdays, hours) ]
    
    for appliance_readings in appliances_readings :
        samples = [ np.concatenate((sample, appliance)) for (sample, appliance) in zip(samples, appliance_readings) ]

    return samples

def pre_process(df: DF, appliances, min_max_dict): 
    
    readings = df['main']
    dates = df['time']
    appliance_readings = [ np.array(df[appliance]) for appliance in appliances ] 
    
    readings = _min_max_normalize(readings, *min_max_dict['main'])
    appliance_readings = [ _min_max_normalize(readings, *min_max_dict[appliance]) for appliance, readings in zip(appliances, appliance_readings) ] 
    hours, weekdays = _encode_dates(dates)
  
    return _as_samples(readings, appliance_readings, hours, weekdays)

def _encode_dates(dates: DF) -> [[arr]]:

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

def _one_hot_encoder(element: int, cardinality: int) -> arr:
    vector = np.zeros(cardinality)
    vector[element] = 1

    return vector

# normalizer that uses min-max scaling to fit the data within the range [0..1]
def _min_max_normalize(data: DF, min_x, max_x) -> arr:
   
    scaled_data = (np.array(data) - min_x) / (max_x - min_x)

    scaled_data = scaled_data.reshape(-1, 1)
    return scaled_data
