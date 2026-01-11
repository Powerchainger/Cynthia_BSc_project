import numpy as np
from sklearn.preprocessing import MinMaxScaler

def pre_process(data_raw):

    readings_raw = data_raw[0]
    dates_raw = data_raw[1]

    readings = _min_max_normalize(readings_raw)
    dates = _encode_dates(dates_raw) 

    samples = [ np.concatenate((E, D, I)) for (E, D, I)
        in zip(readings, dates[1], dates[2]) ]
   
    return samples

def _encode_dates(dates) :

    months = []
    weekdays = []
    hours = []

    for date in dates :
        # get the month, weekday, and hour from the date
        month_raw = date.month - 1 # index month starts from 1 to 12
        weekday_raw = date.dayofweek
        hour_raw = date.hour
        
        month = _onehot_encoder(month_raw, 12)
        weekday = _onehot_encoder(weekday_raw, 7)
        hour = _onehot_encoder(hour_raw, 24)

        months.append(month)
        weekdays.append(weekday)
        hours.append(hour)

    return [ months, weekdays, hours ] 

def _onehot_encoder(element, cardinality):
    vector = np.zeros(cardinality)
    vector[element] = 1

    return vector

# normalizer that uses min-max scaling to fit the data within the range [0..1]
def _min_max_normalize(data):
    
    # first find the minimum and maximum
    min_x = np.nanmin(data)
    max_x = np.nanmax(data)

    data = np.array(data)
    scaled_data = (data - min_x) / (max_x - min_x)

    scaled_data = scaled_data.reshape(-1, 1)
    return scaled_data
