import numpy as np
from sklearn.preprocessing import MinMaxScaler

def pre_process(data_raw):

    readings_raw = data_raw[0]
    months_raw = data_raw[1]
    weekdays_raw = data_raw[2]
    hours_raw = data_raw[3]

    readings_normalized, min_x, max_x = _min_max_normalize(readings_raw)
    months = _onehot_encoder(months_raw, 12)
    weekdays = _onehot_encoder(weekdays_raw, 7)
    hours = _onehot_encoder(hours_raw, 24)

    samples = [ np.concatenate((E, M, D, I)) for (E, M, D, I)
        in zip(readings_normalized, months, weekdays, hours) ]
   
    return (samples, min_x, max_x)

def _onehot_encoder(data, cardinality):
    return[ _onehot_encoder_elem(x, cardinality) for x in data ]

def _onehot_encoder_elem(element, cardinality):
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
    return scaled_data, min_x, max_x
