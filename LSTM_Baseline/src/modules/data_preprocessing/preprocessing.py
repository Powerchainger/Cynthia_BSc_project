import numpy as np
from sklearn.preprocessing import MinMaxScaler

def pre_process_data(data_raw):
   
    readings_raw = data_raw[0]
    time_idx_raw = data_raw[1]
    weekday_idx_raw = data_raw[2]
    holiday_idx_raw = data_raw[3]

    readings_normalized, scaler = _normalize(readings_raw)
    time_idx = _onehot_encoder(time_idx_raw)
    weekday_idx = _onehot_encoder(weekday_idx_raw)
    holiday_idx = _onehot_encoder(holiday_idx_raw)
   
    samples = [ np.concatenate((E, I, D, H)) for (E, I, D, H)
        in zip(readings_normalized, time_idx, weekday_idx, holiday_idx) ]
    
    return samples, scaler

def _onehot_encoder(data):
    # first make sure the lowest value in the data is 0
    shifted_data = [ x + abs(min(data)) for x in data ]

    #cardianlity + 1 because we start from 0 and we need to index the max value
    cardinality = max(shifted_data) + 1 
    
    return[ _onehot_encoder_elem(x, cardinality) for x in shifted_data ]

def _onehot_encoder_elem(element, cardinality):
    vector = np.zeros(cardinality)
    vector[element] = 1

    return vector

# normalizer, uses min-max scaling
def _normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(np.array(data).reshape(-1, 1)), scaler
