import numpy as np
from sklearn.preprocessing import MinMaxScaler

_READINGS_IDX = 0
_APPLIANCES_IDX = 1
_MONTHS_IDX = 2
_WEEKDAYS_IDX = 3
_HOURS_IDX = 4

_MONTHS_DIM = 12
_WEEKDAYS_DIM = 7
_HOURS_DIM = 24

def pre_process(data_raw) :

    readings_raw = data_raw[_READINGS_IDX]
    appliances_raw = data_raw[_APPLIANCES_IDX]
    months_raw = data_raw[_MONTHS_IDX]
    weekdays_raw = data_raw[_WEEKDAYS_IDX]
    hours_raw = data_raw[_HOURS_IDX]

    readings_normalized, min_x, max_x = _min_max_normalize(readings_raw)
    appliance_normalized = _normalize_appliances(appliances_raw)
    months = _onehot_encoder(months_raw, _MONTHS_DIM)
    weekdays = _onehot_encoder(weekdays_raw, _WEEKDAYS_DIM)
    hours = _onehot_encoder(hours_raw, _HOURS_DIM)

    samples = [ np.concatenate((E, M, D, I)) for (E, M, D, I)
        in zip(readings_normalized, months, weekdays, hours) ]
    
    #TODO might not work
    for appliance in appliance_normalized :
        samples = [ np.concatenate((sample, appliance)) for (sample, appliance)
            in zip(samples, appliance) ]
   
    return (samples, min_x, max_x)

def _onehot_encoder(data, cardinality) :
    return[ _onehot_encoder_elem(x, cardinality) for x in data ]

def _onehot_encoder_elem(element, cardinality) :
    vector = np.zeros(cardinality)
    vector[element] = 1

    return vector

# normalizer that uses min-max scaling to fit the data within the range [0..1]
def _min_max_normalize(data) :
    
    # first find the minimum and maximum
    min_x = np.nanmin(data)
    max_x = np.nanmax(data)

    data = np.array(data)
    scaled_data = (data - min_x) / (max_x - min_x)

    scaled_data = scaled_data.reshape(-1, 1)
    return scaled_data, min_x, max_x


# binary for now
def _normalize_appliances(appliances_raw) :
    appliances = []
    for appliance_readings in appliances_raw :
        appliance_readings = np.array(appliance_readings)

        appliance_binary = [ _binary_encode(x) for x in appliance_readings ]
        appliance_binary = appliance_readings.reshape(-1, 1)

        appliances.append(appliance_binary)

    return appliances

def _binary_encode(reading) :
    return 1.0 if reading > 0.0 else 0.0
