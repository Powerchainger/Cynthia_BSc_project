import numpy as np
from sklearn.preprocessing import MinMaxScaler

def pre_process_data(rawData):
   
    rawSamples = rawData[0]
    rawTimeIndex = rawData[1]
    rawWeekdayIndex = rawData[2]
    rawHolidayIndex = rawData[3]

    samples = normalize(rawSamples)
    timeIndex = onehot_encoder(rawTimeIndex)
    weekdayIndex = onehot_encoder(rawWeekdayIndex)
    holidayIndex = onehot_encoder(rawHolidayIndex)

    
    return [ samples, timeIndex, weekdayIndex, holidayIndex ] 

# one hot encoder:
def onehot_encoder(data):
    # first make sure the lowest value in the data is 0
    shifted_data = [ x + abs(min(data)) for x in data ]

    #cardianlity + 1 because we start from 0 and we need to index the max value
    cardinality = max(shifted_data) + 1 
    
    return[ onehot_encoder_elem(x, cardinality) for x in shifted_data ]

def onehot_encoder_elem(element, cardinality):
    vector = np.zeros(cardinality)
    vector[element] = 1

    return vector

# normalizer, uses min-max scaling
def normalize(data):
    data_min = min(data)
    data_max = max(data)

    normalized_data = [ (x - data_min) / (data_max - data_min) for x in data]
    return normalized_data

