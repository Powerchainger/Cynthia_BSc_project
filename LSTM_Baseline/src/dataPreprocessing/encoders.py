import numpy as np
from sklearn.preprocessing import MinMaxScaler

# one hot encoder:
def onehot_encoder(data):
    # first make sure the lowest value in the data is 0
    shifted_data = [ x + abs(min(data)) for x in data ]

    cardinality = max(shifted_data)
    
    return[ one_hot_encoder_elem(x, cardinality) for x in shifted_data ]

def onehot_encoder_elem(element, cardinality):
    vector = np.zeros(cardinality)
    vector[element] = 1

    return vector

# normalizer, uses min-max scaling
def normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

