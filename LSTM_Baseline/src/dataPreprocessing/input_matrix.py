#TODO: import dependencies

def create_input_matrix(load_samples, time_indices, day_indices, holiday_marks):
    
    #normalize the data for the LSTM 
    normalized_load_samples = normalize(load_samples)
    onehot_time_indices = onehot_encoder(time_indices)
    onehot_day_indices = onehot_encoder(day_indices)
    onehot_holiday_marks = onehot_encoder(holiday_marks)

    #TODO:
    #make them a matrix in the form of X={E,I,D,H}

    #return X
