import torch
import numpy as np
import pandas as pd

from modules.data_processing.pre_processing import pre_process

def create_input_matrix(data_raw, time_steps, no_targets=False) :
    if no_targets :
        return _input_matrix_no_targets(data_raw, time_steps)
    else :
        return _input_matrix_with_targets(data_raw, time_steps)

# creates an input matrix without targets, assumes only 1 day needs to be 
# predicted and also assumes the input data is correct
def _input_matrix_no_targets(data_raw, time_steps) :
    
    readings_raw = data_raw[0]
    datetime_raw = data_raw[1]

    samples = pre_process(data_raw)
    
    # assume that we only need to predict one day,
    # and that the input is correct
    assert(len(samples) == time_steps)

    return samples

# creates an input matrix with targets
def _input_matrix_with_targets(data_raw, time_steps) :

    # the raw data contains the readings and the date_time
    readings_raw = data_raw[0]
    datetime_raw = data_raw[1]

    # pre process the data for the samples
    samples = pre_process(data_raw) 

    inputs = []
    targets = []
    target_dates = []

    idx = 0
    while idx < len(datetime_raw) :
       
        # check if we can make a target and input, (X and Y values)
        # if so make them, and add them to the input list
        if _target_exists(readings_raw, datetime_raw, idx) and _input_exists(readings_raw, datetime_raw, time_steps, idx) :
               
            input_samples = _make_input(samples, datetime_raw, time_steps, idx)
            target_samples, dates = _make_target(readings_raw, datetime_raw, idx)

            inputs.append(input_samples)
            targets.append(target_samples)
            target_dates.append(dates)

        idx = _next_day(datetime_raw, idx) 

    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    targets = torch.tensor(np.array(targets), dtype=torch.float32)
    
    return inputs, targets, target_dates

def _make_input(samples, dates, time_steps, idx) : 
    # we need to be at midnight with current idx
    assert(_is_midnight(dates[idx]))

    # previous day 10 am needs to exist
    idx =  _prev_day_10am(dates, idx)
    assert(idx != -1)

    # start end need to be in bounds    
    start = idx - time_steps
    end = idx + 1
    assert(start >= 0)
    assert(end < len(samples))

    # return the samples for the input
    return [ x for x in samples[start:end] ]

def _make_target(readings, dates, idx) :
    # current idx must be at midnight
    assert(_is_midnight(dates[idx]))

    start = idx
    end = idx + 24
    assert(start >= 0)
    assert(end <= len(readings))
    assert(end <= len(dates))

    target = [ x for x in readings[start:end] ]
    target_dates = [ x for x in dates[start:end] ]

    return target, target_dates

# check if we can make a target at idx 
# we can make a target is we have 24 samples for that day, starting at 00:00
# and if none of the readings are nan
def _target_exists(readings, datetime, idx) :

    current_time = datetime[idx]


    if(idx + 23 >= len(datetime)) :
        return False

    midnight = pd.to_datetime('00:00')
    if(current_time.time() != midnight.time()) :
        return False
    
    day_end = datetime[idx+23]
    if(current_time.date() != day_end.date()) :
        return False
       
    return not np.isnan(readings[idx:idx+23]).any()

# we can make an input if the data is complete for the time_steps
def _input_exists(readings, dates, time_steps, idx) : 

    idx = _prev_day_10am(dates, idx)
    # can't find previous day 10 am
    if idx == -1 : 
        return False

    # we can't go back more time_steps than idx
    if time_steps > idx :
        return False

    # we go back time_steps * hours
    delta = pd.Timedelta(hours=time_steps)

    # check if we have complete data for the target 
    if dates[idx-time_steps] == dates[idx] - delta :
        input_readings = readings[idx-time_steps:idx+1]
        # if we don't have nan values then we can make an input
        return not np.isnan(input_readings).any()
    
    return False 
       
def _is_midnight(datetime) :
       
    return datetime.time() == pd.to_datetime('00:00').time()

def _prev_day_10am(dates, idx) :

    # dates[idx] needs to be at 00:00
    if (dates[idx].time() != pd.to_datetime('00:00').time()) :
       return -1 

    # target = previous day 10 am
    target = dates[idx] - pd.Timedelta(hours=14) 

    while idx >= 0 and target != dates[idx] :
        # decrement the idx until we find 10 am prev day
        idx = idx - 1

        # if we are before 10 am then we can't make the input for the target
        # because 10:00 doesn't exists
        if(idx >= 0 and target > dates[idx]) :
            return -1 

    return idx

def _next_day(datetime_raw, idx) :

    day_delta = pd.Timedelta(days=1)
    current_day = datetime_raw[idx]
    next_day = datetime_raw[idx] + day_delta

    # in case data for the day is complete we can jump 24
    if _is_midnight(current_day) and (idx + 24) < len(datetime_raw) and (next_day == datetime_raw[idx+24]) :
       return idx + 24
    
    # either we are not at midnight, or the current day is not complete
    # so we increment until we are at midnight
    idx = idx + 1
    while idx < len(datetime_raw) and not _is_midnight(datetime_raw[idx]) :
        idx = idx + 1
    return idx
