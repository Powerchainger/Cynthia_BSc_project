import torch
import numpy as np
import pandas as pd

from modules.data_processing.pre_processing import pre_process

def create_input_matrix(data_raw, time_steps, no_targets=False) :
    if no_targets :
        return _input_matrix_no_targets(data_raw, time_steps)
    else :
        return _input_matrix_with_targets(data_raw, time_steps)


def _input_matrix_no_targets(data_raw, time_steps) :
    
    readings_raw = data_raw[0]
    datetime_raw = data_raw[1]

    samples = pre_process(data_raw)

    assert(len(samples) == time_steps)

    return samples

def _input_matrix_with_targets(data_raw, time_steps) :

    readings_raw = data_raw[0]
    datetime_raw = data_raw[2]

    samples = pre_process(data_raw)

    inputs = []
    targets = []
    target_dates = []

    idx = 0
    while idx < len(datetime_raw) :

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

    assert(_is_midnight(dates[idx]))

    idx = _prev_day_10am(dates, idx)
    assert(idx != -1)

    start = idx - time_steps
    end = idx + 1
    assert(start >= 0)
    assert(end < len(samples))

    return [ x for x in samples[start:end] ]

def _make_target(readings, dates, idx) :
    assert(_is_midnight(dates[idx]))
    
    start = idx
    end = idx + 24

    assert(start >= 0)
    assert( end <= len(readings))
    assert( end <= len(dates))

    target = [ x for x in readings[start:end] ]
    target_dates = [ x for x in dates[start:end] ] 
    return target, target_dates

def _target_exists(readings, datetime, idx) :

    current_time = datetime[idx]

    if (idx + 23 >= len(datetime)) :
        return False

    midnight = pd.to_datetime('00:00')
    if (current_time.time() != midnight.time()) :
        return False
    
    #TODO: this needs to be adapted to NILM
    return not np.isnan(readings[idx:idx+23]).any()

def _input_exists(readings, dates, time_steps, idx) :

    idx = _prev_day_10am(dates, idx)

    if idx == -1 :
        return False

    if time_steps > idx :
        return False
    
    delta = pd.Timedelta(hours=time_steps)

    if dates[idx - time_steps] == dates[idx] - delta :
        # TODO: adapt this for NILM
        input_readings = readings[idx-time_steps:idx+1]
        return not np.isnan(input_readings).any()

    return False

def _is_midnight(datetime) :
    
    return datetime.time() == pd.to_datetime('00:00').time()

def _prev_day_10am(dates, idx) :

    if (dates[idx].time() != pd.to_datetime('00:00').time()) :
        return -1

    target = dates[idx] - pd.Timedelta(hours=14)

    while idx >= 0 and target != dates[idx] :

        idx = idx - 1

        if (idx >= 0 and target > dates[idx]) :
            return -1 

    return idx

# Function that prepares the input matrix for the forcaster model.
# the samples in data will be put in time_steps size groups, along with the 
# target output of said group.

def _next_day(datetime_raw, idx) :

    day_delta = pd.Timedelta(days=1)
    current_day = datetime_raw[idx]
    next_day = datetime_raw[idx] + day_delta

    if _is_midnight(current_day) and (idx + 24) < len(datetime_raw) and (next_day == datetime_raw[idx+24]) :
        return idx + 24

    idx = idx + 1
    
    while idx < len(datetime_raw) and not _is_midnight(datetime_raw[idx]) :
        idx = idx + 1
    return idx

#def prepare_input_matrix(data, time_steps, no_targets=False) :
#    # init the lists
#    inputs = []
#    targets = []
#    
#    end = len(data) 
#    if not no_targets :
#        target_dist = 38
#        end = len(data) - target_dist 
#
#    for idx in range(time_steps, end) :
#        sequence, target = _prepare_sequence(data, time_steps, idx, no_targets)
#        # check for any nan, if so skip it
#        if not (np.isnan(np.min(sequence)) or np.isnan(np.min(target))):
#            inputs.append(sequence)
#
#            # only append targets if they are needed
#            if not no_targets :
#                targets.append(target)
#
#    # return the inputs along with the targets, or just the inputs if
#    # no_targets is set to true
#    if no_targets :
#        out = torch.tensor(np.array(inputs))
#    else :
#        out = (torch.tensor(np.array(inputs), dtype=torch.float32),
#               torch.tensor(np.array(targets), dtype=torch.float32))
#    
##    return out
#
# helper function for prepare_input_matrix, creates a single sequence given
# an idx, and a single target
#def _prepare_sequence(data, time_steps, idx, no_targets=False) :
#
#    start = idx - time_steps
#    end = idx
#
#    target_start = idx + 14
 #   target_end = target_start + 24
#
#    # assert that we are not out of bounds
#    assert(idx >= time_steps)
#    if no_targets :
#        assert(idx < len(data))
#    else :
#        assert(target_end < len(data))
#
#    # a sequence is just a list of samples of size time_steps
#    sequence = [ x for x in data[start:end] ]
#
#    target = None
#    if not no_targets :
#        # target is the next 24 hours from idx+14 
#        #i.e. at 10am the target is from 00:00 next day to 23:00 next day
#        target = [ x[0] for x in data[target_start:target_end] ] 
#
####    return (sequence, target) 
