import torch
import numpy as np

# Function that prepares the input matrix for the forcaster model.
# the samples in data will be put in time_steps size groups, along with the 
# target output of said group.
def prepare_input_matrix(data, time_steps, no_targets=False) :
    # init the lists
    inputs = []
    targets = []
    
    end = len(data) 
    if not no_targets :
        target_dist = 38
        end = len(data) - target_dist 

    for idx in range(time_steps, end) :
        sequence, target = _prepare_sequence(data, time_steps, idx, no_targets)
        # check for any nan, if so skip it
        if not (np.isnan(np.min(sequence)) or np.isnan(np.min(target))):
            inputs.append(sequence)

            # only append targets if they are needed
            if not no_targets :
                targets.append(target)

    # return the inputs along with the targets, or just the inputs if
    # no_targets is set to true
    if no_targets :
        out = torch.tensor(np.array(inputs))
    else :
        out = (torch.tensor(np.array(inputs), dtype=torch.float32),
               torch.tensor(np.array(targets), dtype=torch.float32))
    
    return out

# helper function for prepare_input_matrix, creates a single sequence given
# an idx, and a single target
def _prepare_sequence(data, time_steps, idx, no_targets=False) :

    start = idx - time_steps
    end = idx

    target_start = idx + 14
    target_end = target_start + 24

    # assert that we are not out of bounds
    assert(idx >= time_steps)
    if no_targets :
        assert(idx < len(data))
    else :
        assert(target_end < len(data))

    # a sequence is just a list of samples of size time_steps
    sequence = [ x for x in data[start:end] ]

    target = None
    if not no_targets :
        # target is the next 24 hours from idx+14 
        #i.e. at 10am the target is from 00:00 next day to 23:00 next day
        target = [ x[0] for x in data[target_start:target_end] ] 

    return (sequence, target) 
