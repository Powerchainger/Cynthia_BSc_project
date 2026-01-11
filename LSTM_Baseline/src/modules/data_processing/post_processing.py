import torch

# function that performs the inverse of min max normalization,
# works on tensors such that it can be used as post_processing before computing
# the loss for training models
def scale_tensor(tensor, min_x, max_x):
    tensor = tensor * (max_x - min_x) + min_x
    return tensor
