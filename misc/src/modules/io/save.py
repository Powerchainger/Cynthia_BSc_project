import torch

# Function that saves a model to the given path.
# This is done by saving the weights of the model.
def save_model(model, path) :
    print(f'saving model to: \'{path}\'')
    torch.save(model.state_dict(), path)
    print(f'Done....')
