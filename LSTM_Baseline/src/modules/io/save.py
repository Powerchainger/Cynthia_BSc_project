import torch

def save_model(model, path):
    print('saving model to: ' + path)
    torch.save(model.state_dict(), path)
