import torch
import os
# Function that saves a model to the given path.
# This is done by saving the weights of the model.
def save_model(model, out_dir, file_name) :
    dir_path = out_dir + '/models/'
    os.makedirs(dir_path, exist_ok=True)

    file_path = dir_path + file_name + '.pt'
    torch.save(model.state_dict(), file_path)

#TODO
def save_values(Y_pred, Y, dates, results_path):
    return
