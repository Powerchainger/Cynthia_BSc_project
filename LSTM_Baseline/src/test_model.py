import sys
import torch

from modules.io.csv_config import Csv_config

from modules.model.model import Model 
from modules.model.model_params import Model_params

from modules.model.tester import Tester 

INPUT_DIM = 57
HIDDEN_DIM = 20
LAYER_DIM = 2 
LOOK_BACK = 2 

MODEL_PATH = '../models/test.pt'
CSV_PATH_TRAINING = '../dataset/training_data.csv'
CSV_PATH_VALIDATION = '../dataset/validation_data.csv'
CSV_PATH_TESTING = '../dataset/testing_data.csv'

def main():
    # the args the module was run with, TODO: do args checking
    #argc = len(sys.argv)
    #argv = sys.argv

    # filepath = argv[1]
    # hiddenLayers = argv[2]
    # hiddenNodes = argv[3]
    # lookBack = argv[4]
    # individual = argv[5] 

    csv_path = CSV_PATH_TESTING 
    csv_config = Csv_config(
        'READING_DATETIME',
        ' GENERAL_SUPPLY_KWH',
        48,
        '30min')

    model_params = Model_params(
        INPUT_DIM, # input dim   
        HIDDEN_DIM, # nodes per hidden layer
        LAYER_DIM,  # hidden layers
        LOOK_BACK)  # lookback

    # init the model
    model = Model(model_params)
   
    # load the weights
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    # test the model on the dataset
    tester = Tester(model, model_params, csv_path, csv_config)
    tester.run()

if __name__ == '__main__':
    main() 
