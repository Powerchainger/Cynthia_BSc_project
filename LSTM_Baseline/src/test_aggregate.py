import sys
import torch

from modules.io.csv_config import Csv_config
from modules.io.save import save_model

from modules.model.model import Model 
from modules.model.model_params import Model_params

from modules.model.trainer import Trainer 

EPOCHS = 150 # TODO: remove these constants
INPUT_DIM = 57
HIDDEN_DIM = 20
LAYER_DIM = 2 
LOOK_BACK = 12 

MODEL_PATH = '../models/test.pt'
CSV_PATH_TRAINING = '../dataset/training_data.csv'
CSV_PATH_VALIDATION = '../dataset/validation_data.csv'
CSV_PATH_TESTING = '../dataset/testing_data.csv'

def main():
    # the args the module was run with, TODO: do args checking
    #argc = len(sys.argv)
    #argv = sys.argv

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
    
    #init the trainer
    trainer = Trainer(
        model,
        model_params,
        csv_config,
        CSV_PATH_VALIDATION,
        CSV_PATH_TRAINING,
        EPOCHS)

    trainer.train()

    print(f'saving model to :{MODEL_PATH}')
    torch.save(model.state_dict(), MODEL_PATH)

# ensure that we only call main if we run the program directly
if __name__ == '__main__':
    main()
