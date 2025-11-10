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
LOOK_BACK = 2 

MODEL_PATH = '../models/'
CSV_PATH_TRAINING = '../dataset/training_data.csv'
CSV_PATH_VALIDATION = '../dataset/validation_data.csv'
CSV_PATH_TESTING = '../dataset/testing_data.csv'

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

def train_customer_model(customer_ID):
    customer_model_path = MODEL_PATH + 'customer_' + str(customer_ID) + '.pt'
    #TODO: data for customers in here too


    #init and train model
    model = Model(model_params)
    trainer = Trainer(
        model,
        model_params,
        csv_config,
        CSV_PATH_VALIDATION,
        CSV_PATH_TRAINING,
        EPOCHS)
    trainer.train()

    # save model
    print(f'saving model to :{customer_path}')
    torch.save(model.state_dict(), customer_path)

def main():
    #load in the customer IDS
    ids = #TODO: 
    for customer_id in ids:
        train_customer_model(customer_id)
    
 # ensure that we only call main if we run the program directly
if __name__ == '__main__':
    main()
