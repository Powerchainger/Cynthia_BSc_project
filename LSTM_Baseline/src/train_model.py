import sys
import torch

from modules.io.csv_config import Csv_config
from modules.io.save import save_model

from modules.model.forecaster import LSTM_forecaster as Model
from modules.model.trainer import LSTM_forecaster_Trainer as Trainer
from modules.model.config import Model_config

def main():
    # the args the module was run with, TODO: do args checking
    #argc = len(sys.argv)
    #argv = sys.argv

    # filepath = argv[1]
    # hiddenLayers = argv[2]
    # hiddenNodes = argv[3]
    # lookBack = argv[4]
    # individual = argv[5] 

    csvPath = '../dataset/training_data.csv'
    csvConfig = Csv_config(
        'READING_DATETIME',
        ' GENERAL_SUPPLY_KWH',
        48,
        '30min')

    modelConfig = Model_config(
        57, # input dim   
        20, # nodes per hidden layer
        2,  # hidden layers
        1,  # output dimension
        2,  # lookback
        torch.nn.MSELoss, #loss function
        torch.optim.Adam, #optimizer function
        150) # epochs

    # init the model
    model = Model(modelConfig)
    
    #init the trainer
    trainer = Trainer(
        model,
        modelConfig,
        csvPath,
        csvConfig)

    trainer.train()
    
    save_model(model,'../models/test.pickle')

# ensure that we only call main if we run the program directly
if __name__ == '__main__':
    main()
