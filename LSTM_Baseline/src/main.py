# 1: pre-process data
# 2: train LSTM on data
# 3: run LSTM

# need functions for:
#   - train model
#   - run model
#   - save model
#   - prepare data
#   - metrics on how the model is doing

# metrics are defined as follows:
#   - Avg. MAPE individiual forecast
#   - Avg. MAPE Aggregating forecast
#   - Avg. MAPE forecasting the aggregate

# Pre processing:
#   1. isolate subset of used data in Kong et al.   [done]
#   2. Run DBSCAN on the data                       [todo] 
#   3. Encode data for the model                    [todo]

# Model:
#   1. train model                                  [todo]
#   2. load/save model                              [todo]

# Metrics:
#   1. Avg. MAPE individual forecast                [todo]
#   2. Avg. MAPE Aggregating forecast               [todo]
#   3. Avg. MAPE forecasting the aggregate          [todo]

import sys
import torch
import torch.nn as nn
import torch.optim as opmtim
from model/forecast_LSTM import LSTMForecaster as Model 
from model/trainer import modelTrainer as Trainer

def main():
    # the args the module was run with, TODO: do args checking
    #argc = len(sys.argv)
    #argv = sys.argv

    # filepath = argv[1]
    # hiddenLayers = argv[2]
    # hiddenNodes = argv[3]
    # lookBack = argv[4]
    # individual = argv[5] 

    filepath = '../dataset/training_data.csv'
    hiddenLayers = 2
    hiddenNodes = 20 
    lookBack = 2
    input_dim = lookBack ** 4

    # init the model
    model = Model(input_dim, hiddenNodes, hiddenLayers, 1)
    
    #init the trainer
    trainer = Trainer(
        model,
        torch.nn.NLLLoss(),
        torch.optim.Adam,
        filepath)

    # for now, we just train
    trainer.train()

# ensure that we only call main if we run the program directly
if __name__ == '__main__':
    main()
