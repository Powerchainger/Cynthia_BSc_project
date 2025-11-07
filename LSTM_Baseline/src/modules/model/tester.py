import torch
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from modules.model.forecaster import LSTM_forecaster
from modules.model.config import Model_config 

from modules.io.csv_config import Csv_config
from modules.io.load import load_data_from_csv 
from modules.data_preprocessing.preprocessing import pre_process_data

class LSTM_forecaster_tester():
    def __init__(self, model, modelConfig, filePath, csvConfig):
        
        self.model = model
        self.modelConfig = modelConfig
        self.epochs = modelConfig.epochs 
        self.lookBack = modelConfig.lookBack
        self.lossFunction = modelConfig.lossFunction
        self.optimizer = modelConfig.optimizer

        self.filePath = filePath
        self.data = self.init_data(filePath, csvConfig) 

    def init_data(self, filePath, csvConfig):
        rawData = load_data_from_csv(filePath, csvConfig)
        normalizedData = pre_process_data(rawData)
        return normalizedData

    # The input matrix is as follows:
    # X = { E', I', D', H'}, where: 
    #   - E' is the sequence of energy consumptions for lookBack time steps
    #   - I' is the corresponding time day indices for lookBack time steps
    #   - D' is the corresponding day of week indices for lookback steps
    #   - H' is the corresponding holiday markers for lookback steps
    #   
    #   - To get E' we normalize E to fit the range [0..1]
    #   - I' D' H' are encoded by a one hot encoder
    def prepare_sample(self, idx):

        E = self.data[0][idx]
        I = self.data[1][idx]
        D = self.data[2][idx]
        H = self.data[3][idx]

        return np.concatenate((E, I, D, H))

    def prepare_sequence(self, idx):
        if (idx < self.lookBack):
            #TODO: throw error here
            print('error: idx < lookBack, can\'t prepare sequence')
        if (idx > len(self.data[0])):
            #TODO: also error here
            print('error: idx > len(data), can\'t prepare sequence')
        
        # prepare the sequence
        sequence = []
        for sequence_idx in range(idx - self.lookBack, idx):
            sequence.append(self.prepare_sample(sequence_idx))

        # prepare the target
        target = [self.data[0][idx]]

        return (sequence, target)

    def prepare_input_matrix(self):
        inputs = []
        targets = []

        # prepare the input matrix for the dataset
        for idx in range(self.lookBack, len(self.data[0])):
            sequence, target = self.prepare_sequence(idx) 
            inputs.append(sequence)
            targets.append(target)

        # converting a list of numpy arrays is extremely slow, so we convert
        # them to a single numpy array before making the tensors 
        return (torch.tensor(np.array(inputs)), torch.tensor(np.array(targets)))

    def test(self):
        print('starting testing LSTM with testing data: ' + self.filePath)
    
        #initialize model, and functions
        torch.set_default_dtype(torch.float64) 
        model = self.model.to(torch.float64)
        lossFunction = self.lossFunction()
        optimizer = self.optimizer(model.parameters())

        # prepare the inputs for the model
        input_matrix, targets = self.prepare_input_matrix()

        # test the model 
        model.eval()
        with torch.no_grad():
            forecasts, _, _ = model(input_matrix) 
       
        for forecast, target in zip(forecasts, targets):
            print(f'forecast:{forecast.item():.5f}\t actual:{target.item():.5f}')

        MAPE = self.computeMAPE(forecasts, targets)

        print(f'MAPE for model, on dataset:{self.filePath} = {MAPE}')


    #TODO: this is bad
    def computeMAPE(self, forecasts, actual):
        #first we de-normalize, we do something stupid here
        forecasts = forecasts  
        actual = actual  
    
        #TODO: find a better way to do this
        forecasts_reshaped = []
        actual_reshaped = []
        for forecast, value in zip(forecasts, actual):
            if(value.item() != 0.0):
                actual_reshaped.append(value.item())
                forecasts_reshaped.append(forecast.item())

        return mean_absolute_percentage_error(actual_reshaped, forecasts_reshaped) 
