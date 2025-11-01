import torch
import numpy as np

from modules.model.forecaster import LSTM_forecaster
from modules.model.config import Model_config 

from modules.io.csv_config import Csv_config
from modules.io.load import load_data_from_csv 
from modules.data_preprocessing.preprocessing import pre_process_data

class LSTM_forecaster_Trainer():
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

        sample = [E]
        sample.extend(I)
        sample.extend(D)
        sample.extend(H)

        return sample

    def prepare_batch(self, epoch, batchSize):

        idx = epoch * batchSize 

        batch = []
        targets = []
        for _ in range(batchSize):
            sequence = []
            target = [] 
            for i in range(self.lookBack):
                sample = self.prepare_sample(idx + i)
                sequence.append(sample)
                target = [self.data[0][idx + i + 1]] 

            # the target for the sequence is the consumption after the sequence
            targets.append(target)
            batch.append(sequence)

            idx = idx + 1

        return (torch.tensor(batch), torch.tensor(targets))

    def initial_input(self):
        sequence = []
        idx = 0
        for _ in range(self.lookBack):
            sample = self.prepare_sample(idx)
            sequence.append(sample)
            idx = idx + 1
        
        target = self.data[0][idx]

        return (torch.tensor([sequence]), torch.tensor([target]))

    def train(self):
        print('starting training LSTM with training data: ' + self.filePath)
    
        #initialize model, and functions
        torch.set_default_dtype(torch.float64) 
        model = self.model.to(torch.float64)
        lossFunction = self.lossFunction()
        optimizer = self.optimizer(model.parameters())

        initialInput, initialTarget = self.initial_input()
        
        # first compute the size of the batches compared to our data
        sequences = len(self.data[0]) 
        batchSize = (sequences - self.lookBack) // self.epochs
        print(f'amount of sequences:{sequences}')
        print(f'batch size for {self.epochs} epochs:{batchSize}')

        # see how the model performs before training:
        with torch.no_grad():
            initialScores, h0, c0 = model(initialInput)

            print('initial prediction:')
            print(initialScores.item())
            print('target:')
            print(initialTarget.item())

        # train the model 
        h0, c0 = None, None
        for epoch in range(self.epochs):
            # clear aout accumulated gradients
            model.zero_grad()

            # prepare the input and the desired output
            inputMatrix, targets = self.prepare_batch(epoch, batchSize)
            # forward pass
            
            predicted, h0, c0 = model(inputMatrix, h0, c0)

            loss = lossFunction(predicted, targets)
            loss.backward()
            optimizer.step()
           
            h0, c0 = h0.detach(), c0.detach()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')


        # see the model scores after training
        with torch.no_grad():
            scores, h0, c0 = model(initialInput)
            print('prediction after training for %d epochs:' % self.epochs)
            print(scores.item())
            print('target:')
            print(initialTarget.item())

