import torch
import numpy as np
from LSTM_Forecaster import LSTMForecaster
from ../io/load import loadData_aggregate as loadData

class modelTrainer():
    def __init__(self,
        model, 
        lookBack,
        loss_func, 
        optimizer,
        data_path,
        csv_config):
        
        self.model = model
        self.data_path = data_path
        self.loss_func = loss_func
        self.optimizer = optimizer
        
        data, time, dayMarker, holidayMarker = loadData(data_path, csv_config)
        self.data = data
        self.time = time
        self.dayMarker = dayMarker
        self.holidayMarker = holidayMarker

    def initial_input():
        self.idx = 0
        return self.prepare_input()

    def prepare_input(self):
        new_idx = self.idx + lookback

        E = data[self.idx:lookback]
        time = data[self.idx:lookback]
        dayMarker = data[self.idx:lookback]
        holidayMarker = data[self.idx:lookback]

        return (E + time + dayMarker + holidayMarker)

    def train(self, epochs=150):
        print('starting training LSTM with training data:' + data_path)

        initial_input = initial_input()
        
        # see how the model performs before training:
        with torch.no_grad():
            initial_scores = model(initial_input)

            print('initial scores:')
            print(initial_scores)

        # train the model  
        for epoch in range(epochs):

            # clear out accumulated gradients
            model.zero_grad()

            # prepare the input and the desired output
            input_matrix, target = self.prepareinput()
            # forward pass
            scores = model(input_matrix)

            loss = loss_func(scores, target)
            loss.backward()
            optimizer.step()

        # see the model scores after training
        with torch.no_grad():
            scores = model(initial_input)
            print('scores after training for %d epochs:' % epochs)
            print(scores)
