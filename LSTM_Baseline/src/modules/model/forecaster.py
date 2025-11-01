import torch
import torch.nn as nn
import numpy as numpy
import matplotlib.pyplot as plt

from modules.model.config import Model_config 

# class responsible for the LSTM that performs load forecasting
class LSTM_forecaster(nn.Module):
    def __init__(self, config) :
        super(LSTM_forecaster, self).__init__()

        self.hidden_dim = config.hiddenDim
        self.layer_dim = config.layerDim

        self.lstm = nn.LSTM(
            config.inputDim,
            config.hiddenDim,
            config.layerDim,
            batch_first=True)

        self.fc = nn.Linear(config.hiddenDim, config.outputDim)

    def forward(self, x, h0=None, c0=None) :
        if h0 is None or c0 is None:
            h0 = torch.randn(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)
            c0 = torch.randn(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn
