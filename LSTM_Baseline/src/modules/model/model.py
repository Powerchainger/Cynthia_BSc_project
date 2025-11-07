import torch 
import torch.nn as nn 
from modules.model.model_params import Model_params 

OUTPUT_DIM = 1

# class responsible for the LSTM that performs load forecasting
class Model(nn.Module):
    def __init__(self, config) :
        super(Model, self).__init__()

        self.hidden_dim = config.hidden_dim
        self.layer_dim = config.layer_dim

        self.lstm = nn.LSTM(
            config.input_dim,
            config.hidden_dim,
            config.layer_dim,
            batch_first=True)

        self.fc = nn.Linear(config.hidden_dim, OUTPUT_DIM)

    def forward(self, x, h0=None, c0=None) :
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        #TODO: check this
        out = self.fc(out[:, -1, :])
        return out, hn, cn
