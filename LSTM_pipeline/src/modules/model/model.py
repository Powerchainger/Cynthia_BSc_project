import torch 
import torch.nn as nn 

# output dimension is 24, the hourly load predicted for the next day 
_OUTPUT_DIM = 24 
_TIMESTEPS_DAY_0 = 10
_TIMESTEPS_PER_DAY = 24

_MAIN_DIM = 1
_APPLIANCE_INPUT_DIM = 1
_DAY_DIM = 7
_HOUR_DIM = 24

# class responsible for the model that performs load forecasting
class Forecaster(nn.Module):
    def __init__(self,
                 nodes_per_layer: int,
                 hidden_layers: int,
                 time_steps: int,
                 lr: float,
                 epochs: int,
                 min_y: int,
                 max_y: int,
                 appliance_amount: int):

        super(Forecaster, self).__init__()

        self.time_steps = time_steps 
        self.lr = lr 
        self.epochs = epochs
        self.min_y = min_y
        self.max_y = max_y

        input_dim = _MAIN_DIM + _DAY_DIM + _HOUR_DIM + appliance_amount * _APPLIANCE_INPUT_DIM

        self.lstm = nn.LSTM(
            input_dim,
            nodes_per_layer,
            hidden_layers)

        self.fc = nn.Linear(nodes_per_layer, _OUTPUT_DIM)

    def forward(self, x):

        # first run input through LSTM
        out, _ = self.lstm(x)

        # then run output of LSTM through Feed Forward Network to get our output
        out = self.fc(out[:, -1, :])
       
        # lastly scale the output according to min max normalization
        out = self.__scale_results(out)

        return out

    def __scale_results(self, x):
        return x * (self.max_y - self.min_y) + self.min_y 

