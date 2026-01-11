import torch 
import torch.nn as nn 

# output dimension is 24, the hourly load predicted for the next day 
_OUTPUT_DIM = 24 

# input consists of load, month, day, time:
#   load_dim    = 1
#   day_dim     = 7
#   time_dim    = 24  +
#   -------------------
#   input_dim   = 44 
_INPUT_DIM = 32  

_TIMESTEPS_DAY_0 = 10
_TIMESTEPS_PER_DAY = 24
# class responsible for the model that performs load forecasting
class Forecaster(nn.Module):
    def __init__(self, hidden_layer_nodes, hidden_layers, input_days, learning_rate, epochs) :
        super(Forecaster, self).__init__()

        self.time_steps = _TIMESTEPS_DAY_0 + _TIMESTEPS_PER_DAY * input_days
        self.lr = learning_rate 
        self.epochs = epochs

        self.lstm = nn.LSTM(
            _INPUT_DIM,
            hidden_layer_nodes,
            hidden_layers)

        self.fc = nn.Linear(hidden_layer_nodes, _OUTPUT_DIM)

    def forward(self, x) :

        # first run input through LSTM
        out, _ = self.lstm(x)

        # then run output of LSTM through Feed Forward Network to get our output
        out = self.fc(out[:, -1, :])
        
        return out
