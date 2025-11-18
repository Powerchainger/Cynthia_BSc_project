import torch 
import torch.nn as nn 

# output dimension is 24, the hourly load predicted for the next day 
_OUTPUT_DIM = 24 

# input consists of load, month, day, time:
#   load_dim    = 1
#   month_dim   = 12
#   day_dim     = 7
#   time_dim    = 24  +
#   -------------------
#   input_dim   = 44 
_INPUT_DIM = 44 

# class responsible for the model that performs load forecasting
class Forecaster(nn.Module):
    def __init__(self, hidden_layer_nodes, hidden_layers, time_steps) :
        super(Forecaster, self).__init__()

        self.time_steps = time_steps

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
