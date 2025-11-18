import torch 
import torch.nn as nn 

# output dimension is 24, the hourly load predicted for the next day 
_OUTPUT_DIM = 24 

# input consists of load, month, day, time:
#   load_dim        = 1
#   appliance_dim   = 1 for each appliance 
#   month_dim       = 12
#   day_dim         = 7
#   time_dim        = 24  +
#   -------------------
#   input_dim       = 44 + 1*appliance
_INPUT_DIM = 44

# currently binary
_APPLIANCE_INPUT_DIM = 1

# class responsible for the model that performs load forecasting
class Forecaster(nn.Module):
    def __init__(self, hidden_layer_nodes, hidden_layers, time_steps, appliance_count) :
        super(Forecaster, self).__init__()

        self.time_steps = time_steps
        self.appliance_count = appliance_count

        input_dim = _INPUT_DIM + appliance_count * _APPLIANCE_INPUT_DIM

        self.lstm = nn.LSTM(
            input_dim,
            hidden_layer_nodes,
            hidden_layers)

        self.fc = nn.Linear(hidden_layer_nodes, _OUTPUT_DIM)

    def forward(self, x) :

        # first run input through LSTM
        out, _ = self.lstm(x)

        # then run output of LSTM through Feed Forward Network to get our output
        out = self.fc(out[:, -1, :])
        
        return out
