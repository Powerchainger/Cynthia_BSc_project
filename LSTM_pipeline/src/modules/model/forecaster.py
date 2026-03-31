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


class Forecaster(nn.Module):
    """ Model that can be trained for performing day ahead forecasting 

    Consists of a LSTM, and a FFNN. Inputs are ran through the LSTM and
    the output of the LSTM is ran through the FFNN to map the results
    to a dimension of 24. Lastly an inverse min-max normalization is
    performed to get the final results

    Keyword initialization arguments:
    nodes_per_layer -- the amount of nodes in each layer
    hidden_layers -- the amount of hidden layers
    time_steps -- the length of the input sequences
    lr -- learning rate of the model
    epochs -- amount of epochs the model needs to be trained for
    min_y -- used for inverse min-max normalization
    max_y -- idem
    appliance_amount -- the amount of appliance features this model
    expects.

    Methods:
    forward -- performs the forward pass of the model, 
               returns the output of the forward pass
    """
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

        input_dim = (_MAIN_DIM 
                     + _DAY_DIM 
                     + _HOUR_DIM 
                     + appliance_amount 
                     * _APPLIANCE_INPUT_DIM)

        self.lstm = nn.LSTM(
            input_dim,
            nodes_per_layer,
            hidden_layers)

        self.fc = nn.Linear(nodes_per_layer, _OUTPUT_DIM)

    def forward(self, x):
        """Performs the forward pass of the model"""
        # first run input through LSTM
        out, _ = self.lstm(x)

        # then run output of LSTM through Feed Forward Network to get our output
        out = self.fc(out[:, -1, :])
       
        # lastly scale the output according to min max normalization
        out = self.__scale_results(out)

        return out

    def __scale_results(self, x):
        """Performs inverse of min max normalization"""
        return x * (self.max_y - self.min_y) + self.min_y 

