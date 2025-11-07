#POD containing the parameters for the LSTM, for ease of tweaking the model

class Model_params():
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        look_back, #TODO: find a better variable name for this
        ):

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.look_back = look_back
