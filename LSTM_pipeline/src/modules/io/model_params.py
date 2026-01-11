import json

class Model_params:
    """ A class that loads the parameters and training information for the model

        Model_params is a class that reads parameters and training information for the
        model. It takes 1 argument, a filepath to a JSON file that contains the values
        that belong to the parameters of the model. The values are read immediatly and
        stored. It also supports converting itself as a tuple such that the model can be
        initialized using this class
           
        The fields are as follows:
        hidden_layers: the amount of hidden layers for the model
        nodes_per_layer: the amount of nodes per layer for the model
        time_steps: the amount of time steps that the model takes in its input
        lr: the learning rate of the model
        epochs: the amount of epochs the model is to be trained for
        main_min: the min value to normalize the values belonging to 'main' with
        main_max: the max value to normalize the values belonging to 'main' wiht
        appliances: an array of names belonging to the appliances that the model also takes as its input
        appliances_min_max: an array containing tuples that belong to the respective appliance in appliances that are to be used for min, max normalization
    """

    def __init__(self, json_path: str) -> None:
        self.__read_data(json_path)

    def __read_data(self, json_path: str) -> None:
        """ Reads the model_params from the JSON file, exception handling must be done by the user of the class """

        with open(json_path, 'r') as file:
            data = json.load(file)

            self.nodes_per_layer = data['nodes_per_layer']
            self.hidden_layers = data['hidden_layers']
            self.time_steps = data['time_steps']
            self.lr = data['lr']
            self.epochs = data['epochs']
            
            self.min_y= data['min_y']
            self.max_y = data['max_y']

            self.appliance_amount = data['appliance_amount']

    #TODO: order on the tuple
    def to_model_args(self) -> tuple[int, int, int, float, int, int, [str], [(int, int)]]:
        """ converts the relevant fields needed for initialization of the model to a tuple"""
        return (self.nodes_per_layer,
                self.hidden_layers,
                self.time_steps,
                self.lr,
                self.epochs,
                self.min_y,
                self.max_y,
                self.appliance_amount)
                
