import json 


class __JSON_loader_base():
    """ Base class for loading self._data from json files

    Initialization args:
    file_path -- the path to the JSON file to load
    """
    def __init__(self, file_path):
        self._data = {} 

        self.__read_data(file_path)

    def __read_data(self, file_path):
        """ Reads JSON from file_path and stores it in self._data """
        with open(file_path, 'r') as file:
            self._data = json.load(file)


class JSON_program_args_loader(__JSON_loader_base):
    """ Class for loading args for the run_experiment script

    Inherits from __JSON_loader_base. The class will load a json file,
    and then set the values of the file as its fields

    Initialization args:
    file_path -- the path to the JSON file to load

    fields:
    csv_path -- path to the self._data to train and test the model on
    out_path -- path to store the results to

    load_params -- boolean
    load_models -- boolean

    time_steps -- the length of the input for both models 
    params_baseline_path -- path to parameters of baseline model
    params_NILM_path -- path to parameters of enhanced model
    models_baseline_path -- path to baseline model file
    models_NILM_path -- path to enhanced model file
    """
    def __init__(self, file_path):
        super().__init__(file_path)
            
        self.__init_fields()


    def __init_fields(self):
        """assigns the values to their respective fields"""
        self.csv_path = self._data['csv_path']
        self.out_path = self._data['out_path']

        self.load_params = self._data['load_params']
        self.load_models = self._data['load_models']
        
        self.params_baseline_path = ''
        self.params_NILM_path = ''
        self.models_baseline_path = ''
        self.models_NILM_path = ''
        self.time_steps = self._data['time_steps']
        
        if(self.load_params) :
            self.params_baseline_path = self._data['params_baseline_path']
            self.params_NILM_path = self._data['params_NILM_path']

        if(self.load_models) :
            self.model_baseline_path = self._data['model_baseline_path']
            self.model_NILM_path = self._data['model_NILM_path']


class JSON_model_params_loader(__JSON_loader_base):
    """ Class for loading model parameters

    Inherits from __JSON_loader_base. This class loads model parameters
    from file_path, and contains a to_tuple method for initializing the
    Model class.

    Initialization args:
    file_path -- the path to the JSON file to load

    Methods:
    to_tuple -- method that can be used to initialize the model(s)
    """
    def __init__(self, file_path):
        super().__init__(file_path)

        self.__init_fields()


    def __init_fields(self):
        """Sets the values from the JSON to the respective fields"""
        # hyper_params
        self.nodes_per_layer = self._data['nodes_per_layer']
        self.hidden_layers = self._data['hidden_layers']
        self.time_steps = self._data['time_steps']
        self.lr = self._data['lr']
        self.epochs = self._data['epochs']
        
        # min-max normalization self._data
        self.min_y= self._data['min_y']
        self.max_y = self._data['max_y']
        
        # amount of appliance features
        self.appliance_amount = self._data['appliance_amount']

    def to_tuple(self):
        """Converts its fields to a tuple and returns it"""
        return (
            self.nodes_per_layer,
            self.hidden_layers,
            self.time_steps,
            self.lr,
            self.epochs,
            self.min_y,
            self.max_y,
            self.appliance_amount
        )
