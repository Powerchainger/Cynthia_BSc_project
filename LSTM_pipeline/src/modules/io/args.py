import json

class Args:
    """ A simple class that reads arguments from a JSON file
        
        Args is a class that reads arguments for the pipeline from JSON.
        The path to the JSON file must be given at initialization,
        the args are then initialized immediatly and stored as plain old data.

        The fields of the data are as follows:
            csv_path                (str)
            out_path                (str)
            load_params             (bool)
            params_baseline_path    (only loaded if load_params == true) (str)
            params_NILM_path        (only loaded if load_params == true) (str)
            load_model              (bool)
            model_baseline_path     (only loaded if load_model == true) (str)
            model_NILM_path         (only loaded if load_model == true) (str)
    """

    def __init__(self, json_path: str) -> None:
        self.__read_data(json_path)

    def __read_data(self, json_path: str) -> None:
        """ Reads the args from the JSON file, exception handling must be done by the user of the class """

        with open(json_path, 'r') as file:
            data = json.load(file)
           
            self.csv_path = data['csv_path']
            self.out_path = data['out_path']

            self.load_params = data['load_params']
            self.load_models = data['load_models']

            
            if(self.load_params) :
                self.params_baseline_path = data['params_baseline_path']
                self.params_NILM_path = data['params_NILM_path']
            else:   
                self.time_steps = data['time_steps']

            if(self.load_models) :
                self.model_baseline_path = data['model_baseline_path']
                self.model_NILM_path = data['model_NILM_path']
