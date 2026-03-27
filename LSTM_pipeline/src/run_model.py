import sys
import pandas as pd

from modules.io.load import load_model_params, load_model, load_data
from modules.model.run import run_model

def main() :
    """ The pipeline to perform a single prediction using an LSTM model

        The program takes 
            - The filepath to the parameters of the model
            - The filepath to the model
            - The filepath to a CSV file containing a single input of data

        It will then perform the following steps:
            1. Load the parameters of the model
            2. Load the model
            3. Load the input for the model
            4. Run the model for the input*
            5. Print the output to stdout
        
        *: 
            Step 4 will return the output of the model as a single 24 dimension array.
            This can be easily integrated or adapted for other use cases if needed. 
            step 5 is only here for completeness sake.
    """

    if(len(sys.argv) < 3):
        print('Error: too few arguments to run program, 3 args needed')
        exit(1)

    model_path = sys.argv[1]
    model_params_path = sys.argv[2]
    input_data_path = sys.argv[3]
    min_max_dict_path = sys.argv[4]
    is_NILM = False 

    # load the model which we want to run
    model_params = load_model_params(model_params_path)
    model = load_model(model_path, model_params)

    # load the data to run the model on
    input_data = pd.read_csv(input_data_path, parse_dates=['time'])

    with open(min_max_dict_path, 'r') as file:
        min_max_dict = json.load(file)

    # run the model
    run_model(model, input_data, min_max_dict, out_dir)

if __name__ == '__main__':
    main() 
