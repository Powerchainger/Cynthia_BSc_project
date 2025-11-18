import sys

from modules.io.load import load_model_params, load_model, load_data
from modules.model.run import run_model

# Program needs 3 args:
#   1. the file_path for the model 
#   2. the file_path for the params for the model
#   3. the file_path for the data to run the model on
def main() :

    if(len(sys.argv) < 3):
        print('Error: too few arguments to run program, 3 args needed')
        exit(1)

    model_path = sys.argv[1]
    model_params_path = sys.argv[2]
    input_data_path = sys.argv[3]

    # load the model which we want to run
    model_params = load_model_params(model_params_path)
    model = load_model(model_path, model_params)

    # load the data to run the model on
    input_data = load_data(input_data_path)

    # run the model
    run_model(model, input_data)

if __name__ == '__main__':
    main() 
