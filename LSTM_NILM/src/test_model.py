import sys

from modules.io.load import load_model_params, load_model, load_data

from modules.data_processing.pre_processing import pre_process

from modules.model.test import test_model

# Program needs 3 args:
#   1. the file_path for the model 
#   2. the file_path for the params for the model
#   3. the file_path for the data to test the model on
def main() :

    if(len(sys.argv) < 5):
        print('Error: too few arguments to run program, 4 args needed')
        exit(1)

    model_path = sys.argv[1]
    model_params_path = sys.argv[2]
    testing_data_path = sys.argv[3]
    testing_results_path = sys.argv[4]

    # load the model which we want to test 
    model_params, appliances = load_model_params(model_params_path)
    model = load_model(model_path, model_params + (len(appliances),))

    # load the data to test the model on
    testing_data = load_data(testing_data_path, appliances)

    # test the model
    test_model(model, testing_data, testing_results_path)

if __name__ == '__main__':
    main() 
