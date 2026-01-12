import sys

from modules.io.args import Args
from modules.io.model_params import Model_params
from modules.io.load import load_data, load_model

from modules.model.train import train_model, hyper_parameter_tuning
from modules.model.test import test_model

def main() -> None:
    """ The pipeline to perform experiments needed for Cynthia's Bachelor's Thesis
    
        The program takes 1 argument:
            - The filepath to a JSON file that contains the arguments for the program

        It will then perform the following steps:
            1. Load the arguments from the JSON file
            2. Load the training, validation, and testing data from a CSV file
            3. Initialize the parameters for the baseline and NILM models:
                3a. If load_params was specified, the params will be loaded from files (JSON files)
                3b. If load_params was not specified, hyper parameter tuning will be performed to compute the params (Note: this might take a while)
            4. Initialize the baseline and NILM models:
                4a. If load_model was specified, the models will be loaded from files (.pt files)
                4b. If load_model was not specified, the models will be trained on the training and validation data
            5. Perform testing on the baseline and NILM models using the testing data 
    """

    if (len(sys.argv) < 2):
        print('Error: path to args.json needed')
        exit(1)
   
    print("Loading args from file....")
    args_path = sys.argv[1]
    args = Args(args_path)  
    print("Done....")

    print("Loading dataset from CSV")
    (training, validation, testing), appliances, min_max_dict = load_data(args.csv_path)
    print("Done....") 

    # load params, or perform hyperparam tuning
    if (args.load_params):
        print("Loading model parameters from file....")
        params_baseline = Model_params(args.params_baseline_path)
        params_NILM = Model_params(args.params_NILM_path)
    else:
        print("computing model parameters from hyper parameter tuning....")
        params_baseline = hyper_parameter_tuning(training, validation, min_max_dict, args.out_path, 'params_baseline')
        params_NILM = hyper_parameter_tuning(training, validation, min_max_dict, args.out_path, 'params_NILM', appliances)

    print("Done....")
    # train the baseline and the NILM models or load the models
    if (args.load_models):
        print("Loading models from file....")
        model_baseline = load_model(args.model_baseline_path, params_baseline)     
        model_NILM = load_model(args.model_NILM_path, params_NILM)
    else:    
        print("Training models....")
        model_baseline = train_model(params_baseline, training, min_max_dict, args.out_path, 'model_baseline')
        model_NILM = train_model(params_NILM, training, min_max_dict, args.out_path, 'model_NILM', appliances)
    print("Done....")
    # test the models and save the results
    print("Testing models....")
    test_model(params_baseline, model_baseline, testing, min_max_dict, args.out_path, 'results_baseline')
    test_model(params_NILM, model_NILM, testing, min_max_dict, args.out_path, 'results_NILM', appliances)
    print("Done....")

if __name__ == '__main__':
    main()
        




    



