import sys

from modules.io.args import Args
from modules.io.model_params import Model_params
from modules.io.load import load_data, load_model

from modules.model.train import train_model, hyper_parameter_tuning
from modules.model.test import test_model

from modules.metrics.plot import running_loss_plot

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

    if (args.load_params):
        print("Loading model parameters from file....")
        params_baseline = Model_params(args.params_baseline_path)
        #params_NILM = Model_params(args.params_NILM_path)
    else:
        print("computing model parameters from hyper parameter tuning for baseline....")
        params_baseline = hyper_parameter_tuning(training, validation, min_max_dict, args.out_path, 'params_baseline', args.time_steps)
        print("Done....")
        print("Computing model parameters from hyper parameter tuning for NILM....")
        #params_NILM = hyper_parameter_tuning(training, validation, min_max_dict, args.out_path, 'params_NILM', appliances)

    print("Done....")

    if (args.load_models):
        print("Loading models from file....")
        model_baseline = load_model(args.model_baseline_path, params_baseline)     
        #model_NILM = load_model(args.model_NILM_path, params_NILM)
    else:    
        print("Training baseline model....")
        model_baseline, baseline_val_loss = train_model(params_baseline, training, validation, min_max_dict, args.out_path, 'model_baseline')
        print("Done....")
        print("Training NILM model....")
        #model_NILM, NILM_val_loss = train_model(params_NILM, training, validation, min_max_dict, args.out_path, 'model_NILM', appliances)
        running_loss_plot(baseline_val_loss, baseline_val_loss, args.out_path)
    print("Done....")
    
    print("Testing baseline model....")
    test_model(params_baseline, model_baseline, testing, min_max_dict, args.out_path, 'results_baseline')
    print("Done....")
    #print("Testing NILM model....")
    #test_model(params_NILM, model_NILM, testing, min_max_dict, args.out_path, 'results_NILM', appliances)
    #print("Done....")

if __name__ == '__main__':
    main()
        




    



