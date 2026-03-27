#unused
import sys

from modules.io.args import Args
from modules.io.model_params import Model_params
from modules.io.load import load_data, load_model

from modules.model.train import train_model, hyper_parameter_tuning
from modules.model.test import EVO_pilot_test_model as test_model

def main() -> None:
    # read the data, 0.6/0.2/0.2 split 
    # the last 0.2 will probably be around a week of time for the predictions
    # hyperparam tuning
    if (len(sys.argv) < 2):
        print('Error: path to args.json needed')
        exit(1)

    print('Loading args....')
    args_path = sys.argv[1]
    args = Args(args_path)
    print('Done....')

    print('Loading data from csv....')
    (training, validation, testing), _, min_max_dict = load_data(args.csv_path)
    print('Done....')

    if (args.load_params):
        print('Loading args from file....')
        params = Model_params(args.params_baseline_path)
    else:
        print('Computing args with hyper parameter tuning....')
        params = hyper_parameter_tuning(training, validation, min_max_dict, args.out_path, 'params')
    print('Done....')

    if (args.load_models):
        print('Loading model from file....')
        model = load_model(args.model_baseline_path, params)
    else:
        print('Training model....')
        model, _ = train_model(params, training, validation, min_max_dict, args.out_path, 'model')
    print('Done....')

    print('Testing model....')
    test_model(params, model, testing, min_max_dict, args.out_path, 'results')
    print('Done....')
   
if __name__ == '__main__':
    main()
