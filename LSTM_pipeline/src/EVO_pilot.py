import sys

def main() -> None:
    # read the data, 0.6/0.2/0.2 split 
    # the last 0.2 will probably be around a week of time for the predictions
    # hyperparam tuning
    if (len(sys.argv) < 2):
        print('Error: path to args.json needed')
        exit(1)

    args_path = sys.argv[1]
    args = Args(args_path)

    (training, validation, testing), _, min_max_dict = load_data(args.csv_path)
    
    if (args.load_params):
        params = Model_params(args.params_baseline_path)
    else:
        params = hyper_parameter_tuning(training, validation, min_max_dict, args.out_path, 'params')

    if (args.load_models):
        model = load_model(args.model_baseline_path, params)
    else:
        model = train_model(params, training, min_max_dict, args.out_path, 'model')

    test_model(params, model, testing, min_max_dict, args.out_path, 'results')

   
if __name__ == '__main__':
    main()
