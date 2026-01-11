import sys

from modules.io.save import save_model
from modules.io.load import load_model_params, load_data

from modules.model.train import train_model

# Program needs 4 args:
#   1. params for the model to train, inside a config file 
#   2. file path to the training data
#   3. file path to the validation data
#   4. epochs to train the model for
#   5. file path to save the model to
def main() :

    if (len(sys.argv) < 5):
        print('Error: too few arguments to run program, 4 args needed')
        exit(1)

    # the args given
    model_params_path = sys.argv[1]
    training_data_path = sys.argv[2]
    validation_data_path = sys.argv[3]
    model_path = sys.argv[4]
  
    # load the data from the files as given in the args 
    model_params, appliances = load_model_params(model_params_path)
    training_data = load_data(training_data_path, appliances)
    validation_data = load_data(validation_data_path, appliances)

    # train the model
    model = train_model(model_params + (len(appliances),), training_data, validation_data)

    # save the model to the path given in the args
    save_model(model, model_path)

# ensure that we only call main if we run the program directly
if __name__ == '__main__' :
    main()
