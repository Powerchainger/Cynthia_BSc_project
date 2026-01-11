import torch

from modules.model.model import Forecaster

# Function that runs a model in the input data received
def run_model(model, input_data, output_path) :

    time_steps = model.time_steps
    X = prepare_input_matrix(input_data, time_steps, no_targets=True)

    # could go wrong if nans?
    X_min = np.min[input_data[0]]
    X_max = np.max[input_data[0]]

    model.eval()
    with torch.no_grad() :
        Y_pred = model(X)

    Y_pred = scale_tensor(Y_pred, X_min, X_max)

    # what to do with Y_pred, turn it into a csv I guess
    df = pd.DataFrame(Y_pred)  
    df.to_csv(output_path, index=False)

    # then plot it as well, could be easy peasy
    plot_day(Y_pred) 
