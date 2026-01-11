import torch

from modules.model.model import Forecaster

# Function that runs a model in the input data received
#TODO
def run_model(model, input_data, output_path) :

    time_steps. model.time_steps
    X = create_input_matrix(input_data, time_steps, no_targets=True)

    X_min = torch.min(input_data)
    X_max = torch.max(input_data)

    model.eval()
    with torch.no_grad() :
        Y_pred = model(X)

    Y_pred = scale_tensor(Y_pred, X_min, X_max)

    df = pd.DataFrame(Y_pred)
    df.to_csv(output_path, index=False)

    plot_day(Y_pred)
