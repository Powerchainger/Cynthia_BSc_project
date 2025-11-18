import torch
from sklearn.metrics import mean_absolute_percentage_error

from .model_wrapper import Model_wrapper

class Tester(Model_wrapper):
    def __init__(self, model, model_params, csv_file_path, csv_config):
        super().__init__(model, model_params, csv_config, csv_file_path)

    def _run_model(self):
        print(f'starting testing LSTM with testing data:{self._eval_file_path}')
        phase = 'eval' 
        # increase precision, some values are very small
        torch.set_default_dtype(torch.float64) 
        model = self._model.to(torch.float64)
        
        # X     : input
        # Y-true: desired output
        X = self._input_matrix[phase] 
        Y_true = self._targets_raw[phase]
        scaler = self._output_scaler[phase]

        # test the model 
        model.eval()
        with torch.no_grad():
            Y_pred, _, _ = model(X) 
        
        Y_pred = scaler.inverse_transform(Y_pred.detach().numpy())
        MAPE = mean_absolute_percentage_error(Y_true, Y_pred)

        print(f'MAPE for model, on dataset:{self._eval_file_path} = {MAPE}')
   
