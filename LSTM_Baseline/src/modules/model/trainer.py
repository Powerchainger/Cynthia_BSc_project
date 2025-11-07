import torch
from sklearn.metrics import mean_absolute_percentage_error

from .model_wrapper import Model_wrapper

_LOSS_FUNCTION = torch.nn.MSELoss
_OPTIMIZER = torch.optim.Adam

class Trainer(Model_wrapper):
    def __init__(self, model, model_params, csv_file_path, csv_config, epochs):
        super().__init__(model, model_params, csv_file_path, csv_config)
       
        #private fields:
        self.__epochs = epochs 

    def _run_model(self):
        print(f'starting training LSTM with training data:{self._csv_file_path}') 

        #increase precision, some values are very small
        torch.set_default_dtype(torch.float64)
        model = self._model.to(torch.float64)

        # init loss function and optimizer
        loss_function = _LOSS_FUNCTION() 
        optimizer = _OPTIMIZER(model.parameters())

        # X     : input
        # Y_true: desired output
        X = self._input_matrix
        Y_true = self._targets

        h0, c0 = None, None
        for epoch in range(self.__epochs):
            # clear accumulated gradients
            model.zero_grad()

            # forward pass
            Y_pred, h0, c0 = model(X, h0, c0)

            # backward pass
            loss = loss_function(Y_pred, Y_true)
            loss.backward()
            optimizer.step()
          
            h0, c0 = h0.detach(), c0.detach()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.__epochs}], Loss: {loss.item():.4f}')

        # training done, show some metrics
        # FIXME: mean_absolute_percentage fails, wrong shape 
        Y_true = self._output_scaler.inverse_transform(Y_true.detach().numpy())
        Y_pred = self._output_scaler.inverse_transform(Y_pred.detach().numpy())
        MAPE = mean_absolute_percentage_error(Y_true, Y_pred)
        print(f'Training finished, MAPE on training dataset={MAPE}')
