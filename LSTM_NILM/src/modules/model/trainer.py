import torch
import time
import os
from tempfile import TemporaryDirectory
from sklearn.metrics import mean_absolute_percentage_error

from .model_wrapper import Model_wrapper

_LOSS_FUNCTION = torch.nn.L1Loss
_OPTIMIZER = torch.optim.Adam

class Trainer(Model_wrapper):
    def __init__(self, model, model_params, csv_config, eval_file_path, train_file_path, epochs):
        super().__init__(model, model_params, csv_config, eval_file_path, train_file_path)
       
        #private fields:
        self.__epochs = epochs 

    def _train_model(self):
        since = time.time()
        print() 
        print('start training LSTM')
        print(f'Training data:  \t{self._train_file_path}') 
        print(f'Validation data:\t{self._eval_file_path}')

        # increase precision, some values are very small
        torch.set_default_dtype(torch.float64)
        model = self._model.to(torch.float64)

        # init loss function and optimizer
        loss_function = _LOSS_FUNCTION() 
        optimizer = _OPTIMIZER(model.parameters())

        # Create a temp dir to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_temp_path = os.path.join(tempdir, 'best_model_params.pt') 
       
            # init saves, MAPE is initialized to inf so all models all better
            torch.save(model.state_dict(), best_temp_path)
            best_MAPE = float('inf') 

            for epoch in range(self.__epochs):
                # each epoch has a training and validation phase
                if((epoch + 1)% 10 == 0):
                    print(f'Epoch_{epoch + 1}/{self.__epochs}')
                for phase in ['train', 'eval']:
                    if phase == 'train' :
                        model.train()
                    else :
                        model.eval()
                   
                    # if training grab the training data
                    # if validating grab the validation data
                    X = self._input_matrix[phase]
                    Y_true = self._targets[phase]
                    Y_true_raw = self._targets_raw[phase]
                    Y_scaler = self._output_scaler[phase]

                    # clear accumulated gradients
                    model.zero_grad()

                    # only track gradients if we are training
                    with torch.set_grad_enabled(phase == 'train'):
                        Y_pred, _, _ = model(X)
                        loss = loss_function(Y_pred, Y_true)

                        # backward  + optimize if we are training
                        if phase == 'train' :
                            loss.backward()
                            optimizer.step()

                    Y_pred = Y_scaler.inverse_transform(Y_pred.detach().clone().numpy())
                    epoch_MAPE = mean_absolute_percentage_error(Y_true_raw, Y_pred)
                    
                    if((epoch + 1) % 10 == 0):
                        print(f'\t{phase} Loss:\t{loss.item():.4f}, MAPE:\t{epoch_MAPE:.4f}')

                    if phase == 'eval' and epoch_MAPE < best_MAPE:
                        best_MAPE = epoch_MAPE
                        torch.save(model.state_dict(), best_temp_path)
                
            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'best validation MAPE: {best_MAPE:.4f}')
            
            #Load the best weights we have found during training
            model.load_state_dict(torch.load(best_temp_path, weights_only=True))
