from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

#TODO type suggestions
def compute_value_metrics(Y_pred, Y, save_path):
    #could add more metrics here
    MAPE = mean_absolute_percentage_error(Y, Y_pred) 
    RMSE = root_mean_squared_error(Y, Y_pred)
    MAE = mean_absolute_error(Y,Y_pred) 
    
    save_path = save_path + 'metrics.txt'
    with open(save_path, 'w') as f:
        print(f'MAPE:{MAPE}', file=f)
        print(f'RMSE:{RMSE}', file=f)
        print(f'MAE:{MAE}', file=f)
