from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error


def save_metrics(
    Y_dict,
    Y_pred_dict,
    keys,
    name_prefix,
    save_path
):
    for house in keys:
        Y_house = Y_dict[house]
        Y_pred_house = Y_pred_dict[house]

        if(Y_house == []):
            continue

        MAPE = mean_absolute_percentage_error(Y_house, Y_pred_house)
        RMSE = root_mean_squared_error(Y_house, Y_pred_house)
        MAE = mean_absolute_error(Y_house, Y_pred_house)

        file_name = name_prefix + house + '.txt'
        file_path = save_path + '/' + file_name

        save_metrics_to_file(MAPE, RMSE, MAE, file_path)

def save_metrics_to_file(MAPE, RMSE, MAE, path):
    with open(path, 'w') as f:
        print(f'MAPE:{MAPE}', file=f)
        print(f'RMSE:{RMSE}', file=f)
        print(f'MAE:{MAE}', file=f)

