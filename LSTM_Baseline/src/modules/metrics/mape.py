from sklearn.metrics import mean_absolute_percentage_error

def MAPE(actual, predicted):
    return mean_absolute_percentage_error(actual, predicted)
