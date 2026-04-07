import os
import matplotlib.pyplot as plt


def _plot_day(
    Y,
    Y_pred,
    name,
    save_path
):
    fix, ax = plt.subplots()
    ax.plot(Y, '-', color='r', label='True')
    ax.plot(Y_pred, '-', color='b', label='Forecast')
    ax.legend()
    
    full_save_path = save_path + '/' + name + '.png'
    plt.suptitle(name)
    plt.savefig(full_save_path, dpi=300)
    plt.close()

def plot_results(
    Y,
    Y_pred,
    dates,
    name,
    save_path
):
    plot_path = save_path + '/' + name + '/'
    os.makedirs(plot_path, exist_ok=True)

    current_day = dates[0]
    day_Y, day_Y_pred = [], []

    for y, y_pred, date in zip(Y, Y_pred, dates):
        if (date.date() == current_day.date()): 
            day_Y.append(y)
            day_Y_pred.append(y_pred)
        else:
            if (len(day_Y) == 24):
                _plot_day(day_Y, day_Y_pred, str(current_day.date()), plot_path)

                current_day = date
                day_Y = [y]
                day_Y_pred = [y_pred]

def result_plots_individual(
    Y_dict,
    Y_pred_dict,
    dates_dict,
    keys,
    save_path
):
    for house in keys:
        if(Y_dict[house] == []):
            continue

        Y_house = Y_dict[house]
        Y_pred_house = Y_pred_dict[house]
        dates_house = dates_dict[house]
        name = house

        plot_results(
            Y=Y_house,
            Y_pred=Y_pred_house,
            dates=dates_house,
            name=name,
            save_path=save_path
        )
