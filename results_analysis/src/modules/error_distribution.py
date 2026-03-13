import matplotlib.pyplot as plt
import statistics as stats

def _plot_error_distribution(Y, Y_pred, name, save_path):
   
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    # histogram
    error = [ 100 * (y_hat - y) / y for (y, y_hat) in zip(Y, Y_pred) ]

    ax.hist(
        x=error,
        bins=50,
        range=(-300, 300),
        density=True,
        color='tab:blue')

    # mean and stdev lines
    mean = sum(error) / len(error)
    stdev = stats.stdev(error) 
    ax.vlines(
        x=mean,
        ymin=0,
        ymax=1,
        transform=ax.get_xaxis_transform(),
        linestyles='dashed',
        colors='tab:red',
        label= f'mean: {mean:.4f}' )

    ax.vlines(
        x=[mean - stdev, mean + stdev],
        ymin=0,
        ymax=1,
        transform=ax.get_xaxis_transform(),
        linestyles='dashed',
        colors='tab:orange',
        label= f'stdev: {stdev:.4f}')

    ax.legend()
    plt.suptitle(name)

    full_save_path = save_path + '/' + name + '.png'
    plt.savefig(full_save_path, dpi=300)
    plt.close()

def cumulative_error_distribution_all(
    Y_dict,
    Y_pred_dict,
    keys,
    name,
    save_path):

    Y, Y_pred = [], []
    for house in keys:
        Y_house = Y_dict[house]
        Y_pred_house = Y_pred_dict[house]

        day_Y, day_Y_pred, count = 0, 0, 0
        for y, y_hat in zip(Y_house, Y_pred_house):
            day_Y = day_Y + y 
            day_Y_pred = day_Y_pred + y_hat
            count = count + 1

            if (count == 24):
                Y.append(day_Y)
                Y_pred.append(day_Y_pred)
                day_Y, day_Y_pred, count = 0, 0, 0
    
    _plot_error_distribution(Y, Y_pred, name, save_path)
    
def error_distribution_all(Y_dict, Y_pred_dict, keys, name, save_path):
    
    # calculate total error
    Y, Y_pred = [], []
    for house in keys:
        Y_house = Y_dict[house]
        Y_pred_house = Y_pred_dict[house]

        Y = Y + Y_house 
        Y_pred = Y_pred + Y_pred_house
    
    _plot_error_distribution(Y, Y_pred, name, save_path)

def error_distribution(Y, Y_pred, name, save_path):
    _plot_error_distribution(Y, Y_pred, name, save_path)

def cumulative_error_distribution(Y, Y_pred, name, save_path):
   
    cum_Y, cum_Y_pred = [], []
    day_Y, day_Y_pred, count = 0, 0, 0
    for y, y_hat in zip(Y, Y_pred):
        day_Y = day_Y + y
        day_Y_pred = day_Y_pred + y_hat
        count = count + 1

        if (count == 24):
            cum_Y.append(day_Y)
            cum_Y_pred.append(day_Y_pred)
            day_Y, day_Y_pred, count = 0, 0, 0

    _plot_error_distribution(cum_Y, cum_Y_pred, name, save_path)

def _plot_weekday_error_distribution(Y, Y_pred, name, save_path):

    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    error = []
    for weekday in zip(Y, Y_pred):
        error.append(
            [ 100 * (y_hat - y) / (y + 1) for (y, y_hat) in zip(*weekday) ])

    ax.boxplot(
        x=error,
        orientation='horizontal',
        tick_labels=[ 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun' ],
        patch_artist=True,
        medianprops={'color' : 'tab:red', 'linewidth' : 1.5},
        boxprops={'facecolor' : 'tab:blue', 'edgecolor' : 'tab:blue'},
        whiskerprops={'color' : 'tab:orange', 'linewidth' : 1.5},
        capprops={'color' : 'tab:orange'},
        flierprops={'markersize': 0.15, 'color' : 'tab:gray'})

    plt.suptitle(name)

    full_save_path = save_path + '/' + name + '.png'
    plt.savefig(full_save_path, dpi=300)
    plt.close()

def cumulative_weekday_error_distribution_all(
    Y_dict,
    Y_pred_dict,
    dates,
    keys,
    name,
    save_path):

    Y       = [ [] for _ in range(0,7) ]
    Y_pred  = [ [] for _ in range(0,7) ]
    for house in keys:
        Y_house = Y_dict[house]
        Y_pred_house = Y_pred_dict[house]
        dates_house = dates[house]

        for weekday in range(0,7):
            day_Y, day_Y_pred = 0, 0
            for y, y_hat, date in zip(Y_house, Y_pred_house, dates_house):

                if (date.weekday() == weekday):
                    counting = True
                else:
                    counting = False

                if counting:
                    day_Y = day_Y + y
                    day_Y_pred = day_Y_pred + y_hat
                elif (day_Y != 0 and day_Y_pred != 0):
                    # if we were acumulating for the day and we are done 
                    Y[weekday].append(day_Y)
                    Y_pred[weekday].append(day_Y_pred)
                    day_Y, day_Y_pred = 0, 0

    _plot_weekday_error_distribution(Y, Y_pred, name, save_path)

def weekday_error_distribution_all(
    Y_dict,
    Y_pred_dict,
    dates,
    keys,
    name,
    save_path):

    Y       = [ [] for _ in range(0,7) ]
    Y_pred  = [ [] for _ in range(0,7) ] 
    for house in keys:
        Y_house = Y_dict[house]
        Y_pred_house = Y_pred_dict[house]
        dates_house = dates[house]

        for weekday in range(0,7):
            for y, y_hat, date in zip(Y_house, Y_pred_house, dates_house):
                if (date.weekday() == weekday):
                    Y[weekday].append(y)
                    Y_pred[weekday].append(y_hat)
    
    _plot_weekday_error_distribution(Y, Y_pred, name, save_path)

def weekday_error_distribution(Y, Y_pred, dates, name, save_path):

    Y_weekday       = [ [] for _ in range(0,7) ]
    Y_pred_weekday  = [ [] for _ in range(0,7) ]
    for weekday in range(0,7):
        for y, y_hat, date in zip(Y, Y_pred, dates):
            if (date.weekday() == weekday):
                Y_weekday[weekday].append(y)
                Y_pred_weekday[weekday].append(y_hat)

    _plot_weekday_error_distribution(Y_weekday, Y_pred_weekday, name, save_path)

def cumulative_weekday_error_distribution(Y, Y_pred, dates, name, save_path):

    Y_weekday       = [ [] for _ in range(0,7) ]
    Y_pred_weekday  = [ [] for _ in range(0,7) ]
    for weekday in range(0,7):
        day_Y, day_Y_pred = 0, 0
        for y, y_hat, date in zip(Y_house, Y_pred_house, dates_house):

            if (date.weekday() == weekday):
                counting = True
            else:
                counting = False

            if counting:
                day_Y = day_Y + y
                day_Y_pred = day_Y_pred + y_hat
            elif (day_Y != 0 and day_Y_pred != 0):
                # if we were acumulating for the day and we are done 
                Y[weekday].append(day_Y)
                Y_pred[weekday].append(day_Y_pred)
                day_Y, day_Y_pred = 0, 0

    _plot_weekday_error_distribution(Y_weekday, Y_pred_weekday, name, save_path)
