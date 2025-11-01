import pandas as pd

from modules.io.csv_config import Csv_config 

# loads data from a csv file:
#   - If the data is from an individual household the output will represent
#     the consumption from said household
#   - If the data is from multiple household the output will represent the
#     aggregated consumption of all households in the CSV
#
#   The output is in the format [ E, I, D, H ], where:
#   E = the samples (in kWh)
#   I = the time index for the samples (from 0 to config.sampleSize - 1)
#   D = the day of the week corresponding to the time index (from 0 to 6)
#   H = binary holiday mark for the time index (1 if the date is on a holiday)
def load_data_from_csv(filePath, config):
    print('loading csv file: ' + filePath)
    csv_df = pd.read_csv(filePath, parse_dates=[config.dtField])

    # define the range for which we load daily load profile
    # TODO: find a way to remove freq='D' and put it in the csv_config
    start_date = csv_df[config.dtField].min()
    end_date = csv_df[config.dtField].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    samples = []
    time_index = []
    weekday_index = []
    holiday_index = []
    for day in date_range:
        # get all the readings that correspond to the current date
        day_readings = csv_df[csv_df[config.dtField].dt.date == day.date()]
        
        time_range = pd.date_range(
                start=day,
                periods=config.sampleSize,
                freq=config.resolution)

        # initialize fields for the readings for the current day
        current_time = 0 
        current_weekday = day.dayofweek
        day_holiday = is_holiday(day)
        for time in time_range:
            # now get all the readings that correspond to the current time
            current_reading = day_readings[day_readings[config.dtField] == time]

            current_usage = current_reading[config.loadField].sum()

            samples.append(current_usage)
            time_index.append(current_time)
            weekday_index.append(current_weekday)
            holiday_index.append(day_holiday)

            current_time = current_time + 1

    return [ samples, time_index, weekday_index, holiday_index ]

#TODO: implement holiday markers
def is_holiday(day):
    return 0 
