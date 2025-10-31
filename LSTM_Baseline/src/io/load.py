import pandas as pd

class csv_config():
    def__init__(self, dtField, loadField, sampleSize, resolution, individual=None):
        self.dtField = dtField
        self.loadField = loadField
        self.sampleSize = sampleSize
        self.resolution = resolution
        self.individual = individual

def load_data_individual(filePath):


# load the aggregate load usage of all customers in a csv
def load_data_aggregate(filePath, config):
    csv_df = pd.load_csv(filePath, parse_dates=config.dtField)

    start_date = csv_df[config.dtField].min()
    end_date = csv_df[config.dtField].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    aggregate_daily_usage = []
    for day in date_range:
        day_readings = csv_df[csv_df[config.dtField].dt.date == day.date()]

        time_range = pd.date_range(
                start=day,
                periods=config.sampleSize,
                freq=resolution)

        daily_usage = []
        for time in time_range:
            current_readings = day_readings[day_readings[config.dtField] == time]

            aggregate_load = current_readings[config.loadField].sum()
            daily_usage.append(aggregate_load)
        
        aggregate_daily_usage.append(daily_usage)

    return aggregate_daily_usage
