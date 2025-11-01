# TODO:
#   Remove the sampleSize field from the csv_config,
#   sample size should be inferred from the resolution

# POD that contains:
#   - The resolution of the data in the CSV
#   - The amount of samples a day contains 
#   - The field that defines the datetime
#   - The field that defines the consumed electricity 
class Csv_config():
    def __init__(self, dtField, loadField, sampleSize, resolution):
        self.dtField = dtField
        self.loadField = loadField
        self.sampleSize = sampleSize
        self.resolution = resolution


