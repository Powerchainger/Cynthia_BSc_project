import pandas as pd

def find_next_day_from_results(dates, idx=0):

    while idx < len(dates) and dates[idx][0].time() != pd.to_datetime('00:00').time():
        idx = idx + 1
    
    return idx

