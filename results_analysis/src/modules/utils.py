import pandas as pd

def REFIT_aggregate():
    return 'aggregate'

def REFIT_houses():
    return [ 'house_' + str(x) for x in range(1,22) ]

# returns 4 dictionaries:
#   keys: 
#       the same for all 4, the string of the house. e.g. 'house_7'
#   vals:
#       results_true:       true values,
#       results_baseline:   values predicted by the baseline models,
#       reluts_NILM:        values predicted by the enhanced models,
#       dates:              the dates corresponding to the values
def load_results_REFIT(
    results_dir,
    house_dirs = REFIT_houses(),
    path_baseline_results = '/results_baseline/values.csv',
    path_NILM_results = '/results_NILM/values.csv'
    ):
   
    results_true, results_baseline, results_NILM, dates = {}, {}, {}, {}
    for house in house_dirs:

        path_to_house = results_dir + '/' + house

        full_baseline_path = path_to_house + '/' + path_baseline_results
        full_NILM_path = path_to_house + '/' + path_NILM_results
   
        try:
            df_baseline = pd.read_csv(full_baseline_path, parse_dates=['time'])
            df_NILM     = pd.read_csv(full_NILM_path, parse_dates=['time'])
        except:
            print(f'Error: could not load results for {house}')

            results_true[house] = []
            results_baseline[house] = []
            results_NILM[house] = []
            dates[house] = []

            continue
    
        true_baseline = df_baseline['true'].to_list()
        true_NILM = df_NILM['true'].to_list()
        # something has gone very wrong if these assertions fail
        assert(true_NILM == true_baseline)
        results_true[house] = true_baseline
        
        dates_baseline = df_baseline['time'].to_list()
        dates_NILM = df_NILM['time'].to_list()
        assert(dates_baseline == dates_NILM)
        dates[house] = dates_baseline 

        baseline = df_baseline['predicted'].to_list()
        results_baseline[house] = baseline

        NILM = df_NILM['predicted'].to_list()
        results_NILM[house] = NILM

    return (results_true, results_baseline, results_NILM, dates) 

def load_REFIT_results_aggregate(
    results_dir,
    aggregate_dir = REFIT_aggregate(),
    path_baseline_results = '/results_baseline/values.csv',
    path_agg_baseline_results = 'aggregating_baseline_values.csv',
    path_agg_NILM_results = 'aggregating_NILM_values.csv'
    ):

    full_baseline_path = results_dir + '/' + aggregate_dir + '/' + path_baseline_results
    full_agg_baseline_path = results_dir + '/' + aggregate_dir + '/' + path_agg_baseline_results
    full_agg_NILM_path = results_dir + '/' + aggregate_dir + '/' + path_agg_NILM_results

    try:
        df_baseline = pd.read_csv(full_baseline_path, parse_dates=['time'])
        df_agg_baseline = pd.read_csv(full_agg_baseline_path, parse_dates=['time'])
        df_agg_NILM = pd.read_csv(full_agg_NILM_path, parse_dates=['time'])
    except:
        print(f'Error: could not load results for the aggregate')
        exit(1)

    true_baseline = df_baseline['true'].to_list()
    results_baseline = df_baseline['predicted'].to_list()
    dates_baseline = df_baseline['time'].to_list()
    results_baseline = (true_baseline, results_baseline, dates_baseline)
    
    true_agg_baseline = df_agg_baseline['true'].to_list()
    results_agg_baseline = df_agg_baseline['predicted'].to_list()
    dates_agg_baseline = df_agg_baseline['time'].to_list()
    results_agg_baseline = (true_agg_baseline, results_agg_baseline, dates_agg_baseline)
     
    true_agg_NILM = df_agg_NILM['true'].to_list()
    results_agg_NILM = df_agg_NILM['predicted'].to_list()
    dates_agg_NILM = df_agg_NILM['time'].to_list()
    results_agg_NILM = (true_agg_NILM, results_agg_NILM, dates_agg_NILM)
    
    return results_baseline, results_agg_baseline, results_agg_NILM
    
