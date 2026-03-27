import sys
import os

from modules.utils import load_REFIT_results_aggregate as load_results

from modules.error_distribution import weekday_error_distribution
from modules.error_distribution import cumulative_weekday_error_distribution

def main():

    if (len(sys.argv) < 3):
        print(f'Error: too few arguments to run, 2 args needed')
        exit(1)

    results_dir = sys.argv[1]
    out_dir = sys.argv[2]

    results = load_results(results_dir)
    
    Y_baseline, Y_pred_baseline, dates_baseline = results[0]
    Y_agg_baseline, Y_pred_agg_baseline, dates_agg_baseline = results[1]
    Y_agg_NILM, Y_pred_agg_NILM, dates_agg_NILM = results[2]
   
    save_dir = out_dir + '/error_distribution/'
    os.makedirs(save_dir, exist_ok=True)
   
    weekday_error_distribution(
        Y=Y_baseline,
        Y_pred=Y_pred_baseline,
        dates=dates_baseline,
        name='Baseline aggregate weekday error distribution',
        save_path=save_dir)
    weekday_error_distribution(
        Y=Y_agg_baseline,
        Y_pred=Y_pred_agg_baseline,
        dates=dates_agg_baseline,
        name='Summed baseline weekday error distribution',
        save_path=save_dir)
    weekday_error_distribution(
        Y=Y_agg_NILM,
        Y_pred=Y_pred_agg_NILM,
        dates=dates_agg_NILM,
        name='Summed enhanced weekday error distribution',
        save_path=save_dir)

    cumulative_weekday_error_distribution(
        Y=Y_baseline,
        Y_pred=Y_pred_baseline,
        dates=dates_baseline,
        name='Baseline aggregate cumulative weekday error distribution',
        save_path=save_dir)
    cumulative_weekday_error_distribution(
        Y=Y_agg_baseline,
        Y_pred=Y_pred_agg_baseline,
        dates=dates_agg_baseline,
        name='Summed baseline cumulative weekday error distribution',
        save_path=save_dir)
    cumulative_weekday_error_distribution(
        Y=Y_agg_NILM,
        Y_pred=Y_pred_agg_NILM,
        dates=dates_agg_NILM,
        name='Summed enhanced cumulative weekday error distribution',
        save_path=save_dir)

if __name__ == '__main__':
    main()
