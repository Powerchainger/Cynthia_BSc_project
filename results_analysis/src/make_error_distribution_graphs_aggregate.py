import sys
import os

from modules.utils import load_REFIT_results_aggregate as load_results

from modules.error_distribution import error_distribution
from modules.error_distribution import cumulative_error_distribution


def main():

    if (len(sys.argv) < 3):
        print(f'Error: too few arguments to run, 2 args needed')
        exit(1)

    results_dir = sys.argv[1]
    out_dir = sys.argv[2]

    results = load_results(results_dir)
    
    Y_baseline, Y_pred_baseline, _ = results[0]
    Y_agg_baseline, Y_pred_agg_baseline, _ = results[1]
    Y_agg_NILM, Y_pred_agg_NILM, _ = results[2]

    save_dir = out_dir + '/error_distribution/'
    os.makedirs(save_dir, exist_ok=True)

    error_distribution(
        Y=Y_baseline,
        Y_pred=Y_pred_baseline,
        name='Baseline aggregate error distribution',
        save_path=save_dir)
    error_distribution(
        Y=Y_agg_baseline,
        Y_pred=Y_pred_agg_baseline,
        name='Summed baseline error distribution',
        save_path=save_dir)
    error_distribution(
        Y=Y_agg_NILM,
        Y_pred=Y_pred_agg_NILM,
        name='Summed enhanced error distribution',
        save_path=save_dir)

    cumulative_error_distribution(
        Y=Y_baseline,
        Y_pred=Y_pred_baseline,
        name='Baseline aggregate daily error distribution',
        save_path=save_dir)
    cumulative_error_distribution(
        Y=Y_agg_baseline,
        Y_pred=Y_pred_agg_baseline,
        name='Summed baseline daily error distribution',
        save_path=save_dir)
    cumulative_error_distribution(
        Y=Y_agg_NILM,
        Y_pred=Y_pred_agg_NILM,
        name='Summed enhanced daily error distribution',
        save_path=save_dir)

if __name__ == '__main__':
    main()
