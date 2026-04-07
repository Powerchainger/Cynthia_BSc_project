import sys
import os

from modules.utils import load_REFIT_results_aggregate as load_results
from modules.plots import plot_results


def main():
    """Creates plots from the results, for the aggregate households
    
    Given a directory containing the results, it creates plots for the 
    results for the aggregate, for the baseline model, aggregated 
    baseline, and the aggregated NILM results. Which is then stored in
    their respective directories

    Program arguments:
    results_dir -- the directory containing the results
    out_dir -- directory to save the plots to
    """
    if (len(sys.argv) < 3): 
        print(f'Error: too few arguments to run, 2 args needed')
        exit(1)

    results_dir = sys.argv[1]
    out_dir = sys.argv[2]

    results = load_results(results_dir)
    Y_baseline, Y_pred_baseline, dates_baseline = results[0]
    Y_agg_baseline, Y_pred_agg_baseline, dates_baseline_agg = results[1]
    Y_agg_NILM, Y_pred_agg_NILM, dates_NILM_agg = results[2]

    save_dir = out_dir + '/plot_results_aggregate/'
    os.makedirs(save_dir, exist_ok=True)

    plot_results(
        Y=Y_baseline,
        Y_pred=Y_pred_baseline,
        dates=dates_baseline,
        save_path=save_dir,
        name = 'Baseline'
    )
    plot_results(
        Y=Y_agg_baseline,
        Y_pred=Y_pred_agg_baseline,
        dates=dates_baseline_agg,
        save_path=save_dir,
        name = 'Baseline_aggregated'
    )
    plot_results(
        Y=Y_agg_NILM,
        Y_pred=Y_pred_agg_NILM,
        dates=dates_NILM_agg,
        save_path=save_dir,
        name = 'NILM_aggregated'
    )

if __name__ == '__main__':
    main()
