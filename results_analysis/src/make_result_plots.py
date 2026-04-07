import sys
import os

from modules.utils import load_results_REFIT as load_results
from modules.utils import REFIT_houses as houses

from modules.plots import result_plots_individual 

def main():
    """Creates plots from the results, showing forecasts and true values
    
    Given a directory containing the results, it creates plots for the 
    results for each household, which is then saved to files in their
    respective directories.

    Program arguments:
    results_dir -- the directory containing the results
    out_dir -- directory to save the plots to
    """
    if (len(sys.argv) < 3): 
        print(f'Error: too few arguments to run, 2 args needed')
        exit(1)

    results_dir = sys.argv[1]
    out_dir = sys.argv[2]

    keys = houses()
    true, baseline, NILM, dates = load_results(results_dir)
    save_dir = out_dir + '/result_plots/'
    save_dir_baseline = save_dir + '/baseline/'
    save_dir_NILM = save_dir + '/NILM/'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_baseline, exist_ok=True)
    os.makedirs(save_dir_NILM, exist_ok=True)

    result_plots_individual(
        Y_dict=true,
        Y_pred_dict=baseline,
        dates_dict=dates,
        keys=keys,
        save_path=save_dir_baseline
    )

    result_plots_individual(
        Y_dict=true,
        Y_pred_dict=NILM,
        dates_dict=dates,
        keys=keys,
        save_path=save_dir_NILM
    )
if __name__ == '__main__':
    main()
