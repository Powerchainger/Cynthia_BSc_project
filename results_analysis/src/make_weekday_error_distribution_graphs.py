import sys
import os

from modules.utils import load_results_REFIT as load_results
from modules.utils import REFIT_houses as houses

from modules.error_distribution import weekday_error_distribution_all
from modules.error_distribution import cumulative_weekday_error_distribution_all

def main():

    if (len(sys.argv) < 3):
        print(f'Error: too few arguments to run, 2 args needed')
        exit(1)

    results_dir = sys.argv[1]
    out_dir = sys.argv[2]

    keys = houses()
    true, baseline, NILM, dates = load_results(results_dir)

    save_dir = out_dir + '/error_distribution/'
    os.makedirs(save_dir, exist_ok=True)
   
    weekday_error_distribution_all(
        Y_dict=true,
        Y_pred_dict=baseline,
        keys=keys,
        dates=dates,
        name='Baseline weekday error distribution',
        save_path=save_dir)

    weekday_error_distribution_all(
        Y_dict=true,
        Y_pred_dict=NILM,
        keys=keys,
        dates=dates,
        name='Enhanced weekday error distribution',
        save_path=save_dir)

    cumulative_weekday_error_distribution_all(
        Y_dict=true,
        Y_pred_dict=baseline,
        keys=keys,
        dates=dates,
        name='Baseline cumulative weekday error distribution',
        save_path=save_dir)

    cumulative_weekday_error_distribution_all(
        Y_dict=true,
        Y_pred_dict=NILM,
        keys=keys,
        dates=dates,
        name='Enhanced cumulative weekday error distribution',
        save_path=save_dir)

if __name__ == '__main__':
    main()
