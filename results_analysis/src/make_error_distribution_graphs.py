import sys
import os

from modules.utils import load_results_REFIT as load_results
from modules.utils import REFIT_houses as houses 

from modules.error_distribution import error_distribution_all
from modules.error_distribution import cumulative_error_distribution_all 

def main():

    if (len(sys.argv) < 3):
        print(f'Error: too few arguments to run, 2 args needed')
        exit(1)

    results_dir = sys.argv[1]
    out_dir = sys.argv[2]

    keys = houses()
    true, baseline, NILM, _ = load_results(results_dir)
 
    save_dir = out_dir + '/error_distribution/'
    os.makedirs(save_dir, exist_ok=True)
    
    error_distribution_all(
        Y_dict=true,
        Y_pred_dict=baseline,
        keys=keys,
        name='baseline error distribution',
        save_path=save_dir
    )

    error_distribution_all(
        Y_dict=true,
        Y_pred_dict=NILM,
        keys=keys,
        name='enhanced error distribution',
        save_path=save_dir
    )

    cumulative_error_distribution_all(
        Y_dict=true,
        Y_pred_dict=baseline,
        keys=keys,
        name='baseline daily error distribution',
        save_path=save_dir
    )

    cumulative_error_distribution_all(
        Y_dict=true,
        Y_pred_dict=NILM,
        keys=keys,
        name='enhanced daily error distribution', 
        save_path=save_dir
    )

if __name__ == '__main__':
    main()
