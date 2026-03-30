import sys
import os

from modules.utils import load_results_REFIT as load_results
from modules.utils import REFIT_houses as houses

from modules.metrics import save_metrics

def main():
    """Computes the RMSE, MAE, and MAPE for the results

    Given a directory containing the results, it computes the metrics 
    for each house, and then saves it to file

    Program arguments:
    results_dir -- directory containing the results
    out_dir -- directory to save the metrics to
    """
    if (len(sys.argv) < 3):
        print(f'Error: too few arguments to run, 2 args needed')
        exit(1)

    results_dir = sys.argv[1]
    out_dir = sys.argv[2]

    keys = houses()
    true, baseline, NILM, _ = load_results(results_dir)

    save_dir = out_dir + '/metrics/'
    os.makedirs(save_dir, exists_ok=True)

    save_metrics(
        Y_dict=true,
        Y_pred_dict=baseline,
        keys=keys,
        name_prefix='metrics_',
        save_path=save_dir
    )

if __name__ == '__main__':
    main()
