'''
Get summary statistics on madrid data and plot them
'''

import glob
import os

config_loader=os.path.join(os.path.abspath('../'), 'predictions_collapsed', 'config_loader.py')

from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    DATASET_PERTSTEP_SPLIT_PATH, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_PATH, RESULTS_PERTSTEP_PATH, RESULTS_TOP_PATH)


rule compute_summary_statistics:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'summary_statistics.py'),
        
    params:
        preproc_data_dir=DATASET_STD_PATH,
        output_dir=RESULTS_TOP_PATH

    shell:
        '''
        python -u {{input.script}} \
            --preproc_data_dir {{params.preproc_data_dir}} \
            --random_seed {split_random_state} \
            --group_cols {split_key_col_names} \
            --test_size {split_test_size} \
            --output_dir {{params.output_dir}}\
        '''.format(
            split_random_state=D_CONFIG['SPLIT_RANDOM_STATE'],
            split_test_size=D_CONFIG['SPLIT_TEST_SIZE'],
            split_key_col_names=D_CONFIG['SPLIT_KEY_COL_NAMES'],
            )
