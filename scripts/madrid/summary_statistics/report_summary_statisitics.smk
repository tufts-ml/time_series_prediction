'''
Report summary statistics on madrid data

>> snakemake --cores 1 --snakefile report_summary_statisitics.smk compute_summary_statistics
'''

import glob
import os
import sys

config_loader_dir=os.path.join(os.path.abspath('../'), 'predictions_collapsed')
sys.path.append(config_loader_dir)

from config_loader import (
    D_CONFIG, DATASET_TOP_PATH,
    DATASET_SITE_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_TOP_PATH)


rule compute_summary_statistics:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'summary_statistics.py'),
        
    params:
        preproc_data_dir=DATASET_SITE_PATH,
        data_dicts_dir=DATASET_SITE_PATH,
        output_dir=RESULTS_TOP_PATH

    shell:
        '''
        python -u {{input.script}} \
            --preproc_data_dir {{params.preproc_data_dir}} \
            --data_dicts_dir {{params.data_dicts_dir}} \
            --random_seed {split_random_state} \
            --group_cols {split_key_col_names} \
            --test_size {split_test_size} \
            --output_dir {{params.output_dir}}\
        '''.format(
            split_random_state=D_CONFIG['SPLIT_RANDOM_STATE'],
            split_test_size=D_CONFIG['SPLIT_TEST_SIZE'],
            split_key_col_names=D_CONFIG['SPLIT_KEY_COL_NAMES'],
            )
