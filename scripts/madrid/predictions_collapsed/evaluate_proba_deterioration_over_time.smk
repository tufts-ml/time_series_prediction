'''
Get the performance plots of a single trained classifier for clinical deterioration at multiple patient-stay-slices

'''

import glob
import os
import sys

from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    DATASET_PERTSTEP_SPLIT_PATH, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_PATH, RESULTS_PERTSTEP_PATH, CLF_TRAIN_TEST_SPLIT_PATH)
    
rule evaluate_proba_deterioration:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_proba_deterioration_over_time.py")

    params:
        clf_models_dir=RESULTS_PERTSTEP_PATH,
        clf_train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        preproc_data_dir=DATASET_STD_PATH
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        python -u {input.script}\
        --clf_models_dir {params.clf_models_dir}\
        --clf_train_test_split_dir {params.clf_train_test_split_dir}\
        --preproc_data_dir {params.preproc_data_dir}\
        --outcome_column_name {{OUTCOME_COL_NAME}}\
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])