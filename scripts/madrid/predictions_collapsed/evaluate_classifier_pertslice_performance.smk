'''
Get the performance plots of a single trained classifier for clinical deterioration at multiple patient-stay-slices

'''

import glob
import os
import sys

from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_SPLIT_PATH, RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH)
    
evaluate_tslice_hours_list=D_CONFIG['EVALUATE_TIMESLICE_LIST']
random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')

rule evaluate_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_classifier_pertslice_performance.py")

    params:
        clf_models_dir=RESULTS_SPLIT_PATH,
        clf_train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        evaluation_tslices=evaluate_tslice_hours_list,
        tslice_folder=DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH,
        preproc_data_dir=DATASET_STD_PATH,
        random_seed_list=random_seed_list
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        python -u {input.script}\
        --clf_models_dir {params.clf_models_dir}\
        --clf_train_test_split_dir {params.clf_train_test_split_dir}\
        --tslice_folder {params.tslice_folder}\
        --evaluation_tslices "{params.evaluation_tslices}"\
        --preproc_data_dir {params.preproc_data_dir}\
        --outcome_column_name {{OUTCOME_COL_NAME}}\
        --random_seed_list "{params.random_seed_list}"\
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])
