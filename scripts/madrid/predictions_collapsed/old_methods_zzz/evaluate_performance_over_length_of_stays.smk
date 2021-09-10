'''
Get the performance plots of a single trained classifier for clinical deterioration at multiple patient-stay-slices

Usage : Evaluate the performance of different classifiers (LR, RF, MEWS) on multiple patient-stay-slices 
----------------------------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile evaluate_performance_over_length_of_stays.smk evaluate_performance

'''

import glob
import os
import sys

from config_loader import (
    D_CONFIG,
    DATASET_SITE_PATH, DATASET_SPLIT_PATH,
    DATASET_FEAT_PER_TSLICE_PATH, DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, 
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_SPLIT_PATH, RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH,
    COLLAPSED_FEAT_PER_TIMESLICE_CLF_MODELS_PATH)
    
evaluate_tslice_hours_list=D_CONFIG['EVALUATE_TIMESLICE_LIST']
random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')
COLLAPSED_FEAT_PER_TIMESLICE_CLF_MODELS_PATH = os.path.join(COLLAPSED_FEAT_PER_TIMESLICE_CLF_MODELS_PATH)

rule evaluate_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_performance_over_length_of_stays.py")

    params:
        clf_models_dir=COLLAPSED_FEAT_PER_TIMESLICE_CLF_MODELS_PATH,
        clf_train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        random_seed_list=random_seed_list,
        output_dir=os.path.join(RESULTS_SPLIT_PATH, 'classifier_performance_over_length_of_stays')
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script}\
        --clf_models_dir {params.clf_models_dir}\
        --clf_train_test_split_dir {params.clf_train_test_split_dir}\
        --outcome_column_name {{OUTCOME_COL_NAME}}\
        --random_seed_list "{params.random_seed_list}"\
        --output_dir {params.output_dir}\
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])
