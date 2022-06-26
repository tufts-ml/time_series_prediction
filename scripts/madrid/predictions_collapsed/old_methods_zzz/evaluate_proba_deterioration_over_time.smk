'''
Get the performance plots of a single trained classifier for clinical deterioration at multiple patient-stay-slices

Usage : Evaluate the probability of deterioration over time using multiple classifiers (LR, RF, RNN, GPVAE, PC-HMM...)
----------------------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile evaluate_proba_deterioration_over_time.smk evaluate_proba_deterioration

'''

import glob
import os
import sys

from config_loader import (
    D_CONFIG,
    DATASET_SITE_PATH, DATASET_SPLIT_PATH,
    DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_SPLIT_PATH, RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH,
    DATASET_FEAT_PER_TSLICE_PATH, RESULTS_FEAT_PER_TSTEP_PATH)
    
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')
RNN_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')


rule evaluate_proba_deterioration:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_proba_deterioration_over_time.py")

    params:
        shallow_clf_models_dir=RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH,
        rnn_models_dir=os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, 'rnn'),
        shallow_clf_train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        rnn_train_test_split_dir=RNN_TRAIN_TEST_SPLIT_PATH,
        preproc_data_dir=DATASET_SITE_PATH,
        output_dir=os.path.join(RESULTS_SPLIT_PATH, 'proba_deterioration_over_time')
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        python -u {input.script}\
        --shallow_clf_models_dir {params.shallow_clf_models_dir}\
        --shallow_clf_train_test_split_dir {params.shallow_clf_train_test_split_dir}\
        --rnn_models_dir {params.rnn_models_dir}\
        --rnn_train_test_split_dir {params.rnn_train_test_split_dir}\
        --preproc_data_dir {params.preproc_data_dir}\
        --outcome_column_name {{OUTCOME_COL_NAME}}\
        --output_dir {params.output_dir}\
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])