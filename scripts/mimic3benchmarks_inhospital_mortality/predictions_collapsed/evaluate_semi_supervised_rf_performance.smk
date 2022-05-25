'''
>> snakemake --cores 1 --snakefile evaluate_semi_supervised_rf_performance.smk evaluate_performance
'''

import glob
import os
import sys
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH, DATASET_STD_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    DATASET_SPLIT_FEAT_PER_TSLICE_PATH)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
    
evaluate_tslice_hours_list=D_CONFIG['EVALUATE_TIMESLICE_LIST']
random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']

RESULTS_FEAT_PER_TSTEP_PATH="/cluster/tufts/hugheslab/prath01/results/mimic3/random_forest/v05112022/"
CLF_TRAIN_TEST_SPLIT_PATH = "/cluster/tufts/hugheslab/prath01/datasets/mimic3_ssl/percentage_labelled_sequnces=100/"
CLF_MODELS_PATH = RESULTS_FEAT_PER_TSTEP_PATH

rule evaluate_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_semi_supervised_rf_performance.py")

    params:
        clf_models_dir=CLF_MODELS_PATH,
        clf_train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        tslice_folder=DATASET_SPLIT_FEAT_PER_TSLICE_PATH,
        preproc_data_dir=DATASET_STD_PATH,
        random_seed_list=random_seed_list,
        output_dir=os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, 'classifier_per_tslice_performance')
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script}\
        --clf_models_dir {params.clf_models_dir}\
        --clf_train_test_split_dir {params.clf_train_test_split_dir}\
        --tslice_folder {params.tslice_folder}\
        --preproc_data_dir {params.preproc_data_dir}\
        --outcome_column_name {{OUTCOME_COL_NAME}}\
        --random_seed_list "{params.random_seed_list}"\
        --output_dir {params.output_dir}\
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])