  
'''
Compare BCE , BCE+Threshold search, Surrogate Loss for LR
----------------------------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile evaluate_skorch_lr_surrogate_loss_vs_bce_plus_threshold_search.smk evaluate_performance
'''

from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, 
    DATASET_SPLIT_FEAT_PER_TSLICE_PATH,
    RESULTS_FEAT_PER_TSLICE_PATH, 
    DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH,
    RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH)


CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH, 'classifier_train_test_split_dir')
RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH="/cluster/tufts/hugheslab/prath01/results/mimic3/skorch_logistic_regression"


rule evaluate_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_skorch_lr_surrogate_loss_vs_bce_plus_threshold_search.py")

    params:
        clf_models_dir=RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH,
        clf_train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        python -u {input.script}\
        --clf_models_dir {params.clf_models_dir}\
        --clf_train_test_split_dir {params.clf_train_test_split_dir}\
        --outcome_column_name {{OUTCOME_COL_NAME}}\
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])