  
'''
Compare Multiple bounds Surrogate Loss for LR
----------------------------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile evaluate_skorch_lr_with_dynamic_input_output_multiple_collapse_feature_ranges.smk evaluate_performance
'''

from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, 
    DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
    RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH)

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 'classifier_train_test_split_dir')

RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH="/cluster/tufts/hugheslab/prath01/results/mimic3/collapsed_features_dynamic_input_output/"


rule evaluate_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_skorch_lr_surrogate_loss_multiple_collapse_feature_ranges.py")

    params:
        clf_models_dir=RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
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