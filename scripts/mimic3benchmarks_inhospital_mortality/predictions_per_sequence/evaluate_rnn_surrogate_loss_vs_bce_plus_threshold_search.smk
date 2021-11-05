'''
Train full sequence classifier on Madrid transfer to ICU task

Usage: 

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores 1 --snakefile evaluate_rnn_surrogate_loss_vs_bce_plus_threshold_search.smk

'''


sys.path.append(os.path.abspath('../predictions_collapsed'))

sys.path.append('../predictions_collapsed/')
from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, 
    DATASET_FEAT_PER_TSTEP_DYNAMIC_INPUT_OUTPUT_PATH)

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_FEAT_PER_TSTEP_DYNAMIC_INPUT_OUTPUT_PATH, 'classifier_train_test_split_dir')
RESULTS_FEAT_PER_TSTEP_PATH = "/cluster/tufts/hugheslab/prath01/results/mimic3/features_per_tstep_dynamic_input_output/rnn_per_tstep/"


rule evaluate_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_rnn_surrogate_loss_vs_bce_plus_threshold_search.py")

    params:
        clf_models_dir=RESULTS_FEAT_PER_TSTEP_PATH,
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