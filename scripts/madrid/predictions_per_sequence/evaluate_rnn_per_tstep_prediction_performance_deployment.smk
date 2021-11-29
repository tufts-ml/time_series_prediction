'''
Train collapsed feature classifier on Madrid transfer to ICU task

>> snakemake --cores 1 --snakefile evaluate_rnn_per_tstep_prediction_performance_deployment.smk
'''

# Default environment variables
# Can override with local env variables

sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    DATASET_SITE_PATH, DATASET_SPLIT_PATH,
    DATASET_FEAT_PER_TSTEP_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_SPLIT_PATH, RESULTS_FEAT_PER_TSTEP_PATH)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))

random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_FEAT_PER_TSTEP_PATH, 'classifier_train_test_split')

rule evaluate_rnn_per_tstep_prediction_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_rnn_per_tstep_prediction_performance_deployment.py"),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv.gz'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_valid.csv.gz'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv.gz'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv.gz'),
        y_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_valid.csv.gz'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json'),

    params:
        clf_models_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.clf_models_dir} && \
        python -u {input.script} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --clf_models_dir {params.clf_models_dir} \
            --output_dir {params.output_dir} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --valid_csv_files {input.x_valid_csv},{input.y_valid_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --merge_x_y False \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])