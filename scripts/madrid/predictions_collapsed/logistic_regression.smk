'''
Train collapsed feature classifier on Madrid transfer to ICU task

>> snakemake --cores 1 --snakefile logistic_regression.smk
'''

# Default environment variables
# Can override with local env variables

from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH, PROJECT_CONDA_ENV_YAML,
    DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH,
    RESULTS_SPLIT_PATH, PROJECT_REPO_DIR,
    RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH)
#tstep_hours_list=D_CONFIG['TIMESTEP_LIST']
random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']

RESULTS_SPLIT_PATH=os.path.join(RESULTS_SPLIT_PATH, 'logistic_regression')
RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH = os.path.join(RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH, 'logistic_regression')
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')


rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'eval_classifier.py'),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json')

    params:
        output_dir=RESULTS_SPLIT_PATH,
        random_seed=int(random_seed_list[0])

    output:
        output_html=os.path.join(RESULTS_SPLIT_PATH, "report.html")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            logistic_regression \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --merge_x_y False \
            --validation_size 0.15 \
            --key_cols_to_group_when_splitting {{SPLIT_KEY_COL_NAMES}} \
            --random_seed {params.random_seed}\
            --n_splits 2 \
            --scoring roc_auc \
            --threshold_scoring balanced_accuracy \
            --class_weight balanced \
            --tol 0.001\
            --max_iter 10000
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])