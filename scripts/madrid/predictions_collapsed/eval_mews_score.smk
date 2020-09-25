'''
Evaluate mews score
'''

# Default environment variables
# Can override with local env variables

import glob

from config_loader import (
    D_CONFIG,
    DATASET_SITE_PATH, DATASET_SPLIT_PATH,
    DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_SPLIT_PATH, RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH)

random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
RESULTS_SPLIT_PATH=os.path.join(RESULTS_SPLIT_PATH, 'mews')
RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH=os.path.join(RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH, 'mews')
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')

rule evaluate_mews_score:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'eval_mews_score.py'),
        mews_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_train.csv'),
        mews_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_test.csv'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv'),
        mews_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json')
    
    params:
        output_dir=RESULTS_SPLIT_PATH,
        random_seed=int(random_seed_list[0])
        

    output:
        output_csv=os.path.join(RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH, "mews_performance_df.csv")
    
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --train_csv_files {input.mews_train_csv},{input.y_train_csv} \
            --test_csv_files {input.mews_test_csv},{input.y_test_csv} \
            --data_dict_files {input.mews_dict_json},{input.y_dict_json} \
            --merge_x_y False \
            --key_cols_to_group_when_splitting {{SPLIT_KEY_COL_NAMES}} \
            --random_seed {params.random_seed}\
            --scoring roc_auc \
            --threshold_scoring balanced_accuracy \
            --validation_size 0.15\
            --n_splits 3\
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])
