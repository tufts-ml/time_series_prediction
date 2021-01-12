'''
Train random fores on collapsed features for mimic3 inhospital mortality

Usage
-----
snakemake --cores 1 --snakefile train_random_forest.smk train_and_evaluate_classifier

'''

sys.path.append('../predictions_collapsed/')

from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, 
    DATASET_SPLIT_FEAT_PER_TSLICE_PATH,
    RESULTS_FEAT_PER_TSLICE_PATH, 
    DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH,
    RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH)

random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split_dir')
RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH = os.path.join(RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH, 'random_forest')

print("Training logistic regression")
print("--------------------------")
print("Results and trained model will go to:")
print(RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH)

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
        output_dir=RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH,
        random_seed=int(random_seed_list[0])

    output:
        output_html=os.path.join(RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH, "report.html")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            random_forest \
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
            --scoring roc_auc_score \
            --threshold_scoring balanced_accuracy_score \
            --class_weight balanced \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])