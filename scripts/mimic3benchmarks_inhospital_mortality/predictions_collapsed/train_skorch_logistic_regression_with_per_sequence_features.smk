'''
Train logistic regression on collapsed features for mimic3 inhospital mortality

Usage
-----
snakemake --cores 1 --snakefile train_skorch_logistic_regression_with_per_sequence_features.smk train_and_evaluate_classifier_many_hyperparams

snakemake --snakefile train_skorch_logistic_regression_with_per_sequence_features.smk --profile ../../../profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams

'''


configfile : "skorch_logistic_regression_single_config.json"

from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, 
    DATASET_SPLIT_FEAT_PER_TSLICE_PATH,
    RESULTS_FEAT_PER_TSLICE_PATH, 
    DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH,
    RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH)

random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH, 'classifier_train_test_split_dir')

RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH="/cluster/tufts/hugheslab/prath01/results/mimic3/skorch_logistic_regression"

print("Training logistic regression")
print("--------------------------")
print("Results and trained model will go to:")
print(RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH)

rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH, "skorch_logistic_regression_lr={lr}-weight_decay={weight_decay}-batch_size={batch_size}-scoring={scoring}-seed={seed}.json").format(lr=lr, weight_decay=weight_decay, batch_size=batch_size, scoring=scoring, seed=seed) for lr in config['lr'] for weight_decay in config['weight_decay'] for batch_size in config['batch_size'] for scoring in config['scoring'] for seed in config['seed']]


rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchLogisticRegression', 'main.py'),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json')

    params:
        output_dir=RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH,
        fn_prefix="skorch_logistic_regression_lr={lr}-weight_decay={weight_decay}-batch_size={batch_size}-scoring={scoring}-seed={seed}"

    output:
        os.path.join(RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH, "skorch_logistic_regression_lr={lr}-weight_decay={weight_decay}-batch_size={batch_size}-scoring={scoring}-seed={seed}.json")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --output_filename_prefix {params.fn_prefix} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --validation_size 0.15 \
            --key_cols_to_group_when_splitting {{SPLIT_KEY_COL_NAMES}} \
            --scoring {wildcards.scoring} \
            --lr {wildcards.lr} \
            --weight_decay {wildcards.weight_decay} \
            --batch_size {wildcards.batch_size} \
            --merge_x_y False \
            --seed {wildcards.seed} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])