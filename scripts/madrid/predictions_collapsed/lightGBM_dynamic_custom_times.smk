'''
Train collapsed feature classifier on Madrid transfer to ICU task

>> snakemake --cores 1 --snakefile lightGBM_dynamic_custom_times.smk
'''

# Default environment variables
# Can override with local env variables

configfile:"lightGBM.json"

from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH, PROJECT_CONDA_ENV_YAML,
    DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
    RESULTS_SPLIT_PATH, PROJECT_REPO_DIR,
    RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH)


RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH = os.path.join(RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 'lightGBM', 'CustomTimes_10_6')
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 'classifier_train_test_split')

rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, "lightGBM_min_samples_per_leaf={min_samples_per_leaf}-max_leaves={max_leaves}-n_estimators={n_estimators}-frac_features_for_clf={frac_features_for_clf}-frac_training_samples_per_tree={frac_training_samples_per_tree}-features_to_include={features_to_include}_perf.csv").format(min_samples_per_leaf=min_samples_per_leaf, max_leaves=max_leaves, n_estimators=n_estimators, frac_training_samples_per_tree=frac_training_samples_per_tree, frac_features_for_clf=frac_features_for_clf, features_to_include=features_to_include) for min_samples_per_leaf in config['min_samples_per_leaf'] for max_leaves in config['max_leaves'] for n_estimators in config['n_estimators'] for frac_features_for_clf in config['frac_features_for_clf'] for frac_training_samples_per_tree in config['frac_training_samples_per_tree'] for features_to_include in config['features_to_include']]

rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'lightGBM', 'main.py'),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_trainCustomTimes_10_6_{features_to_include}.csv.gz'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_validCustomTimes_10_6_{features_to_include}.csv.gz'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_testCustomTimes_10_6_{features_to_include}.csv.gz'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_trainCustomTimes_10_6_{features_to_include}.csv.gz'),
        y_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_validCustomTimes_10_6_{features_to_include}.csv.gz'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_testCustomTimes_10_6_{features_to_include}.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dictCustomTimes_10_6_{features_to_include}.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dictCustomTimes_10_6_{features_to_include}.json')

    params:
        output_dir=RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        fn_prefix="lightGBM_min_samples_per_leaf={min_samples_per_leaf}-max_leaves={max_leaves}-n_estimators={n_estimators}-frac_features_for_clf={frac_features_for_clf}-frac_training_samples_per_tree={frac_training_samples_per_tree}-features_to_include={features_to_include}"

    output:
        os.path.join(RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, "lightGBM_min_samples_per_leaf={min_samples_per_leaf}-max_leaves={max_leaves}-n_estimators={n_estimators}-frac_features_for_clf={frac_features_for_clf}-frac_training_samples_per_tree={frac_training_samples_per_tree}-features_to_include={features_to_include}_perf.csv")

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
            --valid_csv_files {input.x_valid_csv},{input.y_valid_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --key_cols_to_group_when_splitting {{SPLIT_KEY_COL_NAMES}} \
            --merge_x_y False \
            --min_samples_per_leaf {wildcards.min_samples_per_leaf} \
            --max_leaves {wildcards.max_leaves} \
            --n_estimators {wildcards.n_estimators} \
            --frac_training_samples_per_tree {wildcards.frac_training_samples_per_tree} \
            --frac_features_for_clf {wildcards.frac_features_for_clf} \
            --n_splits 1 \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])
