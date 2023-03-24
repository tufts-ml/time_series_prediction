'''
Train collapsed feature classifier on Madrid transfer to ICU task

>> snakemake --cores 1 --snakefile evaluate_dynamic_prediction_performance_scheduled_vs_transfer_from_er.smk
'''

# Default environment variables
# Can override with local env variables

from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH, PROJECT_CONDA_ENV_YAML,
    DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
    RESULTS_SPLIT_PATH, PROJECT_REPO_DIR,
    RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH)

CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 'classifier_train_test_split')
rule evaluate_dynamic_prediction_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_dynamic_prediction_performance_scheduled_vs_transfer_from_er.py"),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json'),

    params:
        clf_models_dir=RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        output_dir=RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        train_test_split_path=CLF_TRAIN_TEST_SPLIT_PATH

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.clf_models_dir} && \
        python -u {input.script} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --clf_models_dir {params.clf_models_dir} \
            --output_dir {params.output_dir} \
            --train_test_split_path {params.train_test_split_path} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --merge_x_y False \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])