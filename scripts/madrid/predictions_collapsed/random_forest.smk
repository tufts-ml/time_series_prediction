'''
Train collapsed feature classifier on Madrid transfer to ICU task

'''

# Default environment variables
# Can override with local env variables

import glob

from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    DATASET_PERTSTEP_SPLIT_PATH, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_PATH, RESULTS_PERTSTEP_PATH)
#tstep_hours_list=D_CONFIG['TIMESTEP_LIST']
tstep_hours_list=[-48, 48]
#random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
random_seed_list=[42]
RESULTS_PATH=os.path.join(RESULTS_PATH, 'random_forest')
RESULTS_PERTSTEP_PATH=os.path.join(RESULTS_PERTSTEP_PATH, 'random_forest')
output_html_files=[os.path.join(RESULTS_PERTSTEP_PATH, "TSTEP={tstep_hours}", "report_random_seed={random_seed}.html").format(tstep_hours=tstep_hours, random_seed=random_seed) for tstep_hours in tstep_hours_list for random_seed in random_seed_list]


rule train_and_evaluate_classifier_many_tsteps:
    input:
        output_html_files

rule train_and_evaluate_classifier_single_tstep:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'eval_classifier.py'),
        x_train_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'y_test.csv'),
        x_dict_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'x_dict.json'),
        y_dict_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'y_dict.json')

    params:
        output_dir=os.path.join(RESULTS_PERTSTEP_PATH, "TSTEP={tstep_hours}")

    output:
        output_html=os.path.join(RESULTS_PERTSTEP_PATH, "TSTEP={tstep_hours}", "report_random_seed={random_seed}.html")

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
            --validation_size 0.15 \
            --key_cols_to_group_when_splitting {{SPLIT_KEY_COL_NAMES}} \
            --random_seed {wildcards.random_seed}\
            --n_splits 3 \
            --scoring roc_auc \
            --threshold_scoring balanced_accuracy \
            --class_weight balanced \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])


rule train_and_evaluate_classifier_full_history:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'eval_classifier.py'),
        x_train_csv=os.path.join(DATASET_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_PATH, 'y_test.csv'),
        x_dict_json=os.path.join(DATASET_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(DATASET_SPLIT_PATH, 'y_dict.json')

    output:
        os.path.join(RESULTS_PATH, 'report.html')

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {{RESULTS_PATH}} && \
        python -u {input.script} \
            random_forest \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {{RESULTS_PATH}} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --validation_size 0.15 \
            --key_cols_to_group_when_splitting "patient_id" \
            --n_splits 3 \
            --scoring roc_auc \
            --threshold_scoring balanced_accuracy \
            --class_weight balanced \
        '''.replace("{{RESULTS_PATH}}", RESULTS_PATH)\
           .replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])
