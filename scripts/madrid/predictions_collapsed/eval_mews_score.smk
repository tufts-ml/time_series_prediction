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
tstep_hours_list=D_CONFIG['TIMESTEP_LIST']
#random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
#tstep_hours_list=[24]
random_seed_list=[42]
RESULTS_PATH=os.path.join(RESULTS_PATH, 'mews')
RESULTS_PERTSTEP_PATH=os.path.join(RESULTS_PERTSTEP_PATH, 'mews')
output_csv_files=[os.path.join(RESULTS_PERTSTEP_PATH, "TSTEP={tstep_hours}", "performance_df_random_seed={random_seed}.csv").format(tstep_hours=tstep_hours, random_seed=random_seed) for tstep_hours in tstep_hours_list for random_seed in random_seed_list]

rule evaluate_mews_score_many_tsteps:
    input:
        output_csv_files

rule evaluate_mews_score_single_tstep:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'eval_mews_score.py'),
        mews_train_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'mews_train.csv'),
        mews_test_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'mews_test.csv'),
        y_train_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'y_test.csv'),
        mews_dict_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'mews_dict.json'),
        y_dict_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'y_dict.json')
    
    params:
        output_dir=os.path.join(RESULTS_PERTSTEP_PATH, "TSTEP={tstep_hours}")

    output:
        output_csv=os.path.join(RESULTS_PERTSTEP_PATH, "TSTEP={tstep_hours}", "performance_df_random_seed={random_seed}.csv")
    
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
            --key_cols_to_group_when_splitting {{SPLIT_KEY_COL_NAMES}} \
            --random_seed {wildcards.random_seed}\
            --scoring roc_auc \
            --threshold_scoring balanced_accuracy \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])
