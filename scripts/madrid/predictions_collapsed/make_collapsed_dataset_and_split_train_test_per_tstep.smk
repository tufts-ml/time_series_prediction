'''
Produce a collapsed feature representation on Madrid Transfer to ICU Prediction
'''

# Default environment variables
# Can override with local env variables

from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_PERTSTEP_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML)

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(DATASET_PERTSTEP_SPLIT_PATH)

# run collapse features on furst 2 hours, first 4 hours etc.
tstep_hours_list=D_CONFIG['TIMESTEP_LIST']
#tstep_hours_list=[2]

collapsed_pertstep_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","CollapsedFeaturesPerSequence.csv").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]
collapsed_pertstep_jsons=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","Spec_CollapsedFeaturesPerSequence.json").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]
mews_pertstep_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","MewsScoresPerSequence.csv").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]
mews_pertstep_jsons=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","Spec_MewsScoresPerSequence.json").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]
train_test_split_jsons=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","{data_dict}").format(tstep_hours=str(tstep_hours), data_dict=data_dict) for tstep_hours in tstep_hours_list for data_dict in ['x_dict.json', 'y_dict.json', 'mews_dict.json']]
train_test_split_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","{data_csv}").format(tstep_hours=str(tstep_hours), data_csv=data_csv) for tstep_hours in tstep_hours_list for data_csv in ['x_train.csv', 'x_test.csv', 'y_train.csv', 'y_test.csv', 'mews_train.csv', 'mews_test.csv']]
missingness_pertstep_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "is_available_per_feature.csv").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]

rule compute_missingness_many_tsteps:
    input:
        missingness_pertstep_csvs

rule compute_mews_score_many_tsteps:
    input:
        mews_pertstep_csvs

rule collapse_features_and_split_into_train_test_many_tsteps:
    input:
        collapsed_pertstep_csvs,
        collapsed_pertstep_jsons,
        mews_pertstep_csvs,
        mews_pertstep_jsons,
        train_test_split_jsons,
        train_test_split_csvs,

rule collapse_features:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'feature_transformation.py'),
        x_csv=os.path.join(DATASET_STD_PATH, 'vitals_before_icu.csv'),
        x_spec_json=os.path.join(DATASET_STD_PATH, 'Spec-Vitals.json')

    output:
        collapsedx_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "CollapsedFeaturesPerSequence.csv"),
        collapsedx_pertstep_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "Spec_CollapsedFeaturesPerSequence.json")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.x_csv} \
            --data_dict {input.x_spec_json} \
            --output "{output.collapsedx_pertstep_csv}" \
            --data_dict_output "{output.collapsedx_pertstep_json}" \
            --max_time_step "{wildcards.tstep_hours}"\
            --collapse_range_features "hours_since_measured present slope std median min max" \
            --range_pairs "[(0,10), (0,25), (0,50), (50,100), (75,100), (90,100), (0,100)]" \
            --collapse
        '''

rule compute_mews_score:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'compute_mews_score.py'),
        x_csv=os.path.join(DATASET_STD_PATH, 'vitals_before_icu.csv'),
        x_spec_json=os.path.join(DATASET_STD_PATH, 'Spec-Vitals.json')
    
    params:
        output_dir=DATASET_PERTSTEP_SPLIT_PATH

    output:
        mews_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "MewsScoresPerSequence.csv"),
        mews_pertstep_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "Spec_MewsScoresPerSequence.json")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.x_csv} \
            --data_dict {input.x_spec_json} \
            --output  "{output.mews_pertstep_csv}" \
            --data_dict_output "{output.mews_pertstep_json}" \
            --max_time_step "{wildcards.tstep_hours}"\
        '''

rule compute_missingness:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'report_missingness.py'),
        x_csv=os.path.join(DATASET_STD_PATH, 'vitals_before_icu.csv'),
        x_spec_json=os.path.join(DATASET_STD_PATH, 'Spec-Vitals.json')

    params:
        output_dir=DATASET_PERTSTEP_SPLIT_PATH

    output:
        missingness_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "is_available_per_feature.csv")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.x_csv} \
            --data_dict {input.x_spec_json} \
            --max_time_step "{wildcards.tstep_hours}"\
            --missingness_csv "{output.missingness_csv}"\
        '''

rule evaluate_missingness_many_tsteps:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'evaluate_missingness_pertstep.py'),
    
    params:    
        missingness_dir=DATASET_PERTSTEP_SPLIT_PATH
    
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --missingness_dir "{params.missingness_dir}"\
        '''

rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        collapsedx_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "CollapsedFeaturesPerSequence.csv"),
        collapsedx_pertstep_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "Spec_CollapsedFeaturesPerSequence.json"),
        mews_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "MewsScoresPerSequence.csv"),
        mews_pertstep_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "Spec_MewsScoresPerSequence.json"),
        collapsedy_csv=os.path.join(DATASET_STD_PATH, 'icu_transfer_outcomes.csv'),
        collapsedy_json=os.path.join(DATASET_STD_PATH, 'Spec-Outcomes_TransferToICU.json')

    output:
        x_dict=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'x_dict.json'),
        y_dict=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'y_dict.json'),
        mews_dict=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'mews_dict.json'),
        x_train_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'y_test.csv'),
        mews_train_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'mews_train.csv'),
        mews_test_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'mews_test.csv')

    conda:
        PROJECT_CONDA_ENV_YAML
    
    shell:
        '''
            python -u {{input.script}} \
                --input {{input.collapsedx_pertstep_csv}} \
                --data_dict {{input.collapsedx_pertstep_json}} \
                --random_state {split_random_state} \
                --test_size {split_test_size} \
                --group_cols {split_key_col_names} \
                --train_csv_filename {{output.x_train_csv}} \
                --test_csv_filename {{output.x_test_csv}} \
                --output_data_dict_filename {{output.x_dict}} \
            
            python -u {{input.script}} \
                --input {{input.collapsedy_csv}} \
                --data_dict {{input.collapsedy_json}} \
                --random_state {split_random_state} \
                --test_size {split_test_size} \
                --group_cols {split_key_col_names} \
                --train_csv_filename {{output.y_train_csv}} \
                --test_csv_filename {{output.y_test_csv}} \
                --output_data_dict_filename {{output.y_dict}} \

            python -u {{input.script}} \
                --input {{input.mews_pertstep_csv}} \
                --data_dict {{input.mews_pertstep_json}} \
                --random_state {split_random_state} \
                --test_size {split_test_size} \
                --group_cols {split_key_col_names} \
                --train_csv_filename {{output.mews_train_csv}} \
                --test_csv_filename {{output.mews_test_csv}} \
                --output_data_dict_filename {{output.mews_dict}} \
        '''.format(
            split_random_state=D_CONFIG['SPLIT_RANDOM_STATE'],
            split_test_size=D_CONFIG['SPLIT_TEST_SIZE'],
            split_key_col_names=D_CONFIG['SPLIT_KEY_COL_NAMES'],
            )
