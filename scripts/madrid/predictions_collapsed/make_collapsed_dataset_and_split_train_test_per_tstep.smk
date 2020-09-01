'''
Produce a collapsed feature representation on Madrid Transfer to ICU Prediction

----------------------------------------------------------------------------------------------------------------------------------------
COLLAPSING FEATURES AND SPLITTING INTO TRAIN AND TEST
---------------------------------------------------------------------------------------------------------------------------------------

Usage : Maintaining minimum required stay length for evey patient stay slice
---------------------------------------------------------------------------
For eg. If we predict using first 4 hours of data, we ensure that every patient in the cohort has atleast 4 hours of data.
>> snakemake --cores all --snakefile make_collapsed_dataset_and_split_train_test_per_tstep.smk filter_admissions_by_stay_length_many_tsteps

Usage : Collapsing features and saving to slice specific folders
----------------------------------------------------------------
>> snakemake --cores all --snakefile make_collapsed_dataset_and_split_train_test_per_tstep.smk collapse_features_many_tsteps

Usage : Computing slice specific MEWS scores
--------------------------------------------
>> snakemake --cores all --snakefile make_collapsed_dataset_and_split_train_test_per_tstep.smk compute_mews_score_many_tsteps

Usage : Dividng the slice specific collapsed features and MEWS score into train and test
----------------------------------------------------------------------------------------
>> snakemake --cores all --snakefile make_collapsed_dataset_and_split_train_test_per_tstep.smk split_into_train_test_many_tsteps

Usage : Do every step above in squence
-------------------------------------
>> snakemake --cores all --snakefile make_collapsed_dataset_and_split_train_test_per_tstep.smk all

--------------------------------------------------------------------------------------------------------------------------------------
GETTING MISSINGNESS STATISTICS
--------------------------------------------------------------------------------------------------------------------------------------

Usage : Get slice specific missingness statistics
-------------------------------------------------
>> snakemake --force --cores all --snakefile make_collapsed_dataset_and_split_train_test_per_tstep.smk compute_missingness_many_tsteps
>> snakemake --force --cores all --snakefile make_collapsed_dataset_and_split_train_test_per_tstep.smk evaluate_missingness_many_tsteps

'''

# Default environment variables
# Can override with local env variables

from config_loader import (
    D_CONFIG, DATASET_TOP_PATH,
    DATASET_STD_PATH, DATASET_PERTSTEP_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML)

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(DATASET_PERTSTEP_SPLIT_PATH)

# run collapse features on first/last 2 hours, first 4 hours etc.
#tstep_hours_list=D_CONFIG['TIMESTEP_LIST']
tstep_hours_list=['%20', '40%', '60%', '80%', -12, '-6']
filtered_pertstep_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","vitals_before_icu_filtered_{tstep_hours}_hours.csv").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]
collapsed_pertstep_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","CollapsedVitalsPerSequence.csv").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]
collapsed_pertstep_jsons=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","Spec_CollapsedVitalsPerSequence.json").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]
mews_pertstep_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","MewsScoresPerSequence.csv").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]
mews_pertstep_jsons=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","Spec_MewsScoresPerSequence.json").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]
train_test_split_jsons=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","{data_dict}").format(tstep_hours=str(tstep_hours), data_dict=data_dict) for tstep_hours in tstep_hours_list for data_dict in ['x_dict.json', 'y_dict.json', 'mews_dict.json']]
train_test_split_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}","{data_csv}").format(tstep_hours=str(tstep_hours), data_csv=data_csv) for tstep_hours in tstep_hours_list for data_csv in ['x_train.csv', 'x_test.csv', 'y_train.csv', 'y_test.csv', 'mews_train.csv', 'mews_test.csv']]
missingness_pertstep_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "is_available_per_feature.csv").replace("{tstep_hours}", str(tstep_hours)) for tstep_hours in tstep_hours_list]

rule compute_missingness_many_tsteps:
    input:
        missingness_pertstep_csvs

rule filter_admissions_by_tslice_many_tslices:
    input:
        filtered_pertstep_csvs

rule collapse_features_many_tslices:
    input:
        collapsed_pertstep_csvs,
        collapsed_pertstep_jsons

rule compute_mews_score_many_tsteps:
    input:
        mews_pertstep_csvs,
        mews_pertstep_jsons

rule split_into_train_test_many_tsteps:
    input:
        train_test_split_csvs,
        train_test_split_jsons

rule all:
    input:
        filtered_pertstep_csvs,
        collapsed_pertstep_csvs,
        collapsed_pertstep_jsons,
        mews_pertstep_csvs,
        mews_pertstep_jsons,
        train_test_split_jsons,
        train_test_split_csvs,

rule filter_admissions_by_tslice:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'filter_admissions_by_tslice.py'),
        demographics_csv=os.path.join(DATASET_STD_PATH, 'demographics_before_icu.csv'),
        labs_csv=os.path.join(DATASET_STD_PATH, 'labs_before_icu.csv'),
        vitals_csv=os.path.join(DATASET_STD_PATH, 'vitals_before_icu.csv'),
        y_csv=os.path.join(DATASET_STD_PATH, 'clinical_deterioration_outcomes.csv'),
        demographics_spec_json=os.path.join(DATASET_TOP_PATH, 'Spec-Demographics.json'),
        labs_spec_json=os.path.join(DATASET_TOP_PATH, 'Spec-Labs.json'),
        vitals_spec_json=os.path.join(DATASET_TOP_PATH, 'Spec-Vitals.json')
    
    params:
        preproc_data_dir = DATASET_STD_PATH,
        output_dir=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}")
    
    output:
        filtered_demographics_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "demographics_before_icu_filtered_{tstep_hours}_hours.csv"),
        filtered_vitals_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "vitals_before_icu_filtered_{tstep_hours}_hours.csv"),
        filtered_labs_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "labs_before_icu_filtered_{tstep_hours}_hours.csv"),
        filtered_y_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "clinical_deterioration_outcomes_filtered_{tstep_hours}_hours.csv")
    
    shell:
        '''
            python -u {input.script} \
                --preproc_data_dir {params.preproc_data_dir} \
                --tslice "{wildcards.tstep_hours}" \
                --output_dir {params.output_dir} \
        '''

rule collapse_features:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'feature_transformation.py'),
        vitals_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "vitals_before_icu_filtered_{tstep_hours}_hours.csv"),
        vitals_spec_json=os.path.join(DATASET_TOP_PATH, 'Spec-Vitals.json'),
        labs_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "labs_before_icu_filtered_{tstep_hours}_hours.csv"),
        labs_spec_json=os.path.join(DATASET_TOP_PATH, 'Spec-Labs.json'),
        tstop_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "tstops_filtered_{tstep_hours}_hours.csv")

    output:
        collapsed_vitals_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "CollapsedVitalsPerSequence.csv"),
        collapsed_vitals_pertstep_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "Spec_CollapsedVitalsPerSequence.json"),
        collapsed_labs_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "CollapsedLabsPerSequence.csv"),
        collapsed_labs_pertstep_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "Spec_CollapsedLabsPerSequence.json")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.vitals_csv} \
            --data_dict {input.vitals_spec_json} \
            --output "{output.collapsed_vitals_pertstep_csv}" \
            --data_dict_output "{output.collapsed_vitals_pertstep_json}" \
            --tstops {input.tstop_csv} \
            --collapse_range_features "std hours_since_measured present slope median min max" \
            --range_pairs "[('50%','100%'), ('75%','100%'), ('0%','100%'), ('T-6h','T'), ('T-12h','T'), ('T-24h','T')]" \
            --collapse \

        python -u {input.script} \
            --input {input.labs_csv} \
            --data_dict {input.labs_spec_json} \
            --output "{output.collapsed_labs_pertstep_csv}" \
            --data_dict_output "{output.collapsed_labs_pertstep_json}" \
            --tstops {input.tstop_csv}\
            --collapse_range_features "std hours_since_measured present slope median min max" \
            --range_pairs "[('50%','100%'), ('75%','100%'), ('0%','100%'), ('T-6h','T'), ('T-12h','T'), ('T-24h','T')]" \
            --collapse \
        '''

rule compute_mews_score:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'compute_mews_score.py'),
        x_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "vitals_before_icu_filtered_{tstep_hours}_hours.csv"),
        x_spec_json=os.path.join(DATASET_TOP_PATH, 'Spec-Vitals.json')
    
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
        x_spec_json=os.path.join(DATASET_TOP_PATH, 'Spec-Vitals.json')

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
        merge_script=os.path.join(os.path.abspath('../'), 'src', 'merge_train_test_splits.py'),
        collapsed_vitals_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "CollapsedVitalsPerSequence.csv"),
        collapsed_vitals_pertstep_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "Spec_CollapsedVitalsPerSequence.json"),
        collapsed_labs_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "CollapsedLabsPerSequence.csv"),
        collapsed_labs_pertstep_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "Spec_CollapsedLabsPerSequence.json"),
        demographics_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "demographics_before_icu_filtered_{tstep_hours}_hours.csv"),
        demographics_pertstep_json=os.path.join(DATASET_TOP_PATH, 'Spec-Demographics.json'),
        mews_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "MewsScoresPerSequence.csv"),
        mews_pertstep_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "Spec_MewsScoresPerSequence.json"),
        collapsedy_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", "clinical_deterioration_outcomes_filtered_{tstep_hours}_hours.csv"),
        collapsedy_json=os.path.join(DATASET_TOP_PATH, 'Spec-Outcomes_TransferToICU.json')
    
    params:
        train_test_split_dir=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}")
 
    output:
        demographics_dict=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'demographics_dict.json'),
        vitals_dict=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'vitals_dict.json'),
        labs_dict=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'labs_dict.json'),
        x_dict=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'x_dict.json'),
        y_dict=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'y_dict.json'),
        mews_dict=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'mews_dict.json'),
        demographics_train_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'demographics_train.csv'),
        demographics_test_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'demographics_test.csv'),
        vitals_train_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'vitals_train.csv'),
        vitals_test_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'vitals_test.csv'),
        labs_train_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'labs_train.csv'),
        labs_test_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tstep_hours}", 'labs_test.csv'),        
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
                --input {{input.collapsed_vitals_pertstep_csv}} \
                --data_dict {{input.collapsed_vitals_pertstep_json}} \
                --random_state {split_random_state} \
                --test_size {split_test_size} \
                --group_cols {split_key_col_names} \
                --train_csv_filename {{output.vitals_train_csv}} \
                --test_csv_filename {{output.vitals_test_csv}} \
                --output_data_dict_filename {{output.vitals_dict}} \

            python -u {{input.script}} \
                --input {{input.collapsed_labs_pertstep_csv}} \
                --data_dict {{input.collapsed_labs_pertstep_json}} \
                --random_state {split_random_state} \
                --test_size {split_test_size} \
                --group_cols {split_key_col_names} \
                --train_csv_filename {{output.labs_train_csv}} \
                --test_csv_filename {{output.labs_test_csv}} \
                --output_data_dict_filename {{output.labs_dict}} \

            python -u {{input.script}} \
                --input {{input.demographics_pertstep_csv}} \
                --data_dict {{input.demographics_pertstep_json}} \
                --random_state {split_random_state} \
                --test_size {split_test_size} \
                --group_cols {split_key_col_names} \
                --train_csv_filename {{output.demographics_train_csv}} \
                --test_csv_filename {{output.demographics_test_csv}} \
                --output_data_dict_filename {{output.demographics_dict}} \

            python -u {{input.merge_script}} \
                --train_test_split_dir {{params.train_test_split_dir}} \
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
