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
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    DATASET_FEATURES_OUTCOMES_PATH,
    CLF_TRAIN_TEST_SPLIT_PATH)

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(CLF_TRAIN_TEST_SPLIT_PATH)

# run collapse features on all selected slices
train_tslice_hours_list=D_CONFIG['TRAIN_TIMESLICE_LIST']
evaluate_tslice_hours_list=D_CONFIG['EVALUATE_TIMESLICE_LIST']

# collapse files
filtered_pertslice_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}","{feature}_before_icu_filtered_{tslice}_hours.csv").format(feature=feature, tslice=str(tslice)) for tslice in evaluate_tslice_hours_list for feature in ['vitals', 'labs']]

collapsed_pertslice_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}","Collapsed{feature}PerSequence.csv").format(feature=feature, tslice=str(tslice)) for tslice in evaluate_tslice_hours_list for feature in ['Vitals', 'Labs']]
collapsed_pertslice_jsons=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}","Spec_Collapsed{feature}PerSequence.json").format(feature=feature, tslice=str(tslice)) for tslice in evaluate_tslice_hours_list for feature in ['Vitals', 'Labs']]

# mews files
mews_pertslice_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}","MewsScoresPerSequence.csv").replace("{tslice}", str(tslice)) for tslice in evaluate_tslice_hours_list]
mews_pertslice_jsons=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}","Spec_MewsScoresPerSequence.json").replace("{tslice}", str(tslice)) for tslice in evaluate_tslice_hours_list]

#train-test files
train_test_split_jsons=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}","{data_dict}").format(tslice=str(tslice), data_dict=data_dict) for tslice in train_tslice_hours_list for data_dict in ['x_dict.json', 'y_dict.json', 'mews_dict.json']]

train_test_split_csvs=[os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}","{data_csv}").format(tslice=str(tslice), data_csv=data_csv) for tslice in train_tslice_hours_list for data_csv in ['x_train.csv', 'x_test.csv', 'y_train.csv', 'y_test.csv', 'mews_train.csv', 'mews_test.csv']]

rule filter_admissions_by_tslice_many_tslices:
    input:
        filtered_pertslice_csvs

rule collapse_features_many_tslices:
    input:
        collapsed_pertslice_csvs,
        collapsed_pertslice_jsons

rule compute_mews_score_many_tslices:
    input:
        mews_pertslice_csvs,
        mews_pertslice_jsons

rule all:
    input:
        filtered_pertslice_csvs,
        collapsed_pertslice_csvs,
        collapsed_pertslice_jsons,
        mews_pertslice_csvs,
        mews_pertslice_jsons,
        train_test_split_jsons,
        train_test_split_csvs,

rule filter_admissions_by_tslice:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'filter_admissions_by_tslice.py'),
    
    params:
        preproc_data_dir = DATASET_STD_PATH,
        output_dir=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}")
    
    output:
        filtered_demographics_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "demographics_before_icu_filtered_{tslice}_hours.csv"),
        filtered_vitals_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "vitals_before_icu_filtered_{tslice}_hours.csv"),
        filtered_labs_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "labs_before_icu_filtered_{tslice}_hours.csv"),
        filtered_y_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "clinical_deterioration_outcomes_filtered_{tslice}_hours.csv")
    
    shell:
        '''
            python -u {input.script} \
                --preproc_data_dir {params.preproc_data_dir} \
                --tslice "{wildcards.tslice}" \
                --output_dir {params.output_dir} \
        '''

rule collapse_features:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'feature_transformation.py'),
        vitals_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "vitals_before_icu_filtered_{tslice}_hours.csv"),
        vitals_spec_json=os.path.join(DATASET_STD_PATH, 'Spec-Vitals.json'),
        labs_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "labs_before_icu_filtered_{tslice}_hours.csv"),
        labs_spec_json=os.path.join(DATASET_STD_PATH, 'Spec-Labs.json'),
        tstop_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "tstops_filtered_{tslice}_hours.csv")

    output:
        collapsed_vitals_pertslice_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "CollapsedVitalsPerSequence.csv"),
        collapsed_vitals_pertslice_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "Spec_CollapsedVitalsPerSequence.json"),
        collapsed_labs_pertslice_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "CollapsedLabsPerSequence.csv"),
        collapsed_labs_pertslice_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "Spec_CollapsedLabsPerSequence.json")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.vitals_csv} \
            --data_dict {input.vitals_spec_json} \
            --output "{output.collapsed_vitals_pertslice_csv}" \
            --data_dict_output "{output.collapsed_vitals_pertslice_json}" \
            --tstops {input.tstop_csv} \
            --collapse_range_features "std hours_since_measured present slope median min max" \
            --range_pairs "[('50%','100%'), ('0%','100%'), ('T-16h','T-0h'), ('T-24h','T-0h')]" \
            --collapse \

        python -u {input.script} \
            --input {input.labs_csv} \
            --data_dict {input.labs_spec_json} \
            --output "{output.collapsed_labs_pertslice_csv}" \
            --data_dict_output "{output.collapsed_labs_pertslice_json}" \
            --tstops {input.tstop_csv}\
            --collapse_range_features "std hours_since_measured present slope median min max" \
            --range_pairs "[('50%','100%'), ('0%','100%'), ('T-16h','T-0h'), ('T-24h','T-0h')]" \
            --collapse \
        '''

rule compute_mews_score:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'compute_mews_score.py'),
        x_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "vitals_before_icu_filtered_{tslice}_hours.csv"),
        x_spec_json=os.path.join(DATASET_STD_PATH, 'Spec-Vitals.json')
    
    params:
        output_dir=DATASET_PERTSTEP_SPLIT_PATH

    output:
        mews_pertstep_csv=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "MewsScoresPerSequence.csv"),
        mews_pertstep_json=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP={tslice}", "Spec_MewsScoresPerSequence.json")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.x_csv} \
            --data_dict {input.x_spec_json} \
            --output  "{output.mews_pertstep_csv}" \
            --data_dict_output "{output.mews_pertstep_json}" \
        '''

rule merge_features_all_tslices:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'merge_features_all_tslices.py')
    
    params:
        tslice_folder=os.path.join(DATASET_PERTSTEP_SPLIT_PATH, "TSTEP="),
        tslice_list=train_tslice_hours_list,
        static_data_dict_dir=DATASET_TOP_PATH,
        output_dir=DATASET_FEATURES_OUTCOMES_PATH
    
    output:
        features_csv=os.path.join(DATASET_FEATURES_OUTCOMES_PATH, "features.csv"),
        outcomes_csv=os.path.join(DATASET_FEATURES_OUTCOMES_PATH, "outcomes.csv")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {{params.output_dir}} && \
        python -u {input.script} \
            --tslice_folder {params.tslice_folder} \
            --tslice_list "{params.tslice_list}" \
            --static_data_dict_dir {params.static_data_dict_dir} \
            --output_dir {params.output_dir}\
        '''

rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        features_csv=os.path.join(DATASET_FEATURES_OUTCOMES_PATH, "features.csv"),
        outcomes_csv=os.path.join(DATASET_FEATURES_OUTCOMES_PATH, "outcomes.csv"),
        mews_csv=os.path.join(DATASET_FEATURES_OUTCOMES_PATH, "mews.csv"),
        features_json=os.path.join(DATASET_FEATURES_OUTCOMES_PATH, "Spec_features.json"),
        outcomes_json=os.path.join(DATASET_FEATURES_OUTCOMES_PATH, "Spec_outcomes.json"),
        mews_json=os.path.join(DATASET_FEATURES_OUTCOMES_PATH, "Spec_mews.json")

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH
 
    output:  
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json'),
        mews_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_train.csv'),
        mews_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_test.csv'),
        mews_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_dict.json')

    conda:
        PROJECT_CONDA_ENV_YAML
    
    shell:
        '''
            mkdir -p {{params.train_test_split_dir}} && \
            python -u {{input.script}} \
                --input {{input.features_csv}} \
                --data_dict {{input.features_json}} \
                --random_state {split_random_state} \
                --test_size {split_test_size} \
                --group_cols {split_key_col_names} \
                --train_csv_filename {{output.x_train_csv}} \
                --test_csv_filename {{output.x_test_csv}} \
                --output_data_dict_filename {{output.x_dict_json}} \

            python -u {{input.script}} \
                --input {{input.outcomes_csv}} \
                --data_dict {{input.outcomes_json}} \
                --random_state {split_random_state} \
                --test_size {split_test_size} \
                --group_cols {split_key_col_names} \
                --train_csv_filename {{output.y_train_csv}} \
                --test_csv_filename {{output.y_test_csv}} \
                --output_data_dict_filename {{output.y_dict_json}} \

            python -u {{input.script}} \
                --input {{input.mews_csv}} \
                --data_dict {{input.mews_json}} \
                --random_state {split_random_state} \
                --test_size {split_test_size} \
                --group_cols {split_key_col_names} \
                --train_csv_filename {{output.mews_train_csv}} \
                --test_csv_filename {{output.mews_test_csv}} \
                --output_data_dict_filename {{output.mews_dict_json}} \
        '''.format(
            split_random_state=D_CONFIG['SPLIT_RANDOM_STATE'],
            split_test_size=D_CONFIG['SPLIT_TEST_SIZE'],
            split_key_col_names=D_CONFIG['SPLIT_KEY_COL_NAMES'],
            )
