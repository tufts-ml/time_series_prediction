'''
Produce a collapsed feature representation on Madrid Transfer to ICU Prediction

----------------------------------------------------------------------------------------------------------------------------------------
COLLAPSING FEATURES IN EACH TSLICE
----------------------------------------------------------------------------------------------------------------------------------------

Usage : Filter admissions by patient-stay-slice (For eg, first 4 hrs, last 4hrs, first 30%)
---------------------------------------------------------------------------
Note : For eg. If we predict using first 4 hours of data, we ensure that every patient in the cohort has atleast 4 hours of data.

>> snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk filter_admissions_by_tslice_many_tslices

Usage : Collapsing features and saving to slice specific folders
----------------------------------------------------------------
>> snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk collapse_features_many_tslices

Usage : Computing slice specific MEWS scores
--------------------------------------------
>> snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk compute_mews_score_many_tslices

---------------------------------------------------------------------------------------------------------------------------------------
MERGE ALL TSLICES AND SPLITTING INTO TRAIN AND TEST
---------------------------------------------------------------------------------------------------------------------------------------

Usage : Merge all the collapsed features across tslices into a single features table
----------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk merge_collapsed_features_all_tslices

Usage : Split the features table into train - test. A single classifier will be trained on this training fold
-------------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk split_into_train_and_test

Usage : Do every step above in squence
-------------------------------------
>> snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk all
'''

# Default environment variables
# Can override with local env variables

from config_loader import (
    D_CONFIG, DATASET_TOP_PATH,
    DATASET_SITE_PATH, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    DATASET_FEAT_PER_TSLICE_PATH,
    DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH
    )
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(CLF_TRAIN_TEST_SPLIT_PATH)

# run collapse features on all selected slices
train_tslice_hours_list=D_CONFIG['TRAIN_TIMESLICE_LIST']
evaluate_tslice_hours_list=D_CONFIG['EVALUATE_TIMESLICE_LIST']

# define the train test split dir
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')


# filtered files
filtered_pertslice_csvs=[os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}","{feature}_before_icu_filtered_{tslice}_hours.csv.gz").format(feature=feature, tslice=str(tslice)) for tslice in evaluate_tslice_hours_list for feature in ['vitals', 'labs', 'medications']]

# collapsed_files
collapsed_pertslice_csvs=[os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}","Collapsed{feature}PerSequence.csv.gz").format(feature=feature, tslice=str(tslice)) for tslice in evaluate_tslice_hours_list for feature in ['Vitals', 'Labs', 'Medications']]
collapsed_pertslice_jsons=[os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}","Spec_Collapsed{feature}PerSequence.json").format(feature=feature, tslice=str(tslice)) for tslice in evaluate_tslice_hours_list for feature in ['Vitals', 'Labs', 'Medications']]

# mews files
mews_pertslice_csvs=[os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}","MewsScoresPerSequence.csv.gz").replace("{tslice}", str(tslice)) for tslice in evaluate_tslice_hours_list]
mews_pertslice_jsons=[os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}","Spec_MewsScoresPerSequence.json").replace("{tslice}", str(tslice)) for tslice in evaluate_tslice_hours_list]

# train-test split files
train_test_split_jsons=[os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "{file}_dict.json").replace("{file}",i) for i in ["x", "y", "mews"]]
train_test_split_csvs=[os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "{file}_{split}.csv.gz").format(file=i, split=j) for i in ["x", "y", "mews"] for j in ["train", "test"]]

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
        collapsed_pertslice_jsons

rule filter_admissions_by_tslice:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'filter_admissions_by_tslice.py'),
    
    params:
        preproc_data_dir = DATASET_SITE_PATH,
        output_dir=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}")
    
    output:
        filtered_demographics_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "demographics_before_icu_filtered_{tslice}_hours.csv.gz"),
        filtered_vitals_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "vitals_before_icu_filtered_{tslice}_hours.csv.gz"),
        filtered_labs_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "labs_before_icu_filtered_{tslice}_hours.csv.gz"),
        filtered_medications_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "medications_before_icu_filtered_{tslice}_hours.csv.gz"),
        filtered_y_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "clinical_deterioration_outcomes_filtered_{tslice}_hours.csv.gz"),
        tstops_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "tstops_filtered_{tslice}_hours.csv.gz")
    
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
        vitals_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "vitals_before_icu_filtered_{tslice}_hours.csv.gz"),
        vitals_spec_json=os.path.join(DATASET_SITE_PATH, 'Spec-Vitals.json'),
        labs_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "labs_before_icu_filtered_{tslice}_hours.csv.gz"),
        labs_spec_json=os.path.join(DATASET_SITE_PATH, 'Spec-Labs.json'),
        medications_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "medications_before_icu_filtered_{tslice}_hours.csv.gz"),
        medications_spec_json=os.path.join(DATASET_SITE_PATH, 'Spec-Medications.json'),        
        tstop_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "tstops_filtered_{tslice}_hours.csv.gz")

    output:
        collapsed_vitals_pertslice_csv=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "CollapsedVitalsPerSequence.csv.gz"),
        collapsed_vitals_pertslice_json=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "Spec_CollapsedVitalsPerSequence.json"),
        collapsed_labs_pertslice_csv=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "CollapsedLabsPerSequence.csv.gz"),
        collapsed_labs_pertslice_json=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "Spec_CollapsedLabsPerSequence.json"),
        collapsed_medications_pertslice_csv=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "CollapsedMedicationsPerSequence.csv.gz"),
        collapsed_medications_pertslice_json=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "Spec_CollapsedMedicationsPerSequence.json")

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
            --range_pairs "[('50%','100%'), ('0%','100%')]" \
            --collapse \

        python -u {input.script} \
            --input {input.labs_csv} \
            --data_dict {input.labs_spec_json} \
            --output "{output.collapsed_labs_pertslice_csv}" \
            --data_dict_output "{output.collapsed_labs_pertslice_json}" \
            --tstops {input.tstop_csv}\
            --collapse_range_features "std hours_since_measured present median min max" \
            --range_pairs "[('0%','100%')]" \
            --collapse \

        python -u {input.script} \
            --input {input.medications_csv} \
            --data_dict {input.medications_spec_json} \
            --output "{output.collapsed_medications_pertslice_csv}" \
            --data_dict_output "{output.collapsed_medications_pertslice_json}" \
            --tstops {input.tstop_csv}\
            --collapse_range_features "std median min max" \
            --range_pairs "[('0%','100%')]" \
            --collapse \
        '''

rule compute_mews_score:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'compute_mews_score.py'),
        x_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "vitals_before_icu_filtered_{tslice}_hours.csv.gz"),
        x_spec_json=os.path.join(DATASET_SITE_PATH, 'Spec-Vitals.json')
    
    params:
        output_dir=DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH

    output:
        mews_pertstep_csv=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "MewsScoresPerSequence.csv.gz"),
        mews_pertstep_json=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "Spec_MewsScoresPerSequence.json")

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

rule merge_collapsed_features_all_tslices:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'merge_features_all_tslices.py')
    
    params:
        collapsed_tslice_folder=os.path.join(DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE="),
        tslice_folder=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE="),
        tslice_list=train_tslice_hours_list,
        static_data_dict_dir=DATASET_SITE_PATH,
        output_dir=CLF_TRAIN_TEST_SPLIT_PATH
    
    output:
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features.csv.gz"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes.csv.gz")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {{params.output_dir}} && \
        python -u {input.script} \
            --collapsed_tslice_folder {params.collapsed_tslice_folder} \
            --tslice_folder {params.tslice_folder} \
            --tslice_list "{params.tslice_list}" \
            --static_data_dict_dir {params.static_data_dict_dir} \
            --output_dir {params.output_dir}\
        '''

rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features.csv.gz"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes.csv.gz"),
        mews_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "mews.csv.gz"),
        features_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "Spec_features.json"),
        outcomes_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "Spec_outcomes.json"),
        mews_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "Spec_mews.json")

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH
 
    output:  
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv.gz'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv.gz'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv.gz'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json'),
        mews_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_train.csv.gz'),
        mews_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_test.csv.gz'),
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
