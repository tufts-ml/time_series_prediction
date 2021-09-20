'''
Produce a collapsed feature representation on Madrid Transfer to ICU Prediction

----------------------------------------------------------------------------------------------------------------------------------------
TRAIN COLLAPSED FEATURES AND OUTCOMES DYNAMICICALLY TO MIMIC REAL TIME DEPLOYMENT 
----------------------------------------------------------------------------------------------------------------------------------------

Usage : Collapsing features in multiple patient stay slices, with their corresponding outputs, dynamically
----------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_labs_vitals_only.smk make_collapsed_features_for_dynamic_output_prediction

----------------------------------------------------------------------------------------------------------------------------------------
COMPUTE MEWS SCORES DYNAMICALLY
----------------------------------------------------------------------------------------------------------------------------------------

Usage : Computing MEWS in multiple patient stay slices, with their corresponding outputs, dynamically
----------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_labs_vitals_only.smk compute_mews_for_dynamic_output_prediction

>> snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_labs_vitals_only.smk prepare_mews_dynamic_features

---------------------------------------------------------------------------------------------------------------------------------------
MERGE ALL VITALS, LABS AND MEDICATIONS INTO A SINGLE FEATURE MATRIX
---------------------------------------------------------------------------------------------------------------------------------------

Usage : Merge all the collapsed features across tslices into a single features table
----------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_labs_vitals_only.smk merge_dynamic_collapsed_features


Usage : Split the features table into train\valid\test based on the year of admission. Trian on first 3 years of admission. Vlaidate and test on the 4th and 5th years of admission
-------------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_labs_vitals_only.smk split_into_train_and_test

Usage : Do every step above in sequence
-------------------------------------
>> snakemake --cores all --snakefile make_collapsed_dataset_dynamic_labs_vitals_only.smk  all
'''

# Default environment variables
# Can override with local env variables

from config_loader import (
    D_CONFIG, DATASET_TOP_PATH,
    DATASET_SITE_PATH, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH
    )

CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 'classifier_train_test_split')

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(CLF_TRAIN_TEST_SPLIT_PATH)

# train-test split files
train_test_split_jsons=[os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "{file}_dict.json").replace("{file}",i) for i in ["x", "y", "mews"]]
train_test_split_csvs=[os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "{file}_{split}.csv.gz").format(file=i, split=j) for i in ["x", "y", "mews"] for j in ["train", "test"]]

rule make_collapsed_features_for_dynamic_output_prediction:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'dynamic_feature_transformation_deployment.py'),
        vitals_csv=os.path.join(DATASET_SITE_PATH, "vitals_before_icu.csv.gz"),
        vitals_spec_json=os.path.join(DATASET_SITE_PATH, 'Spec-Vitals.json'),
        labs_csv=os.path.join(DATASET_SITE_PATH, "labs_before_icu.csv.gz"),
        labs_spec_json=os.path.join(DATASET_SITE_PATH, 'Spec-Labs.json'), 
        medications_csv=os.path.join(DATASET_SITE_PATH, "medications_before_icu.csv.gz"),
        medications_spec_json=os.path.join(DATASET_SITE_PATH, 'Spec-Medications.json'), 
        outcomes_csv=os.path.join(DATASET_SITE_PATH, "clinical_deterioration_outcomes.csv.gz"),
        outcomes_spec_json=os.path.join(DATASET_SITE_PATH, "Spec-Outcomes_TransferToICU.json")

    output:
        collapsed_vitals_dynamic_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
        "CollapsedVitalsDynamic.csv.gz"),
        collapsed_vitals_dynamic_json=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "Spec_CollapsedVitalsDynamic.json"),
        collapsed_labs_dynamic_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
        "CollapsedLabsDynamic.csv.gz"),
        collapsed_labs_dynamic_json=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "Spec_CollapsedLabsDynamic.json"),
        collapsed_medications_dynamic_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
        "CollapsedMedicationsDynamic.csv.gz"),
        collapsed_medications_dynamic_json=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "Spec_CollapsedMedicationsDynamic.json"),
        outputs_dynamic_vitals_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "OutputsDynamicVitals.csv.gz"),
        outputs_dynamic_labs_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "OutputsDynamicLabs.csv.gz"),
        outputs_dynamic_medications_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "OutputsDynamicMedications.csv.gz"),

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.vitals_csv} \
            --ts_data_dict {input.vitals_spec_json} \
            --outcomes {input.outcomes_csv} \
            --outcomes_data_dict {input.outcomes_spec_json} \
            --dynamic_collapsed_features_csv "{output.collapsed_vitals_dynamic_csv}" \
            --dynamic_collapsed_features_data_dict "{output.collapsed_vitals_dynamic_json}" \
            --dynamic_outcomes_csv "{output.outputs_dynamic_vitals_csv}" \
            --features_to_summarize "std time_since_measured count slope median min max" \
            --percentile_ranges_to_summarize "[('0','100')]" \

        python -u {input.script} \
            --input {input.labs_csv} \
            --ts_data_dict {input.labs_spec_json} \
            --outcomes {input.outcomes_csv} \
            --outcomes_data_dict {input.outcomes_spec_json} \
            --dynamic_collapsed_features_csv "{output.collapsed_labs_dynamic_csv}" \
            --dynamic_collapsed_features_data_dict "{output.collapsed_labs_dynamic_json}" \
            --dynamic_outcomes_csv "{output.outputs_dynamic_labs_csv}" \
            --features_to_summarize "std time_since_measured count slope median min max" \
            --percentile_ranges_to_summarize "[(0, 100)]" \

        python -u {input.script} \
            --input {input.medications_csv} \
            --ts_data_dict {input.medications_spec_json} \
            --outcomes {input.outcomes_csv} \
            --outcomes_data_dict {input.outcomes_spec_json} \
            --dynamic_collapsed_features_csv "{output.collapsed_medications_dynamic_csv}" \
            --dynamic_collapsed_features_data_dict "{output.collapsed_medications_dynamic_json}" \
            --dynamic_outcomes_csv "{output.outputs_dynamic_medications_csv}" \
            --features_to_summarize "std time_since_measured count slope median min max" \
            --percentile_ranges_to_summarize "[(0, 100)]" \
        '''

rule compute_mews_for_dynamic_output_prediction:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'compute_mews_score_dynamic.py'),
        vitals_csv=os.path.join(DATASET_SITE_PATH, "vitals_before_icu.csv.gz"),
        vitals_spec_json=os.path.join(DATASET_SITE_PATH, 'Spec-Vitals.json'), 
        outcomes_csv=os.path.join(DATASET_SITE_PATH, "clinical_deterioration_outcomes.csv.gz"),
        outcomes_spec_json=os.path.join(DATASET_SITE_PATH, "Spec-Outcomes_TransferToICU.json")

    output:
        mews_dynamic_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
        "MewsDynamic.csv.gz"),
        mews_dynamic_json=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "Spec_MewsDynamic.json"),
        outputs_dynamic_mews_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "OutputsDynamicMews.csv.gz"),

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.vitals_csv} \
            --data_dict {input.vitals_spec_json} \
            --outcomes {input.outcomes_csv} \
            --data_dict_outcomes {input.outcomes_spec_json} \
            --dynamic_mews_csv "{output.mews_dynamic_csv}" \
            --dynamic_mews_data_dict "{output.mews_dynamic_json}" \
            --dynamic_outcomes_csv "{output.outputs_dynamic_mews_csv}" \
        '''

rule merge_dynamic_collapsed_features:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'merge_dynamic_collapsed_features.py')
    
    params:
        dynamic_collapsed_features_folder=DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        static_data_dict_dir=DATASET_SITE_PATH,
        output_dir=CLF_TRAIN_TEST_SPLIT_PATH
    
    output:
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "dynamic_features.csv.gz"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "dynamic_outcomes.csv.gz")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {{params.output_dir}} && \
        python -u {input.script} \
            --dynamic_collapsed_features_folder {params.dynamic_collapsed_features_folder} \
            --static_data_dict_dir {params.static_data_dict_dir} \
            --output_dir {params.output_dir}\
        '''

rule prepare_mews_dynamic_features:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'prepare_mews_dynamic_features.py')
    
    params:
        dynamic_collapsed_features_folder=DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        static_data_dict_dir=DATASET_SITE_PATH,
        output_dir=CLF_TRAIN_TEST_SPLIT_PATH
    
    output:
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "mews_dynamic_features.csv.gz"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "mews_dynamic_outcomes.csv.gz")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {{params.output_dir}} && \
        python -u {input.script} \
            --dynamic_collapsed_features_folder {params.dynamic_collapsed_features_folder} \
            --static_data_dict_dir {params.static_data_dict_dir} \
            --output_dir {params.output_dir}\
        '''

rule split_into_train_and_test:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'split_dataset_by_timestamp.py'),
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "dynamic_features.csv.gz"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "dynamic_outcomes.csv.gz"),
        mews_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 
        "mews_dynamic_features.csv.gz"),
        mews_outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 
        "mews_dynamic_outcomes.csv.gz"),
        features_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "Spec_features.json"),
        outcomes_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "Spec_outcomes.json"),
        mews_json=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, "Spec_MewsDynamic.json")

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH
 
    output:  
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv.gz'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_valid.csv.gz'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv.gz'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv.gz'),
        y_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_valid.csv.gz'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv.gz'),
        mews_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_train.csv.gz'),
        mews_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_valid.csv.gz'),
        mews_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_test.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json'),
        mews_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'mews_dict.json')

    conda:
        PROJECT_CONDA_ENV_YAML
    
    shell:
        '''
            mkdir -p {{params.train_test_split_dir}} && \
            python -u {{input.script}} \
                --input {{input.features_csv}} \
                --data_dict {{input.features_json}} \
                --test_size {split_test_size} \
                --train_csv_filename {{output.x_train_csv}} \
                --valid_csv_filename {{output.x_valid_csv}} \
                --test_csv_filename {{output.x_test_csv}} \
                --output_data_dict_filename {{output.x_dict_json}} \

            python -u {{input.script}} \
                --input {{input.outcomes_csv}} \
                --data_dict {{input.outcomes_json}} \
                --test_size {split_test_size} \
                --train_csv_filename {{output.y_train_csv}} \
                --valid_csv_filename {{output.y_valid_csv}} \
                --test_csv_filename {{output.y_test_csv}} \
                --output_data_dict_filename {{output.y_dict_json}} \

            python -u {{input.script}} \
                --input {{input.mews_csv}} \
                --data_dict {{input.mews_json}} \
                --test_size {split_test_size} \
                --train_csv_filename {{output.mews_train_csv}} \
                --valid_csv_filename {{output.mews_valid_csv}} \
                --test_csv_filename {{output.mews_test_csv}} \
                --output_data_dict_filename {{output.mews_dict_json}} \
        '''.format(
            split_test_size=D_CONFIG['SPLIT_TEST_SIZE'],
            )
