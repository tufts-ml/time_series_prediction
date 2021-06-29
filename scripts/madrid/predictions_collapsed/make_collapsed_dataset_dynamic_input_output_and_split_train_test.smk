'''
Produce a collapsed feature representation on Madrid Transfer to ICU Prediction

----------------------------------------------------------------------------------------------------------------------------------------
TRAIN COLLAPSED FEATURES AND OUTCOMES DYNAMICICALLY TO MIMIC REAL TIME DEPLOYMENT 
----------------------------------------------------------------------------------------------------------------------------------------

Usage : Collapsing features and saving to slice specific folders
----------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_input_output_and_split_train_test.smk make_collapsed_features_for_dynamic_output_prediction

---------------------------------------------------------------------------------------------------------------------------------------
MERGE ALL VITALS, LABS AND MEDICATIONS INTO A SINGLE FEATURE MATRIX
---------------------------------------------------------------------------------------------------------------------------------------

Usage : Merge all the collapsed features across tslices into a single features table
----------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_input_output_and_split_train_test.smk merge_collapsed_features_all_tslices


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
        script=os.path.join(os.path.abspath('../'), 'src', 'dynamic_feature_transformation.py'),
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
        outputs_dynamic_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "OutputsDynamic.csv.gz"),
        outputs_dynamic_json=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "Spec_OutputsDynamic.json")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.vitals_csv} \
            --data_dict {input.vitals_spec_json} \
            --outcomes {input.outcomes_csv} \
            --data_dict_outcomes {input.outcomes_spec_json} \
            --dynamic_collapsed_features_csv "{output.collapsed_vitals_dynamic_csv}" \
            --dynamic_collapsed_features_data_dict "{output.collapsed_vitals_dynamic_json}" \
            --dynamic_outcomes_csv "{output.outputs_dynamic_csv}" \
            --dynamic_outcomes_data_dict "{output.outputs_dynamic_json}" \
            --collapse_range_features "std hours_since_measured present slope median min max" \
            --range_pairs "[('0%','100%')]" \

        python -u {input.script} \
            --input {input.labs_csv} \
            --data_dict {input.labs_spec_json} \
            --outcomes {input.outcomes_csv} \
            --data_dict_outcomes {input.outcomes_spec_json} \
            --dynamic_collapsed_features_csv "{output.collapsed_labs_dynamic_csv}" \
            --dynamic_collapsed_features_data_dict "{output.collapsed_labs_dynamic_json}" \
            --dynamic_outcomes_csv "{output.outputs_dynamic_csv}" \
            --dynamic_outcomes_data_dict "{output.outputs_dynamic_json}" \
            --collapse_range_features "std hours_since_measured present median min max" \
            --range_pairs "[('0%','100%')]" \

        python -u {input.script} \
            --input {input.medications_csv} \
            --data_dict {input.medications_spec_json} \
            --outcomes {input.outcomes_csv} \
            --data_dict_outcomes {input.outcomes_spec_json} \
            --dynamic_collapsed_features_csv "{output.collapsed_medications_dynamic_csv}" \
            --dynamic_collapsed_features_data_dict "{output.collapsed_medications_dynamic_json}" \
            --dynamic_outcomes_csv "{output.outputs_dynamic_csv}" \
            --dynamic_outcomes_data_dict "{output.outputs_dynamic_json}" \
            --collapse_range_features "std median min max" \
            --range_pairs "[('0%','100%')]" \
        '''

rule compute_mews_score:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'compute_mews_score.py'),
        x_csv=os.path.join(DATASET_SITE_PATH, "vitals_before_icu.csv.gz"),
        x_spec_json=os.path.join(DATASET_SITE_PATH, 'Spec-Vitals.json')
    
    params:
        output_dir=DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH

    output:
        mews_dynamic_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, "MewsScoresDynamic.csv.gz"),
        mews_dynamic_json=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, "Spec_MewsScoresDynamic.json")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.x_csv} \
            --data_dict {input.x_spec_json} \
            --output  "{output.mews_dynamic_csv}" \
            --data_dict_output "{output.mews_dynamic_json}" \
        '''

rule merge_collapsed_features_all_tslices:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'merge_features_all_tslices.py')
    
    params:
        collapsed_tslice_folder=DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
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
        features_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "Spec_features.json"),
        outcomes_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "Spec_outcomes.json"),

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH
 
    output:  
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv.gz'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv.gz'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv.gz'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json')

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
        '''.format(
            split_random_state=D_CONFIG['SPLIT_RANDOM_STATE'],
            split_test_size=D_CONFIG['SPLIT_TEST_SIZE'],
            split_key_col_names=D_CONFIG['SPLIT_KEY_COL_NAMES'],
            )
