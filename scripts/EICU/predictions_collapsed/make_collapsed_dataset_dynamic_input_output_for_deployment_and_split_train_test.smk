'''
Produce a collapsed feature representation and split train test for eicu inhospital mortality
and produce train/test CSV files

Usage
-----
snakemake --cores 1 all

# collapse sequence features
snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_input_output_for_deployment_and_split_train_test.smk make_collapsed_features_for_dynamic_output_prediction

snakemake --snakefile make_collapsed_dataset_dynamic_input_output_for_deployment_and_split_train_test.smk --profile ../../../profiles/hugheslab_cluster/ make_collapsed_features_for_dynamic_output_prediction


# split into train test
snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_input_output_for_deployment_and_split_train_test.smk split_into_train_and_test

'''

sys.path.append('../predictions_collapsed/')
from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH,
    PROJECT_REPO_DIR, 
    PROJECT_CONDA_ENV_YAML,
    DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH)

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 'classifier_train_test_split_dir')

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(CLF_TRAIN_TEST_SPLIT_PATH)


rule make_collapsed_features_for_dynamic_output_prediction:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'dynamic_feature_transformation_deployment.py'),
        features_csv=os.path.join(DATASET_STD_PATH, 'features_per_tstep.csv'),
        features_spec_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerTimestep.json'), 
        outcomes_csv=os.path.join(DATASET_STD_PATH, "outcomes_per_seq.csv"),
        outcomes_spec_json=os.path.join(DATASET_STD_PATH, "Spec_OutcomesPerSequence.json")

    output:
        collapsed_features_dynamic_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
        "CollapsedFeaturesDynamic.csv.gz"),
        collapsed_features_dynamic_json=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "Spec_CollapsedFeaturesDynamic.json"),
        outputs_dynamic_features_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        "OutputsDynamicFeatures.csv.gz")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.features_csv} \
            --ts_data_dict {input.features_spec_json} \
            --outcomes {input.outcomes_csv} \
            --outcomes_data_dict {input.outcomes_spec_json} \
            --dynamic_collapsed_features_csv "{output.collapsed_features_dynamic_csv}" \
            --dynamic_collapsed_features_data_dict "{output.collapsed_features_dynamic_json}" \
            --dynamic_outcomes_csv "{output.outputs_dynamic_features_csv}" \
            --features_to_summarize "std time_since_measured count slope median min max last_value_measured" \
            --percentile_ranges_to_summarize "[('90', '100'), ('0', '100')]" \
        '''

rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        features_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, "CollapsedFeaturesDynamic.csv.gz"),
        outcomes_csv=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, "OutputsDynamicFeatures.csv.gz"),
        features_json=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, "Spec_CollapsedFeaturesDynamic.json"),
        outcomes_json=os.path.join(DATASET_STD_PATH, "Spec_OutcomesPerSequence.json"),

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH
 
    output:  
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json')
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
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