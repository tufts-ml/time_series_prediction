'''
Produce a collapsed feature representation on Madrid Transfer to ICU Prediction


'''

# Default environment variables
# Can override with local env variables

#MADRID_VERSION = os.environ.get('MADRID_VERSION', 'v20200424')
#PROJECT_REPO_DIR = os.environ.get("PROJECT_REPO_DIR", os.path.abspath("../../../"))
#PROJECT_CONDA_ENV_YAML = os.path.join(PROJECT_REPO_DIR, "ts_pred.yml")

#MADRID_DATASET_TOP_PATH = os.path.expandvars(os.path.join("$HOME", "datasets/"))
#SITE_NAME = "HUF/"
#MADRID_DATASET_STD_PATH = os.path.join(MADRID_DATASET_TOP_PATH, MADRID_VERSION, SITE_NAME)
from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML)

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(DATASET_SPLIT_PATH)

rule all:
    input:
        os.path.join(DATASET_STD_PATH, 'CollapsedFeaturesPerSequence.csv'),
        os.path.join(DATASET_STD_PATH, 'Spec_CollapsedFeaturesPerSequence.json'),	
        os.path.join(DATASET_SPLIT_PATH, 'x_train.csv'),
        os.path.join(DATASET_SPLIT_PATH, 'x_test.csv'),
        os.path.join(DATASET_SPLIT_PATH, 'y_train.csv'),
        os.path.join(DATASET_SPLIT_PATH, 'y_test.csv')

rule collapse_features:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'feature_transformation.py'),
        x_csv=os.path.join(DATASET_STD_PATH, 'vitals_before_icu.csv'),
        x_spec_json=os.path.join(DATASET_STD_PATH, 'Spec-Vitals.json')

    output:
        collapsedx_csv=os.path.join(DATASET_STD_PATH, 'CollapsedFeaturesPerSequence.csv'),
        collapsedx_json=os.path.join(DATASET_STD_PATH, 'Spec_CollapsedFeaturesPerSequence.json')

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.x_csv} \
            --data_dict {input.x_spec_json} \
            --output {output.collapsedx_csv} \
            --data_dict_output {output.collapsedx_json} \
            --collapse_range_features "hours_since_measured present slope std median min max" \
            --range_pairs "[(0,10), (0,25), (0,50), (50,100), (75,100), (90,100), (0,100)]" \
            --collapse
        '''


rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        collapsedx_csv=os.path.join(DATASET_STD_PATH, 'CollapsedFeaturesPerSequence.csv'),
        collapsedx_json=os.path.join(DATASET_STD_PATH, 'Spec_CollapsedFeaturesPerSequence.json'),
        collapsedy_csv=os.path.join(DATASET_STD_PATH, 'transfer_to_icu_outcomes.csv'),
        collapsedy_json=os.path.join(DATASET_STD_PATH, 'Spec-Outcomes_TransferToICU.json')

    output:
        x_dict=os.path.join(DATASET_SPLIT_PATH, 'x_dict.json'),
        y_dict=os.path.join(DATASET_SPLIT_PATH, 'y_dict.json'),
        x_train_csv=os.path.join(DATASET_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_PATH, 'y_test.csv')

    conda:
        PROJECT_CONDA_ENV_YAML
    
    shell:
        '''
            python -u {{input.script}} \
                --input {{input.collapsedx_csv}} \
                --data_dict {{input.collapsedx_json}} \
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
        '''.format(
            split_random_state=D_CONFIG['SPLIT_RANDOM_STATE'],
            split_test_size=D_CONFIG['SPLIT_TEST_SIZE'],
            split_key_col_names=D_CONFIG['SPLIT_KEY_COL_NAMES'],
            )
