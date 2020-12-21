'''
Produce a collapsed feature representation for human activities
and produce train/test CSV files

Usage
-----
snakemake --cores 1 all
'''

sys.path.append('../predictions_collapsed/')
from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, DATASET_SPLIT_FEAT_PER_TSTEP_PATH)

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(DATASET_SPLIT_FEAT_PER_TSTEP_PATH)

# Default environment variables
# Can override with local env variables
rule all:
    input:
        x_train_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'y_test.csv')

rule impute_missing_values:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'impute_missing_values.py'),
        features_csv=os.path.join(DATASET_STD_PATH, 'features_per_tstep.csv'),
        features_dict_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerTimestep.json')
    
    params:
        features_dir=DATASET_STD_PATH
    
    shell:
        '''
            python -u {input.script} \
                --features_csv {input.features_csv} \
                --features_data_dict {input.features_dict_json} \
        '''     
        
rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        x_csv=os.path.join(DATASET_STD_PATH, 'features_per_tstep.csv'),
        x_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerTimestep.json'),
        y_csv=os.path.join(DATASET_STD_PATH, 'outcomes_per_seq.csv'),
        y_json=os.path.join(DATASET_STD_PATH, 'Spec_OutcomesPerSequence.json')

    output:
        x_dict=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'x_dict.json'),
        y_dict=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'y_dict.json'),
        x_train_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'y_test.csv')

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
            python -u {{input.script}} \
                --input {{input.x_csv}} \
                --data_dict {{input.x_json}} \
                --random_state {split_random_state} \
                --test_size {split_test_size} \
                --group_cols {split_key_col_names} \
                --train_csv_filename {{output.x_train_csv}} \
                --test_csv_filename {{output.x_test_csv}} \
                --output_data_dict_filename {{output.x_dict}} \
                --normalize True \

            python -u {{input.script}} \
                --input {{input.y_csv}} \
                --data_dict {{input.y_json}} \
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

