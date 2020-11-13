'''
Produce a collapsed feature representation for this toy dataset
and produce train/test CSV files

Usage
-----
snakemake --cores 1 all
'''

sys.path.append('../predictions_collapsed/')
from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML)

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(DATASET_SPLIT_PATH)

# Default environment variables
# Can override with local env variables
rule all:
    input:
        x_train_csv=os.path.join(DATASET_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_PATH, 'y_test.csv')


rule merge_static_highfreq_features:
    input:
        script=os.path.abspath('../src/merge_static_highfreq_features.py'),
        static_features_csv=os.path.join(DATASET_STD_PATH, 'features_per_subject.csv'),
        static_features_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerSubject.json'),
        highfreq_features_csv=os.path.join(DATASET_STD_PATH, 'features_per_timestep.csv'),
        highfreq_features_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerTimestep.json')
    
    params:
        output_dir=DATASET_STD_PATH
    
    output:
        features_csv=os.path.join(DATASET_STD_PATH, 'features_per_sequence.csv'),
        features_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerSequence.json'),
        
    shell:
        '''
            python -u {input.script} \
                --static_features_csv {input.static_features_csv} \
                --static_features_json {input.static_features_json} \
                --highfreq_features_csv {input.highfreq_features_csv} \
                --highfreq_features_json {input.highfreq_features_json} \
                --output_dir {params.output_dir}
        '''
        

rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        collapsedx_csv=os.path.join(DATASET_STD_PATH, 'features_per_sequence.csv'),
        collapsedx_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerSequence.json'),
        collapsedy_csv=os.path.join(DATASET_STD_PATH, 'outcomes_per_sequence.csv'),
        collapsedy_json=os.path.join(DATASET_STD_PATH, 'Spec_OutcomesPerSequence.json')

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
