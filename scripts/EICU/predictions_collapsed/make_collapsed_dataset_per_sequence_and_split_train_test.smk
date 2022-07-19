'''
Produce a collapsed feature representation and split train test for EICU inhospital mortality
and produce train/test CSV files

Usage
-----
snakemake --cores 1 all


# collapse sequence features
snakemake --cores 1 --snakefile make_collapsed_dataset_per_sequence_and_split_train_test.smk collapse_features


# split into train test
snakemake --cores 1 --snakefile make_collapsed_dataset_per_sequence_and_split_train_test.smk split_into_train_and_test

'''

sys.path.append('../predictions_collapsed/')
from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH,
    PROJECT_REPO_DIR, 
    PROJECT_CONDA_ENV_YAML,
    DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH)

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH, 'classifier_train_test_split_dir')

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(CLF_TRAIN_TEST_SPLIT_PATH)



rule collapse_features:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'feature_transformation.py'),
        features_csv=os.path.join(DATASET_STD_PATH, 'features_per_tstep.csv'),
        features_spec_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerTimestep.json')

    output:
        collapsed_features_pertslice_csv=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH,
        "CollapsedFeaturesPerSequence.csv"),
        collapsed_features_pertslice_json=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH,
        "Spec_CollapsedFeaturesPerSequence.json"),

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.features_csv} \
            --data_dict {input.features_spec_json} \
            --output "{output.collapsed_features_pertslice_csv}" \
            --data_dict_output "{output.collapsed_features_pertslice_json}" \
            --collapse_range_features "std hours_since_measured present slope median min max" \
            --range_pairs "[('50%','100%'), ('0%','100%'), ('T-16h','T-0h'), ('T-24h','T-0h')]" \
            --collapse \
        '''

rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        features_csv=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH, "CollapsedFeaturesPerSequence.csv"),
        outcomes_csv=os.path.join(DATASET_STD_PATH, "outcomes_per_seq.csv"),
        features_json=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH, "Spec_CollapsedFeaturesPerSequence.json"),
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