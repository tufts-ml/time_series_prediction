'''
Produce a collapsed feature representation and split train test for mimic3 inhospital mortality
and produce train/test CSV files

Usage
-----
snakemake --cores 1 all

# filter admissions by tslice
snakemake --cores 1 --snakefile make_collapsed_dataset_per_sequence_and_split_train_test.smk filter_admissions_by_tslice_many_tslices

# collapse sequence features in first 24 hours
snakemake --cores 1 --snakefile make_collapsed_dataset_per_sequence_and_split_train_test.smk collapse_features_first_24_hours


# split into train test first 24 hours
snakemake --cores 1 --snakefile make_collapsed_dataset_per_sequence_and_split_train_test.smk split_first_24_hours_data_into_train_and_test

'''

sys.path.append('../predictions_collapsed/')
from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, 
    DATASET_SPLIT_FEAT_PER_TSLICE_PATH,
    DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH,
    RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH)

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH, 'classifier_train_test_split_dir')
DATASET_RAW_PATH = "/cluster/tufts/hugheslab/datasets/mimic-iii-v1.4/v20201130/extract/"


print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(CLF_TRAIN_TEST_SPLIT_PATH)

# evaluate on the filtered tslices
evaluate_tslice_hours_list=D_CONFIG['EVALUATE_TIMESLICE_LIST']

# filtered sequences
filtered_pertslice_csvs=[os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}","features_before_death_filtered_{tslice}_hours.csv").format(tslice=str(tslice)) for tslice in evaluate_tslice_hours_list]


rule filter_admissions_by_tslice_many_tslices:
    input:
        filtered_pertslice_csvs

rule filter_admissions_by_tslice:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'filter_admissions_by_tslice.py'),
    
    params:
        preproc_data_dir = DATASET_RAW_PATH,
        output_dir=os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}")
    
    output:
        filtered_features_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "features_before_death_filtered_{tslice}_hours.csv"),
        filtered_y_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "outcomes_filtered_{tslice}_hours.csv")
    
    shell:
        '''
            python -u {input.script} \
                --preproc_data_dir {params.preproc_data_dir} \
                --tslice "{wildcards.tslice}" \
                --output_dir {params.output_dir} \
        '''
        
rule collapse_features_first_24_hours:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'feature_transformation.py'),
        features_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, "TSLICE=24", "features_before_death_filtered_24_hours.csv"),
        features_spec_json=os.path.join(DATASET_RAW_PATH, 'Spec_FeaturesPerTimestep.json')

    output:
        collapsed_features_pertslice_csv=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH,
        "CollapsedFeaturesPerSequenceFirst24Hours.csv"),
        collapsed_features_pertslice_json=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH,
        "Spec_CollapsedFeaturesPerSequenceFirst24Hours.json"),

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
rule split_first_24_hours_data_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        features_csv=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH, "CollapsedFeaturesPerSequenceFirst24Hours.csv"),
        outcomes_csv=os.path.join(DATASET_RAW_PATH, "outcomes_per_seq.csv"),
        features_json=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_SEQUENCE_PATH, "Spec_CollapsedFeaturesPerSequenceFirst24Hours.json"),
        outcomes_json=os.path.join(DATASET_RAW_PATH, "Spec_OutcomesPerSequence.json"),

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