'''
Produce a collapsed feature representation and split train test for mimic3 inhospital mortality
and produce train/test CSV files

Usage
-----
snakemake --cores 1 all

# filter admissions by tslice
snakemake --cores 1 --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk filter_admissions_by_tslice_many_tslices

# collapse tslice features
snakemake --cores 1 --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk collapse_features_many_tslices

# merge the collapsed features for training tslices mentioned in config file
snakemake --cores 1 --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk merge_collapsed_features_all_tslices

# split into train test
snakemake --cores 1 --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk split_into_train_and_test

'''

sys.path.append('../predictions_collapsed/')
from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, 
    DATASET_SPLIT_FEAT_PER_TSLICE_PATH,
    RESULTS_FEAT_PER_TSLICE_PATH, 
    DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH,
    RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH)

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split_dir')

print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(CLF_TRAIN_TEST_SPLIT_PATH)

# evaluate on the filtered tslices
train_tslice_hours_list=D_CONFIG['TRAIN_TIMESLICE_LIST']
evaluate_tslice_hours_list=D_CONFIG['EVALUATE_TIMESLICE_LIST']

# filtered sequences
filtered_pertslice_csvs=[os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}","features_before_death_filtered_{tslice}_hours.csv").format(tslice=str(tslice)) for tslice in evaluate_tslice_hours_list]

# collapsed files 

collapsed_pertslice_csvs=[os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}","CollapsedFeaturesPerSequence.csv").format(tslice=str(tslice)) for tslice in evaluate_tslice_hours_list]
collapsed_pertslice_jsons=[os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}","Spec_CollapsedFeaturesPerSequence.json").format(tslice=str(tslice)) for tslice in evaluate_tslice_hours_list]

rule filter_admissions_by_tslice_many_tslices:
    input:
        filtered_pertslice_csvs

rule collapse_features_many_tslices:
    input:
        collapsed_pertslice_csvs,
        collapsed_pertslice_jsons

rule filter_admissions_by_tslice:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'filter_admissions_by_tslice.py'),
    
    params:
        preproc_data_dir = DATASET_STD_PATH,
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


rule collapse_features:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'feature_transformation.py'),
        features_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "features_before_death_filtered_{tslice}_hours.csv"),
        features_spec_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerTimestep.json'),
        tstop_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "tstops_filtered_{tslice}_hours.csv")

    output:
        collapsed_features_pertslice_csv=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "CollapsedFeaturesPerSequence.csv"),
        collapsed_features_pertslice_json=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "Spec_CollapsedFeaturesPerSequence.json"),

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --input {input.features_csv} \
            --data_dict {input.features_spec_json} \
            --output "{output.collapsed_features_pertslice_csv}" \
            --data_dict_output "{output.collapsed_features_pertslice_json}" \
            --tstops {input.tstop_csv} \
            --collapse_range_features "std hours_since_measured present slope median min max" \
            --range_pairs "[('50%','100%'), ('0%','100%'), ('T-16h','T-0h'), ('T-24h','T-0h')]" \
            --collapse \
        '''
        
rule merge_collapsed_features_all_tslices:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'merge_features_all_tslices.py')
    
    params:
        collapsed_tslice_folder=os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH, "TSLICE="),
        tslice_folder=os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, "TSLICE="),
        tslice_list=train_tslice_hours_list,
        preproc_data_dir = DATASET_STD_PATH,
        output_dir=CLF_TRAIN_TEST_SPLIT_PATH
    
    output:
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features.csv"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes.csv")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {{params.output_dir}} && \
        python -u {input.script} \
            --collapsed_tslice_folder {params.collapsed_tslice_folder} \
            --tslice_folder {params.tslice_folder} \
            --preproc_data_dir {params.preproc_data_dir} \
            --tslice_list "{params.tslice_list}" \
            --output_dir {params.output_dir}\
        '''

rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features.csv"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes.csv"),
        features_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "Spec_features.json"),
        outcomes_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "Spec_outcomes.json"),

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