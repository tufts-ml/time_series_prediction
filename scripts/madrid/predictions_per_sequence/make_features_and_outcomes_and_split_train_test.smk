'''
Prediction of ICU Deterioration per full sequency with RNN's

Usage : filter and save multiple patient-stay-slices (0-8h, 0-16h, first 30%, last 5 hours) for evaluation
----------------------------------------------------------------------------------------------------------
>> snakemake --cores all --snakefile make_features_and_outcomes_and_split_train_test.smk filter_admissions_by_tslice_many_tslices

Usage : make a single set of features containing from all patient-stays and outcomes for each patient stay
----------------------------------------------------------------------------------------------------------
>>  snakemake --cores 1 --snakefile make_features_and_outcomes_and_split_train_test.smk make_features_and_outcomes

Usage : Split into train and test sets containing sequences
-----------------------------------------------------------
>> snakemake --cores 1 --snakefile make_features_and_outcomes_and_split_train_test.smk split_into_train_and_test

Usage : Impute missing values for train and test sets seaparately by carry forward and population mean
------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_features_and_outcomes_and_split_train_test.smk impute_missing_values_and_normalize_data


'''

sys.path.append(os.path.abspath('../predictions_collapsed'))

# Default environment variables
# Can override with local env variables
from config_loader import (
    D_CONFIG, DATASET_TOP_PATH,
    DATASET_SITE_PATH, DATASET_FEAT_PER_TSLICE_PATH,
    DATASET_FEAT_PER_TSLICE_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    DATASET_SPLIT_PATH)

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')

# evaluate on the filtered tslices
evaluate_tslice_hours_list=D_CONFIG['EVALUATE_TIMESLICE_LIST']

# filtered sequences
filtered_pertslice_csvs=[os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}","{feature}_before_icu_filtered_{tslice}_hours.csv.gz").format(feature=feature, tslice=str(tslice)) for tslice in evaluate_tslice_hours_list for feature in ['vitals', 'labs']]

rule filter_admissions_by_tslice_many_tslices:
    input:
        filtered_pertslice_csvs

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
        filtered_y_csv=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, "TSLICE={tslice}", "clinical_deterioration_outcomes_filtered_{tslice}_hours.csv.gz")
    
    shell:
        '''
            python -u {input.script} \
                --preproc_data_dir {params.preproc_data_dir} \
                --tslice "{wildcards.tslice}" \
                --output_dir {params.output_dir} \
        '''

rule make_features_and_outcomes:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'make_features_and_outcomes_per_sequence.py'),
    
    params:
        preproc_data_dir=DATASET_SITE_PATH,
        output_dir=CLF_TRAIN_TEST_SPLIT_PATH
    
    output:
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features.csv.gz"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes.csv.gz")
    
    shell:
        '''
            python -u {input.script} \
                --preproc_data_dir {params.preproc_data_dir} \
                --output_dir {params.output_dir} \
                --include_medications "True" \
        '''

rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'features.csv.gz'),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes.csv.gz"),
        features_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features_dict.json"),
        outcomes_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes_dict.json"),

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


rule impute_missing_values_and_normalize_data:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'impute_missing_values_and_normalize_data.py'),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv.gz'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json')
    
    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH
    
    shell:
        '''
            python -u {input.script} \
                --train_test_split_dir {params.train_test_split_dir} \
                --output_dir {params.train_test_split_dir} \
                --normalization "zscore" \
        '''     
