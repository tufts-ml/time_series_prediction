'''
Prediction of ICU Deterioration per full sequency with RNN's

Usage : make  features containing from all patient-stays and outcomes per timestep before clinical deterioration
----------------------------------------------------------------------------------------------------------
>>  snakemake --cores 1 --snakefile make_features_and_outcomes_per_tstep_custom_times.smk make_features_and_outcomes_for_custom_times_prediction

Usage : Split into train and test sets containing sequences
-----------------------------------------------------------
>> snakemake --cores 1 --snakefile make_features_and_outcomes_per_tstep_custom_times.smk split_into_train_and_test

Usage : Impute missing values for train and test sets seaparately by carry forward and population mean
------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_features_and_outcomes_per_tstep_custom_times.smk impute_missing_values_and_normalize_data


'''

sys.path.append(os.path.abspath('../predictions_collapsed'))

# Default environment variables
# Can override with local env variables
from config_loader import (
    D_CONFIG, DATASET_TOP_PATH,
    DATASET_SITE_PATH, DATASET_FEAT_PER_TSTEP_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    DATASET_SPLIT_PATH)

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_FEAT_PER_TSTEP_PATH, 'classifier_train_test_split')


rule make_features_and_outcomes_for_custom_times_prediction:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'make_features_and_outcomes_for_custom_times_prediction.py'),
    
    params:
        preproc_data_dir=DATASET_SITE_PATH,
        output_dir=CLF_TRAIN_TEST_SPLIT_PATH
    
    output:
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features_per_tstep_custom_times_labs_vitals.csv.gz"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes_per_tstep_custom_times_labs_vitals.csv.gz")
    
    shell:
        '''
            mkdir -p {{params.output_dir}} && \
            python -u {input.script} \
                --preproc_data_dir {params.preproc_data_dir} \
                --output_dir {params.output_dir} \
                --output_features_csv {output.features_csv} \
                --output_outcomes_csv {output.outcomes_csv} \
                --include_medications "False" \
        '''


rule split_into_train_and_test:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'split_dataset_by_timestamp.py'),
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features_per_tstep_custom_times_labs_vitals.csv.gz"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes_per_tstep_custom_times_labs_vitals.csv.gz"),
        features_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features_dict.json"),
        outcomes_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes_dict.json"),

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH
 
    output:  
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train_labs_vitals.csv.gz'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_valid_labs_vitals.csv.gz'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test_labs_vitals.csv.gz'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train_labs_vitals.csv.gz'),
        y_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_valid_labs_vitals.csv.gz'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test_labs_vitals.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict_labs_vitals.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict_labs_vitals.json')

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
        '''.format(
            split_test_size=D_CONFIG['SPLIT_TEST_SIZE'],
            )

rule impute_missing_values_and_normalize_data:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'impute_missing_values_and_normalize_data.py'),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train_labs_vitals.csv.gz'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_valid_labs_vitals.csv.gz'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test_labs_vitals.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict_labs_vitals.json')
    
    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH
    
    shell:
        '''
            python -u {input.script} \
                --train_test_split_dir {params.train_test_split_dir} \
                --output_dir {params.train_test_split_dir} \
                --x_train_csv {input.x_train_csv} \
                --x_valid_csv {input.x_valid_csv} \
                --x_test_csv {input.x_test_csv} \
                --x_data_dict {input.x_dict_json} \
                --normalization "zscore" \
        '''     
