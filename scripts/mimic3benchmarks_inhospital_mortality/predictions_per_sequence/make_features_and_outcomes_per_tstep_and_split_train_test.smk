'''
Prediction of inhospital mortality per full sequence with RNN's

Usage : make  features containing from all patient-stays and outcomes per timestep before clinical deterioration
----------------------------------------------------------------------------------------------------------
>>  snakemake --cores 1 --snakefile make_features_and_outcomes_per_tstep_and_split_train_test.smk make_features_and_outcomes_for_dynamic_output_prediction

>> snakemake --snakefile make_features_and_outcomes_per_tstep_and_split_train_test.smk --profile ../../../profiles/hugheslab_cluster/ make_features_and_outcomes_for_dynamic_output_prediction

Usage : Split into train and test sets containing sequences
-----------------------------------------------------------
>> snakemake --cores 1 --snakefile make_features_and_outcomes_per_tstep_and_split_train_test.smk split_into_train_and_test

Usage : Impute missing values for train and test sets seaparately by carry forward and population mean
------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile make_features_and_outcomes_per_tstep_and_split_train_test.smk impute_missing_values_and_normalize_data


'''

sys.path.append(os.path.abspath('../predictions_collapsed'))

sys.path.append('../predictions_collapsed/')
from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, 
    DATASET_FEAT_PER_TSTEP_DYNAMIC_INPUT_OUTPUT_PATH)

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_FEAT_PER_TSTEP_DYNAMIC_INPUT_OUTPUT_PATH, 'classifier_train_test_split_dir')


rule make_features_and_outcomes_for_dynamic_output_prediction:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'make_features_and_outcomes_for_dynamic_output_prediction.py'),
        features_csv=os.path.join(DATASET_STD_PATH, 'features_per_tstep.csv'),
        features_spec_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerTimestep.json'), 
        outcomes_csv=os.path.join(DATASET_STD_PATH, "outcomes_per_seq.csv"),
        outcomes_spec_json=os.path.join(DATASET_STD_PATH, "Spec_OutcomesPerSequence.json")

    params:
        output_dir=CLF_TRAIN_TEST_SPLIT_PATH

    output:
        features_per_tstep_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features_per_tstep.csv.gz"),
        outcomes_per_tstep_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes_per_tstep.csv.gz")
    
    shell:
        '''
            mkdir -p {{params.output_dir}} && \
            python -u {input.script} \
                --features_csv {input.features_csv} \
                --features_json {input.features_spec_json} \
                --outcomes_csv {input.outcomes_csv} \
                --outcomes_json {input.outcomes_spec_json} \
                --output_dir {params.output_dir} \
        '''


rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features_per_tstep.csv.gz"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes_per_tstep.csv.gz"),
        features_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features_dict.json"),
        outcomes_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes_dict.json"),

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH
 
    output:  
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv.gz'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_valid.csv.gz'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv.gz'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv.gz'),
        y_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_valid.csv.gz'),
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
                --valid_size {split_valid_size} \
                --test_size {split_test_size} \
                --train_csv_filename {{output.x_train_csv}} \
                --valid_csv_filename {{output.x_valid_csv}} \
                --test_csv_filename {{output.x_test_csv}} \
                --output_data_dict_filename {{output.x_dict_json}} \

            python -u {{input.script}} \
                --input {{input.outcomes_csv}} \
                --data_dict {{input.outcomes_json}} \
                --valid_size {split_valid_size} \
                --test_size {split_test_size} \
                --train_csv_filename {{output.y_train_csv}} \
                --valid_csv_filename {{output.y_valid_csv}} \
                --test_csv_filename {{output.y_test_csv}} \
                --output_data_dict_filename {{output.y_dict_json}} \
        '''.format(
            split_test_size=D_CONFIG['SPLIT_TEST_SIZE'],
            split_valid_size=D_CONFIG['SPLIT_VALID_SIZE']
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