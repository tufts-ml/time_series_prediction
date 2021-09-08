'''
Produce a collapsed feature representation on this toy dataset

Usage
-----
snakemake --cores 1 --snakefile split_train_test.smk split_into_train_and_test
'''

# Default environment variables
# Can override with local env variables

TOY_OVERHEAT_VERSION = os.environ.get('TOY_OVERHEAT_VERSION', 'v20200515')
PROJECT_REPO_DIR = os.environ.get("PROJECT_REPO_DIR", os.path.abspath("../../../../"))
PROJECT_CONDA_ENV_YAML = os.path.join(PROJECT_REPO_DIR, "ts_pred.yml")

DATASET_STD_PATH = os.path.join(PROJECT_REPO_DIR, 'datasets', 'toy_overheat', TOY_OVERHEAT_VERSION)
DATASET_SPLIT_PATH = os.path.join(PROJECT_REPO_DIR, 'datasets', 'toy_overheat', TOY_OVERHEAT_VERSION, 'classifier_train_test_split')

rule all:
    input:
        x_train_csv=os.path.join(DATASET_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_PATH, 'y_test.csv')

rule split_into_train_and_test:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'split_dataset.py'),
        x_csv=os.path.join(DATASET_STD_PATH, 'rnn_features_per_tstep.csv.gz'),
        x_json=os.path.join(DATASET_STD_PATH, 'Spec_FeaturesPerTimestep.json'),
        y_csv=os.path.join(DATASET_STD_PATH, 'rnn_outcomes_per_tstep.csv.gz'),
        y_json=os.path.join(DATASET_STD_PATH, 'Spec_OutcomesPerSequence.json')

    output:
        x_train_csv=os.path.join(DATASET_SPLIT_PATH, 'x_train.csv.gz'),
        x_valid_csv=os.path.join(DATASET_SPLIT_PATH, 'x_valid.csv.gz'),
        x_test_csv=os.path.join(DATASET_SPLIT_PATH, 'x_test.csv.gz'),
        y_train_csv=os.path.join(DATASET_SPLIT_PATH, 'y_train.csv.gz'),
        y_valid_csv=os.path.join(DATASET_SPLIT_PATH, 'y_valid.csv.gz'),
        y_test_csv=os.path.join(DATASET_SPLIT_PATH, 'y_test.csv.gz'),
        x_json=os.path.join(DATASET_SPLIT_PATH, 'x_dict.json'),
        y_json=os.path.join(DATASET_SPLIT_PATH, 'y_dict.json')

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p DATASET_SPLIT_PATH \
        && python -u {input.script} \
            --input {input.x_csv} \
            --data_dict {input.x_json} \
            --test_size 0.1 \
            --group_cols sequence_id \
            --train_csv_filename {output.x_train_csv} \
            --valid_csv_filename {output.x_valid_csv} \
            --test_csv_filename {output.x_test_csv} \
            --output_data_dict_filename {output.x_json} \
        && python -u {input.script} \
            --input {input.y_csv} \
            --data_dict {input.y_json} \
            --test_size 0.1 \
            --group_cols sequence_id \
            --train_csv_filename {output.y_train_csv} \
            --valid_csv_filename {output.y_valid_csv} \
            --test_csv_filename {output.y_test_csv} \
            --output_data_dict_filename {output.y_json} \
        '''.replace("DATASET_SPLIT_PATH", DATASET_SPLIT_PATH)
