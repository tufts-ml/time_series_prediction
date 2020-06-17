'''
Train collapsed feature classifier on Madrid transfer to ICU task

'''

# Default environment variables
# Can override with local env variables

MADRID_VERSION = os.environ.get('MADRID_VERSION', 'v20200424')
PROJECT_REPO_DIR = os.environ.get("PROJECT_REPO_DIR", os.path.abspath("../../../"))
PROJECT_CONDA_ENV_YAML = os.path.join(PROJECT_REPO_DIR, "ts_pred.yml")

MADRID_DATASET_TOP_PATH = os.path.expandvars(os.path.join("$HOME", "datasets/"))
SITE_NAME = "HUF/"
MADRID_DATASET_STD_PATH = os.path.join(MADRID_DATASET_TOP_PATH, MADRID_VERSION, SITE_NAME)

RESULTS_PATH = os.path.abspath('/tmp/html/')

item_list = [(k, v) for (k,v) in locals().items() if k.startswith('PROJECT_')]
for key, val in item_list:
    if key.startswith('PROJECT_'):
        print(val)
        os.environ[key] = val

rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'eval_classifier.py'),
        x_train_csv=os.path.join(MADRID_DATASET_STD_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(MADRID_DATASET_STD_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(MADRID_DATASET_STD_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(MADRID_DATASET_STD_PATH, 'y_test.csv'),
        x_dict_json=os.path.join(MADRID_DATASET_STD_PATH, 'Spec_CollapsedFeaturesPerSequence.json'),
        y_dict_json=os.path.join(MADRID_DATASET_STD_PATH, 'Spec-Outcomes_TransferToICU.json')

    output:
        os.path.join(RESULTS_PATH, 'report.html')

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {{RESULTS_PATH}} && \
        python -u {input.script} \
            logistic_regression \
            --outcome_col_name "transfer_to_ICU_outcome" \
            --output_dir {{RESULTS_PATH}} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --validation_size 0.15 \
            --key_cols_to_group_when_splitting "patient_id" \
            --n_splits 3 \
            --scoring roc_auc \
            --threshold_scoring balanced_accuracy \
            --grid_C 1e-7 1e-5 0.001 0.01 0.1 1 10 100 1000 \
            --class_weight balanced \
	    --tol 0.01\
	    --max_iter 5000\
            --random_state 213134
        '''.replace("{{RESULTS_PATH}}", RESULTS_PATH)
