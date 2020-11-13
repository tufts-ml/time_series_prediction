'''
Train full sequence classifier on Madrid transfer to ICU task

Usage: 
$ snakemake --cores 1 --snakefile train_pchmm.smk

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores all --snakefile train_pchmm.smk train_and_evaluate_classifier_many_hyperparams
'''

configfile:"pchmm.json"

# Default environment variables
# Can override with local env variables
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_FEAT_PER_TSTEP_PATH)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))

RESULTS_FEAT_PER_TSTEP_PATH = os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, 'rnn')
print("Results will be saved in : %s"%RESULTS_FEAT_PER_TSTEP_PATH)
rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"pchmm-lr={lr}-seed={seed}.json").format(lr=lr, seed=seed) for lr in config['lr'] for seed in config['seed']]

rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'PC-VAE', 'main.py'),
        x_train_csv=os.path.join(DATASET_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_PATH, 'y_test.csv'),
        x_dict_json=os.path.join(DATASET_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(DATASET_SPLIT_PATH, 'y_dict.json')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="pchmm-lr={lr}-seed={seed}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "pchmm-lr={lr}-seed={seed}.json")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --validation_size 0.15 \
            --lr {wildcards.lr} \
            --seed {wildcards.seed} \
            --output_filename_prefix {params.fn_prefix} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])
