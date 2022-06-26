'''
Train full sequence classifier on mimic inhospital mortality task

Usage:

Schedule as slurm jobs
----------------------
$ snakemake --snakefile train_gru_d.smk --profile ../../../profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams

Train single hyperparam at a time
---------------------------------
$ snakemake --cores 1 --snakefile train_gru_d.smk


To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores all --snakefile train_rnn.smk train_and_evaluate_classifier_many_hyperparams

'''

configfile:"gru_d.json"

# Default environment variables
# Can override with local env variables
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    DATASET_SPLIT_FEAT_PER_TSLICE_PATH)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))


RESULTS_FEAT_PER_TSTEP_PATH="/cluster/tufts/hugheslab/prath01/results/mimic3/gru_d/"
CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split_dir')


rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}history.json").format(hidden_units=hidden_units, hidden_layers=hidden_layers, lr=lr, dropout=dropout, weight_decay=weight_decay, batch_size=batch_size, seed=seed) for hidden_units in config['hidden_units'] for lr in config['lr'] for hidden_layers in config['hidden_layers'] for dropout in config['dropout'] for weight_decay in config['weight_decay'] for batch_size in config['batch_size'] for seed in config['param_init_seed']]

rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'GRU_D', 'main_mimic.py'),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train_first_24_hours_imputed.csv'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_valid_first_24_hours_imputed.csv'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test_first_24_hours_imputed.csv'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train_first_24_hours.csv'),
        y_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_valid_first_24_hours.csv'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test_first_24_hours.csv'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict_imputed.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}history.json")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --valid_csv_files {input.x_valid_csv},{input.y_valid_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --validation_size 0.15 \
            --hidden_layers {wildcards.hidden_layers} \
            --hidden_units {wildcards.hidden_units} \
            --lr {wildcards.lr} \
            --seed {wildcards.seed} \
            --batch_size {wildcards.batch_size} \
            --dropout {wildcards.dropout} \
            --weight_decay {wildcards.weight_decay} \
            --output_filename_prefix {params.fn_prefix} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])

