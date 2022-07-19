'''
Train full sequence classifier on unimib shar activities

Usage:

$ snakemake --snakefile train_rnn.smk --profile ../../../profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams 

$ snakemake --cores 1 --snakefile train_rnn.smk

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores all --snakefile train_rnn.smk train_and_evaluate_classifier_many_hyperparams
'''

configfile:"rnn.json"

# Default environment variables
# Can override with local env variables
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_FEAT_PER_TSTEP_PATH, DATASET_SPLIT_FEAT_PER_TSTEP_PATH)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))

RESULTS_FEAT_PER_TSTEP_PATH = os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, 'rnn')

rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}.json").format(hidden_units=hidden_units, hidden_layers=hidden_layers, lr=lr, dropout=dropout, weight_decay=weight_decay, batch_size=batch_size) for hidden_units in config['hidden_units'] for lr in config['lr'] for hidden_layers in config['hidden_layers'] for dropout in config['dropout'] for weight_decay in config['weight_decay'] for batch_size in config['batch_size']]

rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'rnn', 'main_mimic.py'),
        x_train_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'y_test.csv'),
        x_dict_json=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'y_dict.json')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}.json")
        
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
            --hidden_layers {wildcards.hidden_layers} \
            --hidden_units {wildcards.hidden_units} \
            --lr {wildcards.lr} \
            --batch_size {wildcards.batch_size} \
            --dropout {wildcards.dropout} \
            --weight_decay {wildcards.weight_decay} \
            --output_filename_prefix {params.fn_prefix} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])
