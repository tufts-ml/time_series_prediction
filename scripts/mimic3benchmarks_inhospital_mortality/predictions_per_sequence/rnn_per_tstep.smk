'''
Train full sequence classifier on Madrid transfer to ICU task

Usage: 

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores 1 --snakefile rnn_per_tstep.smk train_and_evaluate_classifier_many_hyperparams

$ snakemake --snakefile rnn_per_tstep.smk --profile ../../../profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams
'''


configfile:"rnn.json"

sys.path.append(os.path.abspath('../predictions_collapsed'))

sys.path.append('../predictions_collapsed/')
from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, 
    DATASET_FEAT_PER_TSTEP_DYNAMIC_INPUT_OUTPUT_PATH)

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_FEAT_PER_TSTEP_DYNAMIC_INPUT_OUTPUT_PATH, 'classifier_train_test_split_dir')
RESULTS_FEAT_PER_TSTEP_PATH = "/cluster/tufts/hugheslab/prath01/results/mimic3/features_per_tstep_dynamic_input_output/rnn_per_tstep/"

rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"rnn_per_tstep_hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-seed={seed}-batch_size={batch_size}-scoring={scoring}.json").format(hidden_units=hidden_units, hidden_layers=hidden_layers, lr=lr, dropout=dropout, weight_decay=weight_decay, seed=seed, batch_size=batch_size, scoring=scoring) for hidden_units in config['hidden_units'] for lr in config['lr'] for hidden_layers in config['hidden_layers'] for dropout in config['dropout'] for weight_decay in config['weight_decay'] for seed in config['param_init_seed'] for batch_size in config['batch_size'] for scoring in config['scoring']]


rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'rnn', 'main_per_tstep.py'),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train_imputed.csv'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_valid_imputed.csv'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test_imputed.csv'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv.gz'),
        y_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_valid.csv.gz'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json')

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="rnn_per_tstep_hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-seed={seed}-batch_size={batch_size}-scoring={scoring}",
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "rnn_per_tstep_hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-seed={seed}-batch_size={batch_size}-scoring={scoring}.json")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --valid_csv_files {input.x_valid_csv},{input.y_valid_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --key_cols_to_group_when_splitting {{SPLIT_KEY_COL_NAMES}} \
            --seed {wildcards.seed} \
            --validation_size 0.25 \
            --hidden_layers {wildcards.hidden_layers} \
            --hidden_units {wildcards.hidden_units} \
            --lr {wildcards.lr} \
            --batch_size {wildcards.batch_size} \
            --dropout {wildcards.dropout} \
            --weight_decay {wildcards.weight_decay} \
            --output_filename_prefix {params.fn_prefix} \
            --scoring {wildcards.scoring}\
            --merge_x_y False
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])