'''
Train collapsed feature classifier on Madrid transfer to ICU task

>> snakemake --cores 1 --snakefile skorch_mlp_dynamic.smk
'''

# Default environment variables
# Can override with local env variables

configfile:"skorch_mlp.json"

from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH, PROJECT_CONDA_ENV_YAML,
    DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
    RESULTS_SPLIT_PATH, PROJECT_REPO_DIR,
    RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH)
#tstep_hours_list=D_CONFIG['TIMESTEP_LIST']
random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']

RESULTS_SPLIT_PATH=os.path.join(RESULTS_SPLIT_PATH, 'skorch_mlp')
RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH = os.path.join(RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 'skorch_mlp')
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 'classifier_train_test_split')

rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, "skorch_mlp_lr={lr}-weight_decay={weight_decay}-batch_size={batch_size}-scoring={scoring}-seed={seed}-lamb={lamb}-initialization_gain={initialization_gain}-n_hiddens={n_hiddens}-n_layers={n_layers}_perf.csv").format(lr=lr, weight_decay=weight_decay, batch_size=batch_size, scoring=scoring, seed=seed, lamb=lamb, initialization_gain=initialization_gain, n_hiddens=n_hiddens, n_layers=n_layers) for lr in config['lr'] for weight_decay in config['weight_decay'] for batch_size in config['batch_size'] for scoring in config['scoring'] for seed in config['seed'] for lamb in config['lamb'] for initialization_gain in config['initialization_gain'] for n_hiddens in config['n_hiddens'] for n_layers in config['n_layers']]

rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchMLP', 'main.py'),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv.gz'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_valid.csv.gz'),        
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv.gz'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv.gz'),
        y_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_valid.csv.gz'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json')

    params:
        output_dir=RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
        fn_prefix="skorch_mlp_lr={lr}-weight_decay={weight_decay}-batch_size={batch_size}-scoring={scoring}-seed={seed}-lamb={lamb}-initialization_gain={initialization_gain}-n_hiddens={n_hiddens}-n_layers={n_layers}"

    output:
        os.path.join(RESULTS_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, "skorch_mlp_lr={lr}-weight_decay={weight_decay}-batch_size={batch_size}-scoring={scoring}-seed={seed}-lamb={lamb}-initialization_gain={initialization_gain}-n_hiddens={n_hiddens}-n_layers={n_layers}_perf.csv")

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --output_filename_prefix {params.fn_prefix} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --valid_csv_files {input.x_valid_csv},{input.y_valid_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --validation_size 0.2 \
            --key_cols_to_group_when_splitting {{SPLIT_KEY_COL_NAMES}} \
            --scoring {wildcards.scoring} \
            --lr {wildcards.lr} \
            --weight_decay {wildcards.weight_decay} \
            --batch_size {wildcards.batch_size} \
            --merge_x_y False \
            --seed {wildcards.seed} \
            --lamb {wildcards.lamb} \
            --initialization_gain {wildcards.initialization_gain} \
            --n_hiddens {wildcards.n_hiddens} \
            --n_layers {wildcards.n_layers} \
            --n_splits 1 \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])
