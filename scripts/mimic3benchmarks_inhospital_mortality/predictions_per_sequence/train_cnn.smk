'''
Train full sequence classifier on mimic inhospital mortality task

Usage: 
$ snakemake --cores 1 --snakefile train_cnn.smk

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores all --snakefile train_cnn.smk train_and_evaluate_classifier_many_hyperparams
'''

configfile:"cnn_single_config.json"

# Default environment variables
# Can override with local env variables
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_FEAT_PER_TSTEP_PATH, DATASET_SPLIT_FEAT_PER_TSLICE_PATH)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))

CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split_dir')
RESULTS_FEAT_PER_TSTEP_PATH = os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, 'cnn')

rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"conv_layers={n_conv_layers}-filters={n_filters}-kernel_size={kernel_size}-stride={stride}-pool={pool_size}-dense={dense_units}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}.csv").format(n_conv_layers=n_conv_layers, n_filters=n_filters, kernel_size=kernel_size, stride=stride, pool_size=pool_size, dense_units=dense_units, lr=lr, dropout=dropout, weight_decay=weight_decay, batch_size=batch_size, seed=seed) for n_conv_layers in config['n_conv_layers'] for n_filters in config['n_filters'] for kernel_size in config['kernel_size'] for stride in config['stride'] for pool_size in config['pool_size'] for dense_units in config['dense_units'] for lr in config['lr'] for dropout in config['dropout'] for weight_decay in config['weight_decay'] for batch_size in config['batch_size'] for seed in config['seed']]

rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'cnn', 'main.py'),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test.csv'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="conv_layers={n_conv_layers}-filters={n_filters}-kernel_size={kernel_size}-stride={stride}-pool={pool_size}-dense={dense_units}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "conv_layers={n_conv_layers}-filters={n_filters}-kernel_size={kernel_size}-stride={stride}-pool={pool_size}-dense={dense_units}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}.csv")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --outcome_col_name  {{OUTCOME_COL_NAME}}\
            --output_dir {params.output_dir} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --validation_size 0.15 \
            --n_filters {wildcards.n_filters} \
            --kernel_size {wildcards.kernel_size} \
            --n_conv_layers {wildcards.n_conv_layers} \
            --stride {wildcards.stride} \
            --pool_size {wildcards.pool_size} \
            --dense_units {wildcards.dense_units} \
            --lr {wildcards.lr} \
            --batch_size {wildcards.batch_size} \
            --dropout {wildcards.dropout} \
            --weight_decay {wildcards.weight_decay} \
            --output_filename_prefix {params.fn_prefix} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])


