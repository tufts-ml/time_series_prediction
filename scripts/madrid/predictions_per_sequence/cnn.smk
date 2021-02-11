'''
Train full sequence classifier on Madrid transfer to ICU task

Usage: 

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores 1 --snakefile cnn.smk train_and_evaluate_classifier_many_hyperparams
'''

configfile:"cnn.json"

# Default environment variables
# Can override with local env variables
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    DATASET_SITE_PATH, DATASET_SPLIT_PATH,
    DATASET_FEAT_PER_TSLICE_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_SPLIT_PATH, RESULTS_FEAT_PER_TSTEP_PATH)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))

RESULTS_SPLIT_PATH=os.path.join(RESULTS_SPLIT_PATH, 'cnn')
RESULTS_FEAT_PER_TSTEP_PATH = os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, 'cnn')
random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')
           
rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"conv_layers={n_conv_layers}-filters={n_filters}-kernel_size={kernel_size}-stride={stride}-pool={pool_size}-dense={dense_units}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}.json").format(n_conv_layers=n_conv_layers, n_filters=n_filters, kernel_size=kernel_size, stride=stride, pool_size=pool_size, dense_units=dense_units, lr=lr, dropout=dropout, weight_decay=weight_decay, batch_size=batch_size, seed=seed) for n_conv_layers in config['n_conv_layers'] for n_filters in config['n_filters'] for kernel_size in config['kernel_size'] for stride in config['stride'] for pool_size in config['pool_size'] for dense_units in config['dense_units'] for lr in config['lr'] for dropout in config['dropout'] for weight_decay in config['weight_decay'] for batch_size in config['batch_size'] for seed in config['seed']]


rule train_and_evaluate_classifier:
    input:
        script=os.path.abspath('../src/train_cnn_on_patient_stay_slice_sequences.py')

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        train_tslices=['90%', '30%'],
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        tstops_dir=DATASET_FEAT_PER_TSLICE_PATH,
        fn_prefix="conv_layers={n_conv_layers}-filters={n_filters}-kernel_size={kernel_size}-stride={stride}-pool={pool_size}-dense={dense_units}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}",
        pretrained_model_dir=os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "pretrained_model")
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "conv_layers={n_conv_layers}-filters={n_filters}-kernel_size={kernel_size}-stride={stride}-pool={pool_size}-dense={dense_units}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}.json")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --train_test_split_dir {params.train_test_split_dir}\
            --train_tslices "{params.train_tslices}" \
            --tstops_dir {params.tstops_dir} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --seed {wildcards.seed} \
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
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])
