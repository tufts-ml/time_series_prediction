'''
Train full sequence classifier on mimic inhospital mortality task

Usage: 
$ snakemake --cores 1 --snakefile train_rnn_features_per_slices.smk

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores 1 --snakefile train_rnn_features_per_slices.smk train_and_evaluate_classifier_many_hyperparams
'''

configfile:"rnn_multiple_tslices.json"

# Default environment variables
# Can override with local env variables
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_FEAT_PER_TSLICE_PATH, DATASET_SPLIT_FEAT_PER_TSLICE_PATH)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))

RESULTS_FEAT_PER_TSLICE_PATH = os.path.join(RESULTS_FEAT_PER_TSLICE_PATH, 'rnn')
CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split_dir')


rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSLICE_PATH,"hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}.json").format(hidden_units=hidden_units, hidden_layers=hidden_layers, lr=lr, dropout=dropout, weight_decay=weight_decay, batch_size=batch_size) for hidden_units in config['hidden_units'] for lr in config['lr'] for hidden_layers in config['hidden_layers'] for dropout in config['dropout'] for weight_decay in config['weight_decay'] for batch_size in config['batch_size']]


rule train_and_evaluate_classifier:
    input:
        script=os.path.abspath('../src/train_rnn_on_patient_stay_slice_sequences.py')

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        train_tslices=['90%', '60%', '20%'],
        output_dir=RESULTS_FEAT_PER_TSLICE_PATH,
        tstops_dir=DATASET_SPLIT_FEAT_PER_TSLICE_PATH,
        fn_prefix="hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}",
        pretrained_model_dir=os.path.join(RESULTS_FEAT_PER_TSLICE_PATH, "pretrained_model")
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSLICE_PATH, "hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}.json")
        
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
            --seed 1111 \
            --validation_size 0.15 \
            --hidden_layers {wildcards.hidden_layers} \
            --hidden_units {wildcards.hidden_units} \
            --lr {wildcards.lr} \
            --batch_size 7000 \
            --dropout {wildcards.dropout} \
            --weight_decay {wildcards.weight_decay} \
            --output_filename_prefix {params.fn_prefix} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])
