'''
Train full sequence classifier on Madrid transfer to ICU task

Usage: 

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores 2 --snakefile rnn.smk train_and_evaluate_classifier_many_hyperparams
'''


configfile:"rnn.json"

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

RESULTS_SPLIT_PATH=os.path.join(RESULTS_SPLIT_PATH, 'rnn')
RESULTS_FEAT_PER_TSTEP_PATH = os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, 'rnn')
random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
CLF_TRAIN_TEST_SPLIT_PATH=os.path.join(DATASET_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split')
           
rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-seed={seed}.json").format(hidden_units=hidden_units, hidden_layers=hidden_layers, lr=lr, dropout=dropout,
        weight_decay=weight_decay, seed=seed) for hidden_units in config['hidden_units'] for lr in config['lr'] for hidden_layers in config['hidden_layers'] for dropout in config['dropout'] for weight_decay in config['weight_decay'] for seed in config['param_init_seed']]


rule train_and_evaluate_classifier:
    input:
        script=os.path.abspath('../src/train_rnn_on_patient_stay_slice_sequences.py')

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        train_tslices=['90%', '60%', '20%'],
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        tstops_dir=DATASET_FEAT_PER_TSLICE_PATH,
        fn_prefix="hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-seed={seed}",
        pretrained_model_dir=os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "pretrained_model")
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "hiddens={hidden_units}-layers={hidden_layers}-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-seed={seed}.json")
        
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
            --hidden_layers {wildcards.hidden_layers} \
            --hidden_units {wildcards.hidden_units} \
            --lr {wildcards.lr} \
            --batch_size 6000 \
            --dropout {wildcards.dropout} \
            --weight_decay {wildcards.weight_decay} \
            --output_filename_prefix {params.fn_prefix} \
            --pretrained_model_dir {params.pretrained_model_dir} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\
           .replace("{{SPLIT_KEY_COL_NAMES}}", D_CONFIG["SPLIT_KEY_COL_NAMES"])
