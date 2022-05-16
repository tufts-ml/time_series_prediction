'''
Train PC-HMM ON MIMIC Inhospital Mortality Task

Usage
-----
snakemake --cores all --snakefile train_semi_supervised_pchmm.smk train_and_evaluate_classifier_many_hyperparams

Schedule as slurm jobs
----------------------
$ snakemake --snakefile train_semi_supervised_pchmm.smk --profile ../../../profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams

'''

# Default environment variables
# Can override with local env variables
configfile:"semi_supervised_pchmm.json"

# Default environment variables
# Can override with local env variables
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    DATASET_SPLIT_FEAT_PER_TSLICE_PATH)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))


RESULTS_FEAT_PER_TSTEP_PATH="/cluster/tufts/hugheslab/prath01/results/mimic3/semi_supervised_pchmm/v05112022/"
CLF_TRAIN_TEST_SPLIT_PATH = "/cluster/tufts/hugheslab/prath01/datasets/mimic3_ssl/"


rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"semi-supervised-pchmm-lr={lr}-seed={seed}-init_strategy={init_strategy}-batch_size={batch_size}-perc_labelled={perc_labelled}-predictor_l2_penalty={predictor_l2_penalty}-n_states={n_states}-lamb={lamb}.csv").format(lr=lr, seed=seed, init_strategy=init_strategy, batch_size=batch_size, perc_labelled=perc_labelled, n_states=n_states, lamb=lamb, predictor_l2_penalty=predictor_l2_penalty) for lr in config['lr'] for seed in config['seed'] for init_strategy in config['init_strategy'] for batch_size in config['batch_size'] for perc_labelled in config['perc_labelled'] for n_states in config['n_states'] for lamb in config['lamb'] for predictor_l2_penalty in config['predictor_l2_penalty']]
        
rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'PC-HMM', 'main_mimic_semi_supervised.py'),
        x_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'X_train.npy'),
        x_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'X_valid.npy'),
        x_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'X_test.npy'),
        y_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'y_train.npy'),
        y_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'y_valid.npy'),
        y_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'y_test.npy')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="semi-supervised-pchmm-lr={lr}-seed={seed}-init_strategy={init_strategy}-batch_size={batch_size}-perc_labelled={perc_labelled}-predictor_l2_penalty={predictor_l2_penalty}-n_states={n_states}-lamb={lamb}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "semi-supervised-pchmm-lr={lr}-seed={seed}-init_strategy={init_strategy}-batch_size={batch_size}-perc_labelled={perc_labelled}-predictor_l2_penalty={predictor_l2_penalty}-n_states={n_states}-lamb={lamb}.csv")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --train_np_files {input.x_train_np},{input.y_train_np} \
            --valid_np_files {input.x_valid_np},{input.y_valid_np} \
            --test_np_files {input.x_test_np},{input.y_test_np} \
            --lr {wildcards.lr} \
            --n_states {wildcards.n_states} \
            --seed {wildcards.seed} \
            --batch_size {wildcards.batch_size} \
            --perc_labelled {wildcards.perc_labelled} \
            --predictor_l2_penalty {wildcards.predictor_l2_penalty} \
            --init_strategy {wildcards.init_strategy} \
            --output_filename_prefix {params.fn_prefix} \
            --lamb {wildcards.lamb} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\

