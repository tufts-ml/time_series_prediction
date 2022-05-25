'''
Train full sequence classifier on EICU inhospital mortality task

Usage:

Schedule as slurm jobs
----------------------
$ snakemake --snakefile train_gru_d_semi_supervised.smk --profile ../../../profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams

Train single hyperparam at a time
---------------------------------

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores 1 --snakefile train_gru_d_semi_supervised.smk train_and_evaluate_classifier_many_hyperparams

'''

configfile:"semi_supervised_gru_d.json"

# Default environment variables
# Can override with local env variables
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))


RESULTS_FEAT_PER_TSTEP_PATH="/cluster/tufts/hugheslab/prath01/results/eicu/GRUD/v05112022/"
CLF_TRAIN_TEST_SPLIT_PATH = "/cluster/tufts/hugheslab/prath01/datasets/eicu_ssl/"


rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"final_perf_GRUD-semi_supervised-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}-perc_labelled={perc_labelled}.csv").format(lr=lr, dropout=dropout, weight_decay=weight_decay, batch_size=batch_size, seed=seed, perc_labelled=perc_labelled) for lr in config['lr'] for dropout in config['dropout'] for weight_decay in config['weight_decay'] for batch_size in config['batch_size'] for seed in config['param_init_seed'] for perc_labelled in config['perc_labelled']]

rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'GRU_D', 'main_mimic_semi_supervised.py'),
        x_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'X_train.npy'),
        x_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'X_valid.npy'),
        x_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'X_test.npy'),
        y_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'y_train.npy'),
        y_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'y_valid.npy'),
        y_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'y_test.npy')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="GRUD-semi_supervised-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}-perc_labelled={perc_labelled}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "final_perf_GRUD-semi_supervised-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-seed={seed}-perc_labelled={perc_labelled}.csv")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --train_np_files {input.x_train_np},{input.y_train_np} \
            --valid_np_files {input.x_valid_np},{input.y_valid_np} \
            --test_np_files {input.x_test_np},{input.y_test_np} \
            --lr {wildcards.lr} \
            --seed {wildcards.seed} \
            --batch_size {wildcards.batch_size} \
            --dropout {wildcards.dropout} \
            --l2_penalty {wildcards.weight_decay} \
            --output_filename_prefix {params.fn_prefix} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])

