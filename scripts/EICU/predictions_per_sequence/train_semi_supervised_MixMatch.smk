'''
Train MixMatch ON EICU Inhospital Mortality Task

Usage
-----
snakemake --cores 1 --snakefile train_semi_supervised_MixMatch.smk train_and_evaluate_classifier_many_hyperparams

Schedule as slurm jobs
----------------------
$ snakemake --snakefile train_semi_supervised_MixMatch.smk --profile ../../../profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams

'''

# Default environment variables
# Can override with local env variables
configfile:"MixMatch.json"

# Default environment variables
# Can override with local env variables
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))


RESULTS_FEAT_PER_TSTEP_PATH="/cluster/tufts/hugheslab/prath01/results/eicu/MixMatch/v05112022/"
CLF_TRAIN_TEST_SPLIT_PATH = "/cluster/tufts/hugheslab/prath01/datasets/eicu_ssl/"


rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"MixMatch-lr={lr}-seed={seed}-batch_size={batch_size}-perc_labelled={perc_labelled}.csv").format(lr=lr, seed=seed, batch_size=batch_size, perc_labelled=perc_labelled) for lr in config['lr'] for seed in config['seed'] for batch_size in config['batch_size'] for perc_labelled in config['perc_labelled']]
        
rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'MixMatch', 'main.py'),
        x_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'X_train.npy'),
        x_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'X_valid.npy'),
        x_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'X_test.npy'),
        y_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'y_train.npy'),
        y_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'y_valid.npy'),
        y_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'percentage_labelled_sequnces={perc_labelled}', 'y_test.npy')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="MixMatch-lr={lr}-seed={seed}-batch_size={batch_size}-perc_labelled={perc_labelled}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "MixMatch-lr={lr}-seed={seed}-batch_size={batch_size}-perc_labelled={perc_labelled}.csv")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --output_dir {params.output_dir} \
            --train_np_files {input.x_train_np},{input.y_train_np} \
            --valid_np_files {input.x_valid_np},{input.y_valid_np} \
            --test_np_files {input.x_test_np},{input.y_test_np} \
            --lr {wildcards.lr} \
            --manualSeed {wildcards.seed} \
            --perc_labelled {wildcards.perc_labelled} \
            --batch_size {wildcards.batch_size} \
            --output_filename_prefix {params.fn_prefix} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\

