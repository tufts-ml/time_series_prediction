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


RESULTS_FEAT_PER_TSTEP_PATH="/cluster/tufts/hugheslab/prath01/results/mimic3/semi_supervised_pchmm/"
CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split_dir')


rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"semi-supervised-pchmm-lr={lr}-seed={seed}-init_strategy={init_strategy}-imputation_strategy={imputation_strategy}-batch_size={batch_size}-perc_labelled={perc_labelled}-n_states={n_states}-lamb={lamb}.csv").format(lr=lr, seed=seed, init_strategy=init_strategy, batch_size=batch_size, perc_labelled=perc_labelled, n_states=n_states, lamb=lamb, imputation_strategy=imputation_strategy) for lr in config['lr'] for seed in config['seed'] for init_strategy in config['init_strategy'] for batch_size in config['batch_size'] for perc_labelled in config['perc_labelled'] for n_states in config['n_states'] for lamb in config['lamb'] for imputation_strategy in config["imputation_strategy"]]
        
rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'PC-HMM', 'main_mimic_semi_supervised.py'),
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_train_first_24_hours.csv'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_valid_first_24_hours.csv'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_test_first_24_hours.csv'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_train_first_24_hours.csv'),
        y_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_valid_first_24_hours.csv'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_test_first_24_hours.csv'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dict.json')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="semi-supervised-pchmm-lr={lr}-seed={seed}-init_strategy={init_strategy}-imputation_strategy={imputation_strategy}-batch_size={batch_size}-perc_labelled={perc_labelled}-n_states={n_states}-lamb={lamb}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "semi-supervised-pchmm-lr={lr}-seed={seed}-init_strategy={init_strategy}-imputation_strategy={imputation_strategy}-batch_size={batch_size}-perc_labelled={perc_labelled}-n_states={n_states}-lamb={lamb}.csv")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --outcome_col_name {{OUTCOME_COL_NAME}} \
            --output_dir {params.output_dir} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --valid_csv_files {input.x_valid_csv},{input.y_valid_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --validation_size 0.2 \
            --lr {wildcards.lr} \
            --n_states {wildcards.n_states} \
            --imputation_strategy {wildcards.imputation_strategy} \
            --seed {wildcards.seed} \
            --batch_size {wildcards.batch_size} \
            --perc_labelled {wildcards.perc_labelled} \
            --init_strategy {wildcards.init_strategy} \
            --output_filename_prefix {params.fn_prefix} \
            --lamb {wildcards.lamb} \
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])\

