'''
Train full sequence classifier on Toy overheat with transient dynamics

Usage: 

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores all --snakefile evaluate_rnn_fits_for_transient_and_global_overheat.smk train_and_evaluate_classifier_many_hyperparams
'''

# Default environment variables
# Can override with local env variables

TOY_OVERHEAT_VERSION = os.environ.get('TOY_OVERHEAT_VERSION', 'v20200515')
PROJECT_REPO_DIR = os.environ.get("PROJECT_REPO_DIR", os.path.abspath("../../../../"))
PROJECT_CONDA_ENV_YAML = os.path.join(PROJECT_REPO_DIR, "ts_pred.yml")

DATASET_STD_PATH = os.path.join(PROJECT_REPO_DIR, 'datasets', 'toy_overheat', TOY_OVERHEAT_VERSION)
DATASET_TRANSIENT_SPLIT_PATH = os.path.join(DATASET_STD_PATH, 'cnn_data', 'train_test_split_dir')
DATASET_GLOBAL_SPLIT_PATH = os.path.join(DATASET_STD_PATH, 'rnn_data', 'train_test_split_dir')


RESULTS_FEAT_PER_TSTEP_PATH = "/tmp/results/toy_overheat/rnn_vs_cnn_comparison/"


rule evaluate_classifier:
    input:
        script='evaluate_rnn_fits_for_transient_and_global_overheat.py',

    params:
        fits_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.fits_dir} && \
        python -u {input.script} \
            --fits_dir  {params.fits_dir}\
        '''
