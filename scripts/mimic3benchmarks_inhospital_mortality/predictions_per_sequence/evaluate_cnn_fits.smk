'''
Evaluate RNN fits for mimic

Usage: 

$ snakemake --cores all --snakefile evaluate_cnn_fits.smk train_and_evaluate_classifier_many_hyperparams
'''

# Default environment variables
# Can override with local env variables
sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (
    D_CONFIG,
    DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_FEAT_PER_TSTEP_PATH, DATASET_SPLIT_FEAT_PER_TSTEP_PATH)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))

RESULTS_FEAT_PER_TSTEP_PATH = os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, 'cnn')


rule evaluate_classifier:
    input:
        script='evaluate_cnn_fits.py',

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


