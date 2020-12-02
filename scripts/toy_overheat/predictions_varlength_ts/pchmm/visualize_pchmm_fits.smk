'''
Visualize fit pchmm models

Usage
-----
snakemake --cores 1 --snakefile visualize_pchmm_fits

'''
PROJECT_REPO_DIR = os.environ.get("PROJECT_REPO_DIR", os.path.abspath("../../../../"))
PROJECT_CONDA_ENV_YAML = os.path.join(PROJECT_REPO_DIR, "ts_pred.yml")
RESULTS_FEAT_PER_TSTEP_PATH = "/tmp/results/toy_overheat/pchmm/"
DATASET_DIR = os.path.join(PROJECT_REPO_DIR, "datasets", "toy_overheat", "v20200515", "")


rule visualize_pchmm_fits:
    input:
        script='visualize_pchmm_fits.py'

    params:
        fits_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        data_dir=DATASET_DIR
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --data_dir {params.data_dir} \
            --fits_dir {params.fits_dir} \
        '''