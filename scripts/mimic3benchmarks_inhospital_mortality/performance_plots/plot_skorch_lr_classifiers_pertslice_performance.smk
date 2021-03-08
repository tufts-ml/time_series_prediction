'''
Plot performance plots of all the classifiers
>> snakemake --cores 1 --snakefile plot_skorch_lr_classifiers_pertslice_performance.smk plot_performance 
'''
import glob
import os
import sys

sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (D_CONFIG, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, RESULTS_TOP_PATH)

RESULTS_PATH="/cluster/tufts/hugheslab/prath01/results/mimic3/skorch_logistic_regression/classifier_per_tslice_performance"

rule plot_performance:
    input:
        script=os.path.abspath('../src/plot_skorch_lr_classifiers_pertslice_performance.py')

    params:
        performance_csv_dir=RESULTS_PATH
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        python -u {input.script} \
        --performance_csv_dir {params.performance_csv_dir} \
        '''