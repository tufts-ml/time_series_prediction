'''
Plot performance plots of all the classifiers
>> snakemake --cores 1 --snakefile plot_semi_supervised_classifiers_performance.smk plot_first_24_hours_performance 
'''
import glob
import os
import sys

sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (D_CONFIG, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML)

RESULTS_TOP_PATH="/cluster/tufts/hugheslab/prath01/results/eicu/"

rule plot_first_24_hours_performance:
    input:
        script=os.path.abspath('../src/plot_semi_supervised_classifiers_first_24_hrs_performance.py')

    params:
        performance_csv_dir=RESULTS_TOP_PATH
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        python -u {input.script} \
        --performance_csv_dir {params.performance_csv_dir} \
        '''