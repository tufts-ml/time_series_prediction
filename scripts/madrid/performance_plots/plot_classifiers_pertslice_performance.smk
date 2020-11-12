'''
Plot performance plots of all the classifiers
>> snakemake --cores 1 --snakefile plot_classifiers_pertslice_performance.smk plot_performance 
'''
import glob
import os
import sys

sys.path.append(os.path.abspath('../predictions_collapsed'))
from config_loader import (D_CONFIG, PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, RESULTS_SPLIT_PATH)

rule plot_performance:
    input:
        script=os.path.abspath('../src/plot_classifiers_pertslice_performance.py')

    params:
        performance_csv_dir=os.path.join(RESULTS_SPLIT_PATH, "classifier_per_tslice_performance")
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        python -u {input.script} \
        --performance_csv_dir {params.performance_csv_dir} \
        --output_dir {params.performance_csv_dir} \
        '''
