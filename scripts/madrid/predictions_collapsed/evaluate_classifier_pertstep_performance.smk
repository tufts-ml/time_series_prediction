'''
Get the performance plots for the different classifiers on ICU intervention task (Deprecated)

'''

import glob

from config_loader import (
    D_CONFIG,PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML,
    RESULTS_PATH, RESULTS_PERTSTEP_PATH)

rule evaluate_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_classifier_pertstep_performance.py"),
        clf_performance_dir=RESULTS_PERTSTEP_PATH

    shell:
        '''
        python -u {input.script}\
        --clf_performance_dir {input.clf_performance_dir}\
        '''

