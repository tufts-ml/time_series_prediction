import os
import json

def update_os_environ_vars():
    for key, val in list(globals().items()):
        if (key.split("_")[0] in ("DATASET", "PROJECT", "RESULTS")) and isinstance(val, str):
            os.environ[key] = val
    for key, val in D_CONFIG.items():
        if isinstance(val, str):
            os.environ[key] = val

# Default environment variables
# Can override with local env variables
PROJECT_REPO_DIR = os.environ.get("PROJECT_REPO_DIR", os.path.abspath("../../../"))
PROJECT_CONDA_ENV_YAML = os.path.join(PROJECT_REPO_DIR, "ts_pred.yml")

DATASET_SCRIPTS_ROOT = os.path.join(PROJECT_REPO_DIR, 'scripts', 'madrid')

# Load this dataset's config file
# Input/output paths, etc.
with open(os.path.join(DATASET_SCRIPTS_ROOT, 'config.json'), 'r') as f:
    D_CONFIG = json.load(f)
update_os_environ_vars()

DATASET_TOP_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["DATASET_TOP_PATH_LIST"])))
update_os_environ_vars()

DATASET_SITE_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["DATASET_SITE_PATH_LIST"])))
update_os_environ_vars()

# DATASET_STD_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["STD_PATH_LIST"])))
# update_os_environ_vars()

DATASET_SPLIT_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["DATASET_SPLIT_PATH_LIST"])))
update_os_environ_vars()

DATASET_FEAT_PER_TSLICE_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["FEAT_PER_TIMESLICE_PATH_LIST"])))
update_os_environ_vars()

DATASET_COLLAPSED_FEAT_PER_TSLICE_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["COLLAPSED_FEAT_PER_TIMESLICE_PATH_LIST"])))
update_os_environ_vars()

DATASET_COLLAPSED_FEAT_PER_SEQUENCE_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["COLLAPSED_FEAT_PER_SEQUENCE_PATH_LIST"])))
update_os_environ_vars()

DATASET_FEAT_PER_SEQUENCE_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["FEAT_PER_SEQUENCE_PATH_LIST"])))
update_os_environ_vars()

DATASET_FEAT_PER_TSTEP_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["FEAT_PER_TIMESTEP_PATH_LIST"])))
update_os_environ_vars()

RESULTS_TOP_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["RESULTS_TOP_PATH_LIST"])))
update_os_environ_vars()

RESULTS_SPLIT_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["RESULTS_SPLIT_PATH_LIST"])))
update_os_environ_vars()

RESULTS_FEAT_PER_TSLICE_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["RESULTS_FEAT_PER_TIMESLICE_PATH_LIST"])))
update_os_environ_vars()

RESULTS_COLLAPSED_FEAT_PER_TSLICE_PATH = os.path.join(*list(map(os.path.expandvars,
                                                                D_CONFIG["RESULTS_COLLAPSED_FEAT_PER_TIMESLICE_PATH_LIST"])))
update_os_environ_vars()

RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH = os.path.join(*list(map(os.path.expandvars,
                                                                  D_CONFIG["RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH_LIST"])))
update_os_environ_vars()

RESULTS_FEAT_PER_SEQUENCE_PATH = os.path.join(*list(map(os.path.expandvars,
                                                        D_CONFIG["RESULTS_FEAT_PER_SEQUENCE_PATH_LIST"])))
update_os_environ_vars()

RESULTS_FEAT_PER_TSTEP_PATH = os.path.join(*list(map(os.path.expandvars,
                                                     D_CONFIG["RESULTS_FEAT_PER_TIMESTEP_PATH_LIST"])))
update_os_environ_vars()