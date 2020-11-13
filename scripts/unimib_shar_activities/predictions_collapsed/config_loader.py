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

DATASET_SCRIPTS_ROOT = os.path.join(PROJECT_REPO_DIR, 'scripts', 'unimib_shar_activities')

# Load this dataset's config file
# Input/output paths, etc.
with open(os.path.join(DATASET_SCRIPTS_ROOT, 'config.json'), 'r') as f:
    D_CONFIG = json.load(f)
update_os_environ_vars()

DATASET_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["DATASET_PATH_LIST"])))
update_os_environ_vars()

DATASET_STD_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["STD_PATH_LIST"])))
update_os_environ_vars()

DATASET_SPLIT_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["SPLIT_PATH_LIST_FEAT_PER_SEQUENCE"])))
update_os_environ_vars()

DATASET_SPLIT_FEAT_PER_TSTEP_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["SPLIT_PATH_LIST_FEAT_PER_TIMESTEP"])))
update_os_environ_vars()

RESULTS_TOP_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["RESULTS_TOP_PATH_LIST"])))
update_os_environ_vars()

RESULTS_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["RESULTS_PATH_LIST_COLLAPSED_FEAT_PER_SEQUENCE"])))
update_os_environ_vars()

RESULTS_FEAT_PER_TSTEP_PATH = os.path.join(*list(map(os.path.expandvars, D_CONFIG["RESULTS_PATH_LIST_FEAT_PER_TSTEP"])))
update_os_environ_vars()
