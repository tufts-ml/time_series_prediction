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

