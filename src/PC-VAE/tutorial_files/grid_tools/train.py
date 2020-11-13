import pickle
import sys, os
from pcvae.experiments import run_trials

args = {}
with open(os.path.join(sys.argv[1], 'config.pkl'), 'rb') as f:
    args = pickle.load(f)

results = run_trials(return_data=False, save=sys.argv[1], **args)
del results['model']
with open(os.path.join(sys.argv[1], 'results.pkl'), 'wb') as f:
    args = pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

