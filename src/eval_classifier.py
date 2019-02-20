# eval_classifier.py

# Input:  Argument 1: a classifier type (currently only 'logistic' is supported)
#         Argument 2: a folder containing time-series files train.csv,
#                     valid.csv, and test.csv
#         Argument 3: data dictionary for the above files
#         --static_file: joins the indicated static file to the time-series data
#           on all columns in the static data of role 'id'
#         --validation_size: fraction of the training data to be used for
#           model selection (default 0.1)
#         - Arguments starting with 'grid_' specify the hyperparameter values
#           to be tested as a list following the hyperparameter name: for
#           example, "--grid_C 0.1 1 10". These arguments should go at the end.
#           They must be implemented for individual classifiers.
#         - Any other arguments are passed through to the classifier.
#
#         Example: python eval_classifier.py logistic <ts_folder> <data_dict>
#                  --static_file <static_file> --validation_size 0.1
#                  --class_weight balanced --grid_C 0.1 1 10
#
# Output: TODO (currently prints the accuracy)

# TODO: valid.csv as input, or create it here?

import argparse
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from sklearn.model_selection import ParameterGrid, GridSearchCV, ShuffleSplit

# Modify LogisticRegression to allow custom thresholds

class LogisticRegressionWithThreshold(LogisticRegression):

    # See LogisticRegression.__init__() in sklearn source code
    # linear_model/logistic.py. This is not robust to changes in sklearn,
    # but it won't let me pass through **kwargs to a subclass constructor.
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None, threshold=0):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.threshold = threshold

    # Same as LinearClassifierMixin.predict() in sklearn source code 
    # linear_model/base.py, but with a custom threshold rather than 0.
    def predict(self, X):
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > self.threshold).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

# Parse pre-specified command line arguments

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers()
logistic_parser = subparsers.add_parser('logistic')
logistic_parser.add_argument('--grid_C', type=float, nargs='*', default=[1])
# Decision threshold uses the output of decision_function(), the signed distance
# to the hyperplane (default 0), not predicted probabilities (default 0.5).
logistic_parser.add_argument('--grid_threshold', type=float, nargs='*',
                             default=[0])
logistic_parser.set_defaults(clf=LogisticRegressionWithThreshold,
                             default_clf_args={'solver': 'lbfgs'})

for p in [logistic_parser]: # list of all supported classifiers
    p.add_argument('ts_folder')
    p.add_argument('data_dict')
    p.add_argument('--static_file')
    p.add_argument('--validation_size', type=float, default=0.1)

args, unknown_args = parser.parse_known_args()
generic_args = ('ts_folder', 'data_dict', 'static_file', 'validation_size',
                 'clf', 'default_clf_args')
# key[5:] strips the 'grid_' prefix from the argument
param_grid = {key[5:]: vars(args)[key] for key in vars(args)
              if key not in generic_args}

# Parse unspecified arguments to be passed through to the classifier

def convert_if_float(x):
    try:
        return float(x)
    except ValueError:
        return False
    return True

passthrough_args = {}
for i in range(len(unknown_args)):
    arg = unknown_args[i]
    if arg.startswith('--'):
        val = unknown_args[i+1]
        try:
            # [2:] strips the '--' prefix
            passthrough_args[arg[2:]] = float(unknown_args[i+1])
        except ValueError:
            passthrough_args[arg[2:]] = unknown_args[i+1]


# Import data

data_dict = json.load(open(args.data_dict))
train = pd.read_csv(args.ts_folder + '/train.csv')
valid = pd.read_csv(args.ts_folder + '/valid.csv')
test = pd.read_csv(args.ts_folder + '/test.csv')
if args.static_file:
    static = pd.read_csv(args.static_file)
    static_id_cols = [c['name'] for c in data_dict['fields']
                      if c['role'] == 'id' and c['name'] in static.columns]
    train = train.merge(static, on=static_id_cols)
    valid = valid.merge(static, on=static_id_cols)
    test = test.merge(static, on=static_id_cols)

# Prepare data for classification

feature_cols = [c['name'] for c in data_dict['fields']
                if c['role'] == 'feature']
outcome_col = [c['name'] for c in data_dict['fields']
               if c['role'] == 'outcome']
if len(outcome_col) != 1:
    raise Exception('Data must have exactly one outcome column')
x_train = train[feature_cols]
y_train = np.ravel(train[outcome_col])
x_valid = valid[feature_cols]
y_valid = np.ravel(valid[outcome_col])
x_test = test[feature_cols]
y_test = np.ravel(test[outcome_col])

# Grid search (currently ignores validation data and prohibits customization)

clf = args.clf(**args.default_clf_args, **passthrough_args)
# Despite using GridSearchCV, this uses a single validation set.
# TODO: specify seed
grid = GridSearchCV(clf, param_grid,
                    cv=ShuffleSplit(test_size=args.validation_size, n_splits=1))
grid.fit(x_train, y_train)
print('Grid search results:')
for x in grid.cv_results_:
    if 'train' not in x:
        print(x, grid.cv_results_[x])
y_test_pred = grid.predict(x_test)
y_test_pred_proba = grid.predict_proba(x_test)[:, 1]
accuracy = accuracy_score(y_test, y_test_pred)
avg_precision = average_precision_score(y_test, y_test_pred_proba)
auroc = roc_auc_score(y_test, y_test_pred_proba)
print('Best accuracy: {:.3f}'.format(accuracy))
print('Best average precision: {:.3f}'.format(avg_precision))
print('Best AUROC: {:.3f}'.format(auroc))
