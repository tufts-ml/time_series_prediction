# eval_classifier.py

# Input:  First argument: a classifier type ('logistic', 'dtree', 'rforest',
#             'mlp')
#         --ts_dir: (required) a directory containing time-series files
#             train.csv and test.csv
#         --data_dict: (required) data dictionary for the above files
#         --static_file: joins the indicated static file to the time-series data
#             on all columns in the static data of role 'id'
#         --validation_size: fraction of the training data to be used for
#             model selection (default 0.1)
#         --scoring: scoring method to determine best hyperparameters, passed
#             directly to GridSearchCV as 'scoring' parameter
#         --thresholds: list of threshold values to be tested; classifier must
#             support a 'threshold' parameter
#         --threshold_scoring: scoring method to determine best threshold;
#             default is 'balanced_accuracy'
#         - Arguments starting with '--grid_' specify the hyperparameter values
#           to be tested as a list following the hyperparameter name: for
#           example, "--grid_C 0.1 1 10". These arguments should go at the end.
#           They must be implemented for individual classifiers.
#         - Any other arguments are passed through to the classifier.
#
#         Example (with TS files in src directory):
#             python eval_classifier.py logistic
#                 --ts_dir .
#                 --data_dict transformed.json
#                 --static_file ../datasets/eeg/eeg_static.csv
#                 --validation_size 0.1
#                 --scoring roc_auc
#                 --max_iter 100
#                 --grid_C 0.1 1 10
#                 --threshold -1 0 1
#                 --threshold_scoring balanced_accuracy
#
# Output: TODO (currently prints results of grid search)

# TODO: allow specifying classifer settings implemented for grid search without
#       applying to grid search. For example, '--C 1' instead of '--grid_C 1'.
# TODO: clean up this file

import argparse
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from custom_classifiers import ThresholdClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import GridSearchCV, ShuffleSplit

# Parse pre-specified command line arguments

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

logistic_parser = subparsers.add_parser('logistic')
logistic_parser.set_defaults(clf=LogisticRegression,
                             default_clf_args={'solver': 'lbfgs'})
logistic_parser.add_argument('--grid_C', type=float, nargs='*', default=[1])

dtree_parser = subparsers.add_parser('dtree')
dtree_parser.set_defaults(clf=DecisionTreeClassifier, default_clf_args={})
dtree_parser.add_argument('--grid_max_depth', type=int, nargs='*', 
                          default=[None])

rforest_parser = subparsers.add_parser('rforest')
rforest_parser.set_defaults(clf=RandomForestClassifier, default_clf_args={})
rforest_parser.add_argument('--grid_n_estimators', type=int, nargs='*',
                            default=[10])
rforest_parser.add_argument('--grid_max_depth', type=int, nargs='*',
                            default=[None])

mlp_parser = subparsers.add_parser('mlp')
mlp_parser.set_defaults(clf=MLPClassifier, default_clf_args={})
# ast.literal_eval evaluates strings, converting to a tuple in this case
# (may need to put tuples in quotes for command line)
mlp_parser.add_argument('--grid_hidden_layer_sizes', type=ast.literal_eval,
                        nargs='*', default=[(100,)])
mlp_parser.add_argument('--grid_alpha', type=float, nargs='*', default=[0.0001])

all_subparsers = (logistic_parser, dtree_parser, rforest_parser, mlp_parser)
for p in all_subparsers:
    p.add_argument('--ts_dir', required=True)
    p.add_argument('--data_dict', required=True)
    p.add_argument('--static_file')
    p.add_argument('--validation_size', type=float, default=0.1)
    p.add_argument('--scoring')
    p.add_argument('--thresholds', type=float, nargs='*')
    p.add_argument('--threshold_scoring', default='balanced_accuracy')

args, unknown_args = parser.parse_known_args()
generic_args = ('ts_dir', 'data_dict', 'static_file', 'validation_size',
                 'scoring', 'clf', 'default_clf_args', 'thresholds',
                 'threshold_scoring')
# key[5:] strips the 'grid_' prefix from the argument
param_grid = {key[5:]: vars(args)[key] for key in vars(args)
              if key not in generic_args}

# Parse unspecified arguments to be passed through to the classifier

def auto_convert_str(x):
    try:
        x_float = float(x)
        x_int = int(x_float)
        if x_int == x_float:
            return x_int
        else:
            return x_float
    except ValueError:
        return x

passthrough_args = {}
for i in range(len(unknown_args)):
    arg = unknown_args[i]
    if arg.startswith('--'):
        val = unknown_args[i+1]
        passthrough_args[arg[2:]] = auto_convert_str(unknown_args[i+1])

# Import data

data_dict = json.load(open(args.data_dict))
train = pd.read_csv(args.ts_dir + '/train.csv')
test = pd.read_csv(args.ts_dir + '/test.csv')
if args.static_file:
    static = pd.read_csv(args.static_file)
    static_id_cols = [c['name'] for c in data_dict['fields']
                      if c['role'] == 'id' and c['name'] in static.columns]
    train = train.merge(static, on=static_id_cols)
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
x_test = test[feature_cols]
y_test = np.ravel(test[outcome_col])

# Grid search

clf = args.clf(**args.default_clf_args, **passthrough_args)
# Despite using GridSearchCV, this uses a single validation set.
# TODO: specify seed
splitter = ShuffleSplit(test_size=args.validation_size, n_splits=1)
grid = GridSearchCV(clf, param_grid, scoring=args.scoring, cv=splitter)
grid.fit(x_train, y_train)
print('\nGrid search results:')
for x in grid.cv_results_:
    if 'train' not in x:
        print(x, grid.cv_results_[x])

# Threshold search

if args.thresholds is not None:
    grid = GridSearchCV(ThresholdClassifier(grid.best_estimator_),
                        {'threshold': args.thresholds},
                        scoring=args.threshold_scoring, cv=splitter)
    grid.fit(x_train, y_train)
    print('\nThreshold search results:')
    for x in grid.cv_results_:
        if 'train' not in x:
            print(x, grid.cv_results_[x])


# Evaluation
y_test_pred = grid.predict(x_test)
y_test_pred_proba = grid.predict_proba(x_test)[:, 1]
accuracy = accuracy_score(y_test, y_test_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
avg_precision = average_precision_score(y_test, y_test_pred_proba)
auroc = roc_auc_score(y_test, y_test_pred_proba)
roc_fpr, roc_tpr, _ = roc_curve(y_test, y_test_pred_proba)
pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
confusion = confusion_matrix(y_test, y_test_pred)
print('\nAccuracy of best model: {:.3f}'.format(accuracy))
print('Balanced accuracy of best model: {:.3f}'.format(balanced_accuracy))
print('Average precision of best model: {:.3f}'.format(avg_precision))
print('AUROC of best model: {:.3f}'.format(auroc))
print('Confusion matrix of best model:\n', confusion)

# Plots

plt.plot(roc_fpr, roc_tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.savefig('roc.png')
plt.clf()
plt.plot(pr_recall, pr_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('pr.png')
