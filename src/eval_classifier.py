# eval_classifier.py

# Input:  Argument 1: a classifier type (currently only 'logistic' is supported)
#         Argument 2: a folder containing time-seriesfiles train.csv, valid.csv,
#                     and test.csv
#         Argument 3: data dictionary for the above files
#         --static_file: joins the indicated static file to the time-series data
#                        on all columns in the static data of role 'id'
#         - Other arguments are passed directly to the classifier (currently
#           only --C for logistic regression is supported)
# Output: TODO (currently prints the accuracy)

import argparse
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Parse command line arguments

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers()
logistic_parser = subparsers.add_parser('logistic')
logistic_parser.add_argument('--C', type=float)
logistic_parser.set_defaults(clf=LogisticRegression, solver='lbfgs', C=1)

for p in [logistic_parser]: # list of all supported classifiers
    p.add_argument('ts_folder')
    p.add_argument('data_dict') 
    p.add_argument('--static_file')

args = parser.parse_args()
clf_args = {key: vars(args)[key] for key in vars(args)
            if key not in ('ts_folder', 'data_dict', 'static_file', 'clf')}

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

# Train classifier (currently ignores validation data)

clf = args.clf(**clf_args)
clf.fit(x_train, y_train)
y_test_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_test_pred)
print('Accuracy: {:.3f}'.format(accuracy))

# Grid search (currently ignores validation data and prohibits customization)

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(clf, param_grid, cv=5) # other options?
grid.fit(x_train, y_train)
print('Grid search results:')
for x in grid.cv_results_:
    if 'train' not in x:
        print(x, grid.cv_results_[x])
y_test_pred = grid.predict(x_test)
accuracy = accuracy_score(y_test, y_test_pred)
print('Best accuracy: {:.3f}'.format(accuracy))