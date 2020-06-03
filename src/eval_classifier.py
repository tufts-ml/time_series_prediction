'''
Train and evaluate binary classifier

Produce a human-readable HTML report with performance plots and metrics

Usage
-----
```
python eval_classifier.py {classifier_name} --output_dir /path/ \
    {clf_specific_kwargs} {data_kwargs} {protocol_kwargs}
```

For detailed help message:
```
python eval_classifier.py {classifier_name} --help
```

Will produce

Examples
--------

TODO
----
* Save classifiers in reproducible format on disk (ONNX??)
* Add reporting for calibration (and perhaps adjustment to improve calibration)

'''

import argparse
import ast
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
from yattag import Doc

import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from custom_classifiers import ThresholdClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import GridSearchCV, ShuffleSplit

from split_dataset import Splitter
from utils_scoring import (THRESHOLD_SCORING_OPTIONS, calc_score_for_binary_predictions)

def get_sorted_list_of_kwargs_specific_to_group_parser(group_parser):
    keys = [a.option_strings[0].replace('--', '') for a in group_parser._group_actions]
    return [k for k in sorted(keys)]

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

default_json_dir = os.path.join(PROJECT_REPO_DIR, 'src', 'default_hyperparameters')
if not os.path.exists(default_json_dir):
    raise ValueError("Could not read default hyperparameters from file")

DEFAULT_SETTINGS_JSON_FILES = glob.glob(os.path.join(default_json_dir, '*.json'))
if len(DEFAULT_SETTINGS_JSON_FILES) == 0:
    raise ValueError("Could not read default hyperparameters from file")

try:
    TEMPLATE_HTML_PATH = os.path.join(PROJECT_REPO_DIR, 'src', 'template.html')
except KeyError:
    TEMPLATE_HTML_PATH = None


if __name__ == '__main__':

    # Parse pre-specified command line arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="clf_name", dest="clf_name")

    subparsers_by_name = dict()
    for json_file in DEFAULT_SETTINGS_JSON_FILES:
        with open(json_file, 'r') as f:
            defaults = json.load(f)
        clf_name = os.path.basename(json_file).split('.')[0]
        clf_parser = subparsers.add_parser(clf_name)

        default_group = clf_parser.add_argument_group('fixed_clf_settings')
        hyperparam_group = clf_parser.add_argument_group('hyperparameters')
        for key, val in defaults.items():
            if key.count('constructor'):
                assert val.count(' ') == 0
                assert val.startswith('sklearn')
                for ii, name in enumerate(val.split('.')):
                    if ii == 0:
                        mod = globals().get(name)
                    else:
                        mod = getattr(mod, name)
                clf_parser.add_argument('--clf_constructor', default=mod)
            elif isinstance(val, list):
                if key.startswith('grid_'):
                    hyperparam_group.add_argument("--%s" % key, default=val, type=type(val[0]), nargs='*')
                else:
                    default_group.add_argument("--%s" % key, default=val, type=type(val[0]), nargs='*')
            else:
                has_simple_type = isinstance(val, str) or isinstance(val, int) or isinstance(val, float)
                assert has_simple_type
                if key.startswith('grid_'):
                    hyperparam_group.add_argument("--%s" % key, default=val, type=type(val))
                else:
                    default_group.add_argument("--%s" % key, default=val, type=type(val))
        subparsers_by_name[clf_name] = clf_parser

    '''
    logistic_parser = subparsers.add_parser('logistic')
    logistic_parser.set_defaults(clf=LogisticRegression,
                                 default_clf_args={'solver': 'lbfgs',
                                                   'multi_class': 'auto'})
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
    '''

    for p in subparsers_by_name.values():

        data_group = p.add_argument_group('data')
        data_group.add_argument('--train_csv_files', type=str, required=True)
        data_group.add_argument('--test_csv_files', type=str, required=True)
        data_group.add_argument('--data_dict_files', type=str, required=True)
        data_group.add_argument('--output_dir', default='./html/', type=str, required=False)

        protocol_group = p.add_argument_group('protocol')
        protocol_group.add_argument('--outcome_col_name', type=str, required=False)
        protocol_group.add_argument('--validation_size', type=float, default=0.1)
        protocol_group.add_argument('--key_cols_to_group_when_splitting', type=str,
            default=None, nargs='*')
        protocol_group.add_argument('--scoring', type=str, default='roc_auc_score')
        protocol_group.add_argument('--random_seed', type=int, default=8675309)
        protocol_group.add_argument('--n_splits', type=int, default=1)
        protocol_group.add_argument('--threshold_scoring', type=str,
            default=None, choices=[None, 'None'] + THRESHOLD_SCORING_OPTIONS)
        #p.add_argument('-a-ts_dir', required=True)
        #p.add_argument('--data_dict', required=True)
        #p.add_argument('--static_files', nargs='*')

    args, unknown_args = parser.parse_known_args()
    fig_dir = os.path.abspath(args.output_dir)

    # key[5:] strips the 'grid_' prefix from the argument
    param_grid = {key[5:]: vars(args)[key] for key in vars(args)
                  if key.startswith('grid_')}
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

    # Import data

    feature_cols = []
    outcome_cols = []

    df_by_split = dict()
    for split_name, csv_files in [('train', args.train_csv_files.split(',')), ('test', args.test_csv_files.split(','))]:
        cur_df = None
        for csv_file, data_dict_file in zip(csv_files, args.data_dict_files.split(',')):
            with open(data_dict_file, 'r') as f:
                data_fields = json.load(f)['schema']['fields']
                key_cols = [c['name'] for c in data_fields if c['role'] in ('key', 'id')]

                feature_cols.extend([
                    c['name'] for c in data_fields if (
                        c['role'].lower() in ('feature', 'covariate', 'measurement')
                        and c['name'] not in feature_cols)])

                outcome_cols.extend([
                    c['name'] for c in data_fields if (
                        c['role'].lower() in ('output', 'outcome')
                        and c['name'] not in feature_cols)])

            # TODO use json data dict to load specific columns as desired types
            more_df =  pd.read_csv(csv_file)
            if cur_df is None:
                cur_df = more_df
            else:
                cur_df = cur_df.merge(more_df, on=key_cols)
        df_by_split[split_name] = cur_df

    '''
    data_dict = json.load(open(args.data_dict))
    train = pd.read_csv(args.ts_dir + '/train.csv')
    test = pd.read_csv(args.ts_dir + '/test.csv')
    if args.static_files:
        for f in args.static_files:
            static = pd.read_csv(f)
            join_cols = [c['name'] for c in data_dict['fields']
                         if c['role'] == 'id' and c['name'] in static.columns
                            and c['name'] in train.columns]
            train = train.merge(static, on=join_cols)
            test = test.merge(static, on=join_cols)


    feature_cols = [c['name'] for c in data_dict['fields']
                    if c['role'] == 'feature' and c['name'] in train]
    outcome_col = [c['name'] for c in data_dict['fields']
                   if c['role'] == 'outcome' and c['name'] in train]
    '''

    outcome_col_name = args.outcome_col_name
    if outcome_col_name is None:
        if len(outcome_cols) > 1:
            raise Exception('Data has multiple outcome column, need to select one via --outcome_col_name')
        elif len(outcome_cols) == 0:
            raise Exception("Data has no outcome columns. Need to label at least one with role='outcome'")
        outcome_col_name = outcome_cols[0]

    if outcome_col_name not in outcome_cols:
        raise Exception("Selected --outcome_col_name=%s not labeled in data_dict with role='outcome'" % (
            outcome_col_name))

    # Prepare data for classification
    x_train = df_by_split['train'][feature_cols].values
    y_train = np.ravel(df_by_split['train'][outcome_col_name])

    x_test = df_by_split['test'][feature_cols].values
    y_test = np.ravel(df_by_split['test'][outcome_col_name])
    is_multiclass = len(np.unique(y_train)) > 2



    fixed_args = {}
    fixed_group = None
    for g in subparsers_by_name[args.clf_name]._action_groups:
        if g.title.count('fixed'):
            fixed_group = g
            break
    for key in get_sorted_list_of_kwargs_specific_to_group_parser(fixed_group):
        fixed_args[key] = vars(args)[key]

    passthrough_args = {}
    for i in range(len(unknown_args)):
        arg = unknown_args[i]
        if arg.startswith('--'):
            val = unknown_args[i+1]
            passthrough_args[arg[2:]] = auto_convert_str(val)

    # Create classifier object
    clf = args.clf_constructor(**fixed_args, **passthrough_args)

    # Perform hyper_searcher search
    splitter = Splitter(size=args.validation_size, random_state=args.random_seed, n_splits=args.n_splits, cols_to_group=args.key_cols_to_group_when_splitting)
    hyper_searcher = GridSearchCV(clf, param_grid,
        scoring=args.scoring, cv=splitter, refit=True, return_train_score=True)
    key_train = splitter.make_groups_from_df(df_by_split['train'][key_cols])
    hyper_searcher.fit(x_train, y_train, groups=key_train)
    
    # Pretty tables for results of hyper_searcher search
    cv_perf_df = pd.DataFrame(hyper_searcher.cv_results_)
    tr_split_keys = ['mean_train_score'] + ['split%d_train_score' % a for a in range(args.n_splits)]
    te_split_keys = ['mean_test_score'] + ['split%d_test_score' % a for a in range(args.n_splits)]
    cv_tr_perf_df = cv_perf_df[['params'] + tr_split_keys].copy()
    cv_te_perf_df = cv_perf_df[['params'] + te_split_keys].copy()

    # Threshold search
    # TODO make cast wider net at nearby settings to the best estimator??
    if str(args.threshold_scoring) != 'None':
        # hyper_searcher search on validation over possible threshold values
        # Make sure all candidates at least provide
        # one instance of each class (positive and negative)
        yproba_class1_vals = list()
        for tr_inds, va_inds in splitter.split(x_train, groups=key_train):
            x_valid = x_train[va_inds]
            yproba_valid = hyper_searcher.best_estimator_.predict_proba(x_valid)[:,1]
            yproba_class1_vals.extend(yproba_valid)

        unique_yproba_vals = np.unique(yproba_class1_vals)
        if unique_yproba_vals.shape[0] == 1:
            nontrivial_thr_vals = unique_yproba_vals
        else:
            # Try all thr values that would give at least one pos and one neg decision
            nontrivial_thr_vals = np.unique(unique_yproba_vals)[1:-1]

        if nontrivial_thr_vals.size > 100:
            # Too many for possible thr values for typical compute power
            # Cover the space of typical computed values well
            # But also include some extreme values
            dense_thr_grid = np.linspace(
                np.percentile(nontrivial_thr_vals, 5),
                np.percentile(nontrivial_thr_vals, 95),
                90)
            extreme_thr_grid = np.linspace(
                nontrivial_thr_vals[0],
                nontrivial_thr_vals[-1],
                10)
            thr_grid = np.unique(np.hstack([extreme_thr_grid, dense_thr_grid]))
        else:
            # Seems feasible to just look at all possible thresholds
            # that give distinct operating points.
            thr_grid = nontrivial_thr_vals

        print("Searching thresholds...")
        if thr_grid.shape[0] > 3:
            print("thr_grid = %.4f, %.4f, %.4f ... %.4f, %.4f" % (
                thr_grid[0], thr_grid[1], thr_grid[2], thr_grid[-2], thr_grid[-1]))

        ## TODO find better way to do this fast
        # So we dont need to call fit at each thr value
        score_grid_SG = np.zeros((splitter.n_splits, thr_grid.size))
        for ss, (tr_inds, va_inds) in enumerate(
                splitter.split(x_train, y_train, groups=key_train)):
            x_tr = x_train[tr_inds].copy()
            y_tr = y_train[tr_inds].copy()
            x_va = x_train[va_inds]
            y_va = y_train[va_inds]

            tmp_clf = ThresholdClassifier(hyper_searcher.best_estimator_)
            tmp_clf.fit(x_tr, y_tr)

            for gg, thr in enumerate(thr_grid):
                tmp_clf = tmp_clf.set_params(threshold=thr)
                yhat = tmp_clf.predict(x_va)
                score_grid_SG[ss, gg] = calc_score_for_binary_predictions(y_va, yhat, scoring=args.threshold_scoring)
        ## TODO read off best average score
        avg_score_G = np.mean(score_grid_SG, axis=0)
        gg = np.argmax(avg_score_G)

        # Keep best scoring estimator 
        best_thr = thr_grid[gg]
        print("Chosen Threshold: %.4f" % best_thr)
        best_clf = ThresholdClassifier(hyper_searcher.best_estimator_, threshold=best_thr)
    else:
        best_clf = hyper_searcher.best_estimator_

    # Evaluation
    row_dict_list = list()
    extra_list = list()
    for split_name, x, y in [
            ('train', x_train, y_train),
            ('test', x_test, y_test)]:
        row_dict = dict(split_name=split_name, n_examples=x.shape[0], n_labels_positive=np.sum(y))
        row_dict['frac_labels_positive'] = np.sum(y) / x.shape[0]

        y_pred = best_clf.predict(x)
        y_pred_proba = best_clf.predict_proba(x)[:, 1]

        confusion_arr = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(confusion_arr, columns=[0,1], index=[0,1])
        cm_df.columns.name = 'Predicted label'
        cm_df.index.name = 'True label'
        
        accuracy = accuracy_score(y, y_pred)
        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        log2_loss = log_loss(y, y_pred_proba, normalize=True) / np.log(2)
        row_dict.update(dict(confusion_html=cm_df.to_html(), cross_entropy_base2=log2_loss, accuracy=accuracy, balanced_accuracy=balanced_accuracy))
        if not is_multiclass:
            f1 = f1_score(y, y_pred)

            avg_precision = average_precision_score(y, y_pred_proba)
            roc_auc = roc_auc_score(y, y_pred_proba)
            row_dict.update(dict(f1_score=f1, average_precision=avg_precision, AUROC=roc_auc))

            # Plots
            roc_fpr, roc_tpr, _ = roc_curve(y, y_pred_proba)
            ax = plt.gca()
            ax.plot(roc_fpr, roc_tpr)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.savefig(os.path.join(fig_dir, '{split_name}_roc_curve.png'.format(split_name=split_name)))
            plt.clf()

            pr_precision, pr_recall, _ = precision_recall_curve(y, y_pred_proba)
            ax = plt.gca()
            ax.plot(pr_recall, pr_precision)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.savefig(os.path.join(fig_dir, '{split_name}_pr_curve.png'.format(split_name=split_name)))
            plt.clf()

        row_dict_list.append(row_dict)

    perf_df = pd.DataFrame(row_dict_list)
    perf_df = perf_df.set_index('split_name')



    # Set up HTML report
    try:
        os.mkdir(fig_dir)
    except OSError:
        pass
    os.chdir(fig_dir)

    doc, tag, text = Doc().tagtext()
    pd.set_option('precision', 4)

    with tag('html'):
        if os.path.exists(TEMPLATE_HTML_PATH):
            with open(TEMPLATE_HTML_PATH, 'r') as f:
                for line in f.readlines():
                    doc.asis(line)

        with tag('div', klass="container-fluid text-center"):
            with tag('div', klass='row content'):
                with tag('div', klass="col-sm-1 sidenav"):
                    text("")

                with tag('div', klass="col-sm-10 text-left"):
                    with tag('h3'):
                        with tag('a', name='settings-hyperparameter'):
                            text('Settings: Hyperparameters to Tune')
                    with tag('pre'):
                        hyper_group = None
                        for g in subparsers_by_name[args.clf_name]._action_groups:
                            if g.title.count('hyper'):
                                hyper_group = g
                                break
                        for x in get_sorted_list_of_kwargs_specific_to_group_parser(hyper_group):
                            text(x, ': ', str(vars(args)[x]), '\n')

                    with tag('h3'):
                        with tag('a', name='settings-protocol'):
                            text('Settings: Protocol')
                    with tag('pre'):
                        for x in get_sorted_list_of_kwargs_specific_to_group_parser(protocol_group):
                            text(x, ': ', str(vars(args)[x]), '\n')

                    with tag('h3'):
                        with tag('a', name='settings-data'):
                            text('Settings: Data')
                    with tag('pre'):
                        for x in get_sorted_list_of_kwargs_specific_to_group_parser(data_group):
                            text(x, ': ', str(vars(args)[x]), '\n')

                    with tag('h3'):
                        with tag('a', name='results-hyper_searcher-search'):
                            text('hyper_searcher Search results')
                    with tag('h4'):
                        text('Train Scores across splits')

                    doc.asis(pd.DataFrame(cv_tr_perf_df).to_html())

                    with tag('h4'):
                        text('Heldout Scores across splits')
                    doc.asis(pd.DataFrame(cv_te_perf_df).to_html())

                    with tag('h3'):
                        text('Hyperparameters of best model')
                    with tag('pre'):
                        text(str(best_clf))

                    with tag('h3'):
                        with tag('a', name='results-data-summary'):
                            text('Input Data Summary')
                    doc.asis(perf_df[['n_examples', 'n_labels_positive', 'frac_labels_positive']].to_html())

                    with tag('h3'):
                        with tag('a', name='results-performance-plots'):
                            text('Performance Plots')

                    with tag('table'):
                        with tag('tr'):
                            with tag('th', **{'text-align':'center'}):
                                text('Train')
                            with tag('th', **{'text-align':'center'}):
                                text('Test')
                        with tag('tr'):
                            with tag('td', align='center'):
                                doc.stag('img', src=os.path.join(fig_dir, 'train_roc_curve.png'), width=400)
                            with tag('td', align='center'):
                                doc.stag('img', src=os.path.join(fig_dir, 'test_roc_curve.png'), width=400)
                        with tag('tr'):
                            with tag('td', align='center'):
                                doc.stag('img', src=os.path.join(fig_dir, 'train_pr_curve.png'), width=400)
                            with tag('td', align='center'):
                                doc.stag('img', src=os.path.join(fig_dir, 'test_pr_curve.png'), width=400)
                        with tag('tr'):
                            with tag('td', align='center'):
                                doc.asis(str(perf_df.iloc[0][['confusion_html']].values[0]).replace('&lt;', '<').replace('&gt;', '>').replace('\\n', ''))

                            with tag('td', align='center'):
                                doc.asis(str(perf_df.iloc[1][['confusion_html']].values[0]).replace('&lt;', '<').replace('&gt;', '>').replace('\\n', ''))
                   
                    with tag('h3'):
                        with tag('a', name='results-performance-metrics-proba'):
                            text('Performance Metrics using Probabilities')
                    doc.asis(perf_df[['AUROC', 'average_precision', 'cross_entropy_base2']].to_html())

                    with tag('h3'):
                        with tag('a', name='results-performance-metrics-thresh'):
                            text('Performance Metrics using Thresholded Decisions')
                    doc.asis(perf_df[['balanced_accuracy', 'accuracy', 'f1_score']].to_html())

                # Add dark region on right side
                with tag('div', klass="col-sm-1 sidenav"):
                    text("")

    with open('report.html', 'w') as f:
        f.write(doc.getvalue())
