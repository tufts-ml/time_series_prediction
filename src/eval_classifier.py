'''
Train and evaluate binary classifier
Produce a human-readable HTML report with performance plots and metrics
Usage
-----
```
python eval_classifier.py {classifier_name} \
    --output_dir /path/ \
    {clf_specific_kwargs} \
    {data_kwargs} \
    {protocol_kwargs}
```
For detailed help message:
```
python eval_classifier.py {classifier_name} --help
```
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
from joblib import dump

import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sys

from custom_classifiers import ThresholdClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from split_dataset import Splitter

from utils_scoring import (
    HYPERSEARCH_SCORING_OPTIONS, THRESHOLD_SCORING_OPTIONS,
    HARD_DECISION_SCORERS,
    calc_cross_entropy_base2_score,
    calc_score_for_binary_predictions)
from utils_calibration import plot_binary_clf_calibration_curve_and_histograms

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))


def get_sorted_list_of_kwargs_specific_to_group_parser(group_parser):
    keys = [a.option_strings[0].replace('--', '') for a in group_parser._group_actions]
    return [k for k in sorted(keys)]



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

FIG_KWARGS = dict(
    figsize=(4, 4),
    tight_layout=True)

def load_data_dict_json(data_dict_file):
    with open(data_dict_file, 'r') as f:
        data_dict = json.load(f)
        try:
            data_dict['fields'] = data_dict['schema']['fields']
        except KeyError:
            pass
    return data_dict


if __name__ == '__main__':

    # Parse pre-specified command line arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="clf_name", dest="clf_name")
    subparsers_by_name = dict()
    
    
    # Create classifier-specific options for the parser
    # Read in 'defaults' from json files, allow overriding with kwarg args
    for json_file in DEFAULT_SETTINGS_JSON_FILES:
        clf_name = os.path.basename(json_file).split('.')[0]
        clf_parser = subparsers.add_parser(clf_name)

        # Hyperparameters are tuned on validation sets or via CV
        hyperparam_group = clf_parser.add_argument_group('hyperparameters')

        # Specialized functions that filter out irrelevant hyperparameter candidates for this dataset
        # These functions take the form:
        # filter(val, n_examples, n_features) -> acceptable_value
        filter_func_group = clf_parser.add_argument_group('filter_funcs_that_produce_acceptable_hypers')

        # Fixed settings are recommended defaults that do not need tuning
        default_group = clf_parser.add_argument_group('fixed_clf_settings')

        # Read in defaults from JSON file
        with open(json_file, 'r') as f:
            defaults = json.load(f)
        
        
        # Setup parser options using the contents of JSON file
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
                if key.startswith('grid__'):
                    hyperparam_group.add_argument("--%s" % key, default=val, type=type(val[0]), nargs='*')
                else:
                    default_group.add_argument("--%s" % key, default=val, type=type(val[0]), nargs='*')
            else:
                has_simple_type = isinstance(val, str) or isinstance(val, int) or isinstance(val, float)
                assert has_simple_type
                if key.startswith('grid__'):
                    hyperparam_group.add_argument("--%s" % key, default=val, type=type(val))
                elif key.startswith('filter__') or key == 'simplicity_score_func':
                    filter_func_group.add_argument("--%s" % key, default=val, type=type(val))
                else:
                    default_group.add_argument("--%s" % key, default=val, type=type(val))
        subparsers_by_name[clf_name] = clf_parser

    # Create dataset-specific and experimental design options for the parser
    for p in subparsers_by_name.values():
        data_group = p.add_argument_group('data')
        data_group.add_argument('--train_csv_files', type=str, required=True)
        data_group.add_argument('--test_csv_files', type=str, required=True)
        data_group.add_argument('--data_dict_files', type=str, required=True)
        data_group.add_argument('--output_dir', default='./html/', type=str, required=False)
        data_group.add_argument('--merge_x_y', default=True,
                                type=lambda x: (str(x).lower() == 'true'), required=False,
                                help='If True, features and outcomes are merged on id columns, if false, they get concatenated.')
        protocol_group = p.add_argument_group('protocol')
        protocol_group.add_argument('--outcome_col_name', type=str, required=False)
        protocol_group.add_argument('--validation_size', type=float, default=0.1)
        protocol_group.add_argument('--standardize_numeric_features', type=bool, default=True)
        protocol_group.add_argument('--key_cols_to_group_when_splitting', type=str,
            default=None, nargs='*')
        protocol_group.add_argument('--scoring', type=str, default='roc_auc_score+0.001*cross_entropy_base2_score')
        protocol_group.add_argument('--random_seed', type=int, default=8675309)
        protocol_group.add_argument('--n_splits', type=int, default=1)
        protocol_group.add_argument('--max_recall_at_fixed_precision_param', type=float, default=0.0)
        protocol_group.add_argument('--threshold_scoring', type=str,
            default=None, choices=[None, 'None'] + THRESHOLD_SCORING_OPTIONS)

    # Parse known arguments from stdin
    args, unknown_args = parser.parse_known_args()
    argdict = vars(args)


    # Prepare output directory
    fig_dir = os.path.abspath(args.output_dir)
    try:
        os.mkdir(fig_dir)
    except OSError:
        pass

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
    print('Reading train-test data...')
    feature_cols = []
    outcome_cols = []
    info_per_feature_col = dict()
    feature_ranges = []

    df_by_split = dict()
    for split_name, csv_files in [
            ('train', args.train_csv_files.split(',')),
            ('test', args.test_csv_files.split(','))]:
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
                
                feature_ranges.extend([
                        [c['constraints']['minimum'], c['constraints']['maximum']] for c in data_fields if (
                        c['role'].lower() in ('feature', 'covariate', 'measurement')
                        and c['name'] in feature_cols)])

                for info_d in data_fields:
                    try:
                        i = feature_cols.index(info_d['name'])
                        info_per_feature_col[info_d['name']] = info_d
                    except ValueError:
                        # skip fields not found
                        pass

            # TODO use json data dict to load specific columns as desired types
            more_df =  pd.read_csv(csv_file)
            if cur_df is None:
                cur_df = more_df
            else:
                if args.merge_x_y:
                    cur_df = cur_df.merge(more_df, on=key_cols)
                else:
                    cur_df = pd.concat([cur_df, more_df], axis=1)
                    cur_df = cur_df.loc[:,~cur_df.columns.duplicated()]
        
        df_by_split[split_name] = cur_df

    # for consistency
    feature_cols.sort()
    
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
    x_train = df_by_split['train'][feature_cols].values.astype(np.float32)
    y_train = np.ravel(df_by_split['train'][outcome_col_name])
    
    x_test = df_by_split['test'][feature_cols].values.astype(np.float32)
    y_test = np.ravel(df_by_split['test'][outcome_col_name])
    is_multiclass = len(np.unique(y_train)) > 2
    
    fixed_args = {}
    fixed_group = None
    for g in subparsers_by_name[args.clf_name]._action_groups:
        if g.title.count('fixed'):
            fixed_group = g
            break
    for key in get_sorted_list_of_kwargs_specific_to_group_parser(fixed_group):
        val = vars(args)[key]
        if isinstance(val, str):
            if val.lower() == 'none':
                val = None

        if isinstance(val, str):
            if 'lambda x' in val:
                continue
        fixed_args[key] = val

    passthrough_args = {}
    for i in range(len(unknown_args)):
        arg = unknown_args[i]
        if arg.startswith('--'):
            val = unknown_args[i+1]
            passthrough_args[arg[2:]] = auto_convert_str(val)


    # Prepare hyperparameter search
    # -----------------------------
    # Read in the hyperparameter candiate values grid from stdin
    # Then apply relevant filters to create acceptable candidate values for this dataset
    raw_param_grid = dict()
    for key, val  in argdict.items():
        if key.startswith('grid__'):
            # key[5:] strips the 'grid__' prefix from the argument
            raw_param_grid[key[6:]] = val
        elif key.startswith('filter__'):
            raw_param_grid[key] = val
    n_examples = int(np.ceil(x_train.shape[0] * (1 - args.validation_size)))
    n_features = x_train.shape[1]
    pipeline_param_grid_dict = dict()

    # Perform hyper_searcher search
    n_examples = int(np.ceil(x_train.shape[0] * (1 - args.validation_size)))
    n_features = x_train.shape[1]
    print('training on %s examples and %s features...'%(x_train.shape[0], x_train.shape[1]))

    for key, grid in raw_param_grid.items():
        filter_key = 'filter__' + key
        if filter_key in argdict:
            filter_func = eval(argdict[filter_key])
            filtered_grid = np.unique([filter_func(g, n_examples, n_features) for g in grid]).tolist()
        else:
            filtered_grid = np.unique(grid).tolist()

        filtering_successful = 'lambda x' not in str(filtered_grid[0])
        if filtering_successful:
            if len(filtered_grid) == 0:
                raise Warning("Candidate grid has zero values for parameter: %s")
            elif len(filtered_grid) == 1:
                fixed_args[key] = filtered_grid[0]
                raise Warning("Skipping parameter %s with only one grid value")
            else:
                # Safe to use filtered grid for this parameter
                pipeline_param_grid_dict['classifier__' + key] = filtered_grid
    
    if 'hidden_layer_sizes' in fixed_args.keys():
        fixed_args['hidden_layer_sizes'] = eval(fixed_args['hidden_layer_sizes'])
    
    # Create classifier pipeline
    # --------------------------
    clf = args.clf_constructor(
        random_state=int(args.random_seed), **fixed_args, **passthrough_args)    
    # Perform hyper_searcher search
    splitter = Splitter(
        size=args.validation_size, random_state=args.random_seed,
        n_splits=args.n_splits,
        cols_to_group=args.key_cols_to_group_when_splitting)
    step_list = list()
    if args.standardize_numeric_features:
        numeric_feature_ids = [
            i for i, col_name in enumerate(feature_cols)
                if info_per_feature_col[col_name]['type'] == 'numeric']
        std_scaler_preprocessor = ColumnTransformer(
            [("std_scaler", StandardScaler(), numeric_feature_ids)],
            remainder='passthrough')
        step_list.append(('standardize_numeric_features', std_scaler_preprocessor))
    step_list.append(('classifier', clf))
    prediction_pipeline = Pipeline(step_list)
    

    # Establish train/test splits
    # ---------------------------
    # Create object that knows how to divide data into train/test/valid
    splitter = Splitter(
        size=args.validation_size, random_state=args.random_seed,
        n_splits=args.n_splits, cols_to_group=args.key_cols_to_group_when_splitting)
    # Assign training instances to splits by provided keys
    key_train = splitter.make_groups_from_df(df_by_split['train'][key_cols])

    
    scoring_dict = dict()
    scoring_weights_dict = dict()
    for score_expr in args.scoring.split("+"):
        if score_expr.count("*"):
            score_wt, score_func_name = score_expr.split("*")
            score_wt = float(score_wt)
        else:
            score_wt = 1.0
            score_func_name = score_expr
        if score_func_name in HYPERSEARCH_SCORING_OPTIONS.keys():
            scoring_dict[score_func_name] = HYPERSEARCH_SCORING_OPTIONS[score_func_name]
        elif score_func_name in HARD_DECISION_SCORERS.keys():
            scoring_dict[score_func_name] = HARD_DECISION_SCORERS[score_func_name]
        scoring_weights_dict[score_func_name] = score_wt

    
    
    # Run expensive training + selection
    # ----------------------------------
    hyper_searcher = GridSearchCV(
        prediction_pipeline, pipeline_param_grid_dict,
        scoring=scoring_dict, cv=splitter, refit=False,
        return_train_score=True, verbose=5, n_jobs=-1)
    hyper_searcher.fit(x_train, y_train, groups=key_train)

    # Pretty tables for results of hyper_searcher search
    cv_perf_df = pd.DataFrame(hyper_searcher.cv_results_)    
    for split in ['train', 'test']:
        for prefix in ['mean'] + ['split%d' % a for a in range(args.n_splits)]:
            src_colname_pattern = '%s_%s_{sname}' % (prefix, split)
            target_colname = '%s_%s_score' % (prefix, split)
            target_arr = np.zeros(cv_perf_df.shape[0])
            for sname, wt in scoring_weights_dict.items():
                src_colname = src_colname_pattern.replace("{sname}", sname)
                target_arr += wt * cv_perf_df[src_colname]
            cv_perf_df[target_colname] = target_arr
    
    tr_split_keys = ['params', 'mean_train_score'] + [
        'split%d_train_score' % a for a in range(args.n_splits)]
    te_split_keys = ['params', 'mean_test_score'] + [
        'split%d_test_score' % a for a in range(args.n_splits)]
    cv_tr_perf_df = cv_perf_df[tr_split_keys].copy()
    cv_te_perf_df = cv_perf_df[te_split_keys].copy()
    cv_tr_perf_df.rename(
        dict(zip(
            tr_split_keys,
            [a.replace('_train', '') for a in tr_split_keys])),
        axis='columns',
        inplace=True)
    cv_te_perf_df.rename(
        dict(zip(
            te_split_keys,
            [a.replace('_test', '') for a in te_split_keys])),
        axis='columns',
        inplace=True)

    if False:
        # TODO release this feature
        # If any simpler model achieved a score within the 'best' model's range,
        # use that simpler model instead.

        # Build the simplicity score function from its str specification
        calc_simplicity_score = eval(args.simplicity_score_func.replace(
            "d['", "d['classifier__"))
        S = cv_te_perf_df.shape[0]
        simplicity_S = np.zeros(S)
        for ss in range(S):
            param_dict = cv_te_perf_df.loc[ss, 'params']
            simplicity_S[ss] = calc_simplicity_score(param_dict, n_examples, n_features)
        cv_te_perf_df['simplicity_score'] = simplicity_S
    
        score_on_worst_split_S = np.min(cv_te_perf_df.values[:,2:], axis=1)
        cv_te_perf_df['worst_split_+_simplicity_score'] = score_on_worst_split_S + simplicity_S

        mean_test_score_S = cv_te_perf_df['mean_score'].values
        best_row_id = mean_test_score_S.argmax()
        bmask_S = mean_test_score_S >= score_on_worst_split_S[best_row_id]
        total_score_S = -np.inf + np.zeros(S)
        total_score_S[bmask_S] = (
            mean_test_score_S[bmask_S] + simplicity_S[bmask_S])
        cv_te_perf_df['total_score'] = total_score_S
    else:
        mean_test_score_S = cv_te_perf_df['mean_score'].values
        cv_te_perf_df['total_score'] = mean_test_score_S.copy()
    
    # Select the single best set of hyperparameters
    if args.max_recall_at_fixed_precision_param>0:
        # if the user wants to maximize recall at fixed precision, keep only the indices that achieve fixed precision and choose the best hyperparameter as the max recall among those achieving fixed precision
        alpha = args.max_recall_at_fixed_precision_param
        default_alpha=0.2
        
        # if none of the indices achieve fixed precision, then set fixed precision to default alpha
        print('Choosing hyperpapram with maximum precision on validation set..')
        best_row_id = cv_perf_df.mean_test_precision_score.argmax()
        best_param_dict = cv_perf_df.loc[best_row_id, 'params']
    else:
        best_row_id = cv_te_perf_df['total_score'].argmax()
        best_param_dict = cv_te_perf_df.loc[best_row_id, 'params']
    hyper_searcher.best_estimator_ = hyper_searcher.estimator
    hyper_searcher.best_estimator_.set_params(**best_param_dict)

    # Refit the best estimator with these hypers!
    hyper_searcher.best_estimator_.fit(x_train, y_train)
    
    # Threshold search
    # TODO cast wider net at nearby settings to the best estimator??
    if str(args.threshold_scoring) != 'None':
        # hyper_searcher search on validation over possible threshold values
        # Make sure all candidates at least provide
        # one instance of each class (positive and negative)
        yproba_class1_vals = list()
        for tr_inds, va_inds in splitter.split(x_train, groups=key_train):
            x_valid = x_train[va_inds]
            yproba_valid = hyper_searcher.best_estimator_.predict_proba(x_valid)[:,1]
            yproba_class1_vals.extend(yproba_valid)

        # Try all thr values that would give at least one pos and one neg decision
        nontrivial_thr_vals = np.unique(yproba_class1_vals)

        if nontrivial_thr_vals.size > 100:
            # Too many for possible thr values for typical compute power
            # Cover the space of typical computed values well
            # But also include some extreme values
            dense_thr_grid = np.linspace(
                np.percentile(nontrivial_thr_vals, 1),
                np.percentile(nontrivial_thr_vals, 99),
                100)
            extreme_thr_grid = np.linspace(
                nontrivial_thr_vals[0],
                nontrivial_thr_vals[-1],
                10)
            thr_grid = np.unique(np.hstack([
                np.min(extreme_thr_grid) - 0.0001,
                np.max(extreme_thr_grid) + 0.0001,
                extreme_thr_grid, dense_thr_grid]))
        else:
            # Seems feasible to just look at all possible thresholds
            # that give distinct operating points.
            extreme_thr_grid = np.linspace(
                nontrivial_thr_vals[0],
                nontrivial_thr_vals[-1],
                5)
            thr_grid = np.unique(np.hstack([
                np.min(extreme_thr_grid) - 0.0001,
                np.max(extreme_thr_grid) + 0.0001,
                extreme_thr_grid, nontrivial_thr_vals]))

        print("Searching thresholds...")
        if thr_grid.shape[0] >= 5:
            print("thr_grid = %.4f, %.4f, %.4f ... %.4f, %.4f" % (
                thr_grid[0], thr_grid[1], thr_grid[2], thr_grid[-2], thr_grid[-1]))
        else:
            print("thr_grid = %s" % (
                ', '.join(['%.4f' % a for a in thr_grid])))
        
        if args.max_recall_at_fixed_precision_param>0:
            # Search thresholds that achieve atleast fixed precsion and choose threshold as the one that maximizes recall among them
            precision_score_grid_SG = np.zeros((splitter.n_splits, thr_grid.size))
            recall_score_grid_SG = np.zeros((splitter.n_splits, thr_grid.size))
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
                    precision_score_grid_SG[ss, gg] = calc_score_for_binary_predictions(
                        y_va, yhat, scoring='precision_score')
                    recall_score_grid_SG[ss, gg] = calc_score_for_binary_predictions(
                        y_va, yhat, scoring='recall_score')
            
            
            avg_precision_score_G = np.mean(precision_score_grid_SG, axis=0)
            avg_recall_score_G = np.mean(recall_score_grid_SG, axis=0)
            
            # keep only thresholds that match atleast the specified precision
            keep_inds = avg_precision_score_G>=alpha
            if keep_inds.sum()==0:
                print('Could not find threshold achieving precision of %.3f'%alpha)
                print('Dropping fixed precision to %.3f'%default_alpha)
                keep_inds = avg_precision_score_G>=default_alpha
                if keep_inds.sum()==0:
                    keep_inds = (avg_precision_score_G >= np.percentile(avg_precision_score_G, 75)) 
            

            avg_precision_score_G = avg_precision_score_G[keep_inds]
            avg_recall_score_G = avg_recall_score_G[keep_inds]
            thr_grid = thr_grid[keep_inds]
            gg = np.argmax(avg_recall_score_G)
            best_thr = thr_grid[gg]
            thr_perf_df = pd.DataFrame(np.vstack([
                    thr_grid[np.newaxis,:],
                    avg_precision_score_G[np.newaxis,:],
                    avg_recall_score_G[np.newaxis,:]]).T,
                columns=['thr', 'precision_score', 'recall_score'])          

        else:
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
                    score_grid_SG[ss, gg] = calc_score_for_binary_predictions(
                        y_va, yhat, scoring=args.threshold_scoring)

            avg_score_G = np.mean(score_grid_SG, axis=0)

            # Do a 2nd order quadratic fit to the scores
            # Focusing weight on points near the maximizer
            # This gives us a "smoothed" function mapping thresholds to scores
            # Avoids issues if scores are "wiggly" and don't want to rely on max
            # Also will do right thing when there are many thresholds that work
            # Smoothed scores guarantees we get maximizer in middle of that range
            weights_G = np.exp(-10.0 * np.abs(avg_score_G - np.max(avg_score_G)))
            poly_coefs = np.polyfit(thr_grid, avg_score_G, 2, w=weights_G)
            smoothed_score_G = np.polyval(poly_coefs, thr_grid)

            # Keep best scoring estimator 
            gg = np.argmax(smoothed_score_G)
            best_thr = thr_grid[gg]
            thr_perf_df = pd.DataFrame(np.vstack([
                    thr_grid[np.newaxis,:],
                    avg_score_G[np.newaxis,:],
                    smoothed_score_G[np.newaxis,:]]).T,
                columns=['thr', 'score', 'smoothed_score'])
        G = thr_perf_df.shape[0]
        disp_ids = np.unique(np.hstack([
            [0,1,2], [gg-2, gg-1, gg, gg+1, gg+2], [G-3, G-2, G-1]]))
        
        disp_ids = disp_ids[disp_ids>=0]
        print(thr_perf_df.loc[disp_ids].to_string(index=False))
        print("Chosen Threshold: %.4f" % best_thr)
        best_clf = ThresholdClassifier(
            hyper_searcher.best_estimator_, threshold=best_thr)
    else:
        best_clf = hyper_searcher.best_estimator_
    
    # save model to disk
    print('saving %s model to disk' % args.clf_name)
    model_file = os.path.join(
        args.output_dir, args.clf_name+'_trained_model.joblib')
    dump(best_clf, model_file)


    # Evaluation of selected model
    # ----------------------------
    # Loop thru each split, and produce:
    # * performance metrics for this split
    # * diagnostic plots for this split
    row_dict_list = list()
    extra_list = list()
    for split_name, x, y in [
            ('train', x_train, y_train),
            ('test', x_test, y_test)]:
        row_dict = dict(split_name=split_name, n_examples=x.shape[0], n_labels_positive=np.sum(y))
        row_dict['frac_labels_positive'] = np.sum(y) / x.shape[0]

        y_pred = best_clf.predict(x)
        y_pred_proba = best_clf.predict_proba(x)[:, 1]

        # Metrics that require a threshold
        # --------------------------------
        # (e.g. use y_pred not y_pred_proba)
        confusion_arr = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(confusion_arr, columns=[0,1], index=[0,1])
        cm_df.columns.name = 'Predicted label'
        cm_df.index.name = 'True label'
        
        accuracy = accuracy_score(y, y_pred)
        balanced_accuracy = balanced_accuracy_score(y, y_pred)

        row_dict.update(dict(
            confusion_html=cm_df.to_html(),
            accuracy=accuracy,
            f1_score=f1_score(y, y_pred),
            balanced_accuracy=balanced_accuracy))


        # Metrics that consume probabilities
        # ----------------------------------
        if not is_multiclass:
            log2_loss = calc_cross_entropy_base2_score(y, y_pred_proba)
            avg_precision = average_precision_score(y, y_pred_proba)
            roc_auc = roc_auc_score(y, y_pred_proba)
            row_dict.update(dict(
                cross_entropy_base2=log2_loss,
                average_precision=avg_precision,
                AUROC=roc_auc))
            # This computation is ordered with *decreasing* recall
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            # To compute area under PR curve, integrate *increasing* recall
            row_dict['AUPRC'] = np.trapz(precision[::-1], recall[::-1])
        
            npv, ppv = np.diag(cm_df.values) / cm_df.sum(axis=0)
            tnr, tpr = np.diag(cm_df.values) / cm_df.sum(axis=1)
            row_dict.update(dict(TPR=tpr, TNR=tnr, PPV=ppv, NPV=npv))
            row_dict.update(dict(
                tn=cm_df.values[0,0], fp=cm_df.values[0,1],
                fn=cm_df.values[1,0], tp=cm_df.values[1,1]))

            # Make plots and save to disk
            # ---------------------------
            # PLOT #1: Calibration
            B = 0.03
            fig_h = plt.figure(**FIG_KWARGS)
            plot_binary_clf_calibration_curve_and_histograms(
                y, y_pred_proba, bins=11, B=B)
            plt.savefig(os.path.join(fig_dir,
                '{split_name}_calibration_curve.png'.format(
                    split_name=split_name)))
            plt.close()

            # PLOT #2: ROC curve
            roc_fpr, roc_tpr, _ = roc_curve(y, y_pred_proba)
            fig_h = plt.figure(**FIG_KWARGS)
            ax = plt.gca()
            ax.plot(roc_fpr, roc_tpr, 'b.-')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_xlim([-B, 1.0 + B])
            ax.set_ylim([-B, 1.0 + B])
            plt.savefig(os.path.join(fig_dir,
                '{split_name}_roc_curve.png'.format(
                    split_name=split_name)))
            plt.close()

            # Plot #3: PR curve
            fig_h = plt.figure(**FIG_KWARGS)
            ax = plt.gca()
            ax.plot(recall, precision, 'b.-') # computed above to get AUPRC
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim([-B, 1.0 + B])
            ax.set_ylim([-B, 1.0 + B])
            plt.savefig(os.path.join(fig_dir,
                '{split_name}_pr_curve.png'.format(
                    split_name=split_name)))
            plt.close()


        # Append current split's metrics to list for all splits
        row_dict_list.append(row_dict)


    # Write performance metrics to CSV
    # --------------------------------
    # Package up all per-split metrics into one dataframe and save to CSV
    perf_df = pd.DataFrame(row_dict_list)
    perf_df = perf_df.set_index('split_name')
    csv_path = os.path.join(fig_dir, 'performance_df.csv')
    perf_df.to_csv(csv_path)
    print("Wrote performance metrics to: %s" % csv_path)


    # Write HTML report
    # ------------------
#     os.chdir(fig_dir)
    doc, tag, text = Doc().tagtext()
    pd.set_option('display.precision', 4)
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
                                doc.stag('img', src=os.path.join(fig_dir, 'train_calibration_curve.png'), width=400)
                            with tag('td', align='center'):
                                doc.stag('img', src=os.path.join(fig_dir, 'test_calibration_curve.png'), width=400)
                        with tag('tr'):
                            with tag('td', align='center', **{'text-align':'center'}):
                                doc.asis(str(perf_df.iloc[0][['confusion_html']].values[0]).replace('&lt;', '<').replace('&gt;', '>').replace('\\n', ''))
                            with tag('td', align='center', **{'text-align':'center'}):
                                doc.asis(str(perf_df.iloc[1][['confusion_html']].values[0]).replace('&lt;', '<').replace('&gt;', '>').replace('\\n', ''))
                   
                    with tag('h3'):
                        with tag('a', name='results-performance-metrics-proba'):
                            text('Performance Metrics using Probabilities')
                    doc.asis(perf_df[['AUROC', 'AUPRC', 'average_precision', 'cross_entropy_base2']].to_html())

                    with tag('h3'):
                        with tag('a', name='results-performance-metrics-thresh'):
                            text('Performance Metrics using Thresholded Decisions')
                    doc.asis(perf_df[['balanced_accuracy', 'accuracy', 'f1_score', 'TPR', 'TNR', 'PPV', 'NPV']].to_html())

                    with tag('h3'):
                        with tag('a', name='settings-hyperparameter'):
                            text('Settings: Hyperparameters to Tune')
                    with tag('pre'):
                        hyper_group = None
                        for g in subparsers_by_name[args.clf_name]._action_groups:
                            if g.title.count('hyperparameters'):
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
                        with tag('a', name='results-hyper-search'):
                            text('Hyperparameter Search results')
                    with tag('h4'):
                        text('Train Scores across splits')
                    with tag('p'):
                        text("Score function: %s" % args.scoring)
                    with tag('p'):
                        text("Selected hypers: %s" % str(best_param_dict))
                    doc.asis(pd.DataFrame(cv_tr_perf_df).to_html())

                    with tag('h4'):
                        text('Heldout Scores across splits')
                    with tag('p'):
                        text("Score function: %s" % args.scoring)
                    with tag('p'):
                        text("Selected hypers: %s" % str(best_param_dict))
                    doc.asis(pd.DataFrame(cv_te_perf_df).to_html())

                # Add dark region on right side
                with tag('div', klass="col-sm-1 sidenav"):
                    text("")

    with open(os.path.join(args.output_dir, 'report.html'), 'w') as f:
        f.write(doc.getvalue())
