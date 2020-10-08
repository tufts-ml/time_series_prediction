import argparse
import ast
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve)

from sklearn.svm import SVC


DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))

from utils_scoring import (THRESHOLD_SCORING_OPTIONS, calc_score_for_binary_predictions)
from custom_classifiers import ThresholdClassifier
from split_dataset import Splitter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


try:
    TEMPLATE_HTML_PATH = os.path.join(PROJECT_REPO_DIR, 'src', 'template.html')
except KeyError:
    TEMPLATE_HTML_PATH = None

if __name__ == '__main__':

    # Parse pre-specified command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_files', type=str, required=True)
    parser.add_argument('--test_csv_files', type=str, required=True)
    parser.add_argument('--data_dict_files', type=str, required=True)
    parser.add_argument('--output_dir', default='./html/', type=str, required=False)
    parser.add_argument('--merge_x_y', default=True,
                                type=lambda x: (str(x).lower() == 'true'), required=False,
                                help='If True, features and outcomes are merged on id columns, if false, they get concatenated.')
    parser.add_argument('--outcome_col_name', type=str, required=False)
    parser.add_argument('--key_cols_to_group_when_splitting', type=str,
            default=None, nargs='*')
    parser.add_argument('--scoring', type=str, default='roc_auc_score')
    parser.add_argument('--random_seed', type=int, default=8675309)
    parser.add_argument('--threshold_scoring', type=str,
            default=None, choices=[None, 'None'] + THRESHOLD_SCORING_OPTIONS)
    parser.add_argument('--validation_size', type=float, default=0.1)
    parser.add_argument('--n_splits', type=int, default=1)
    
    args=parser.parse_args()
    fig_dir = os.path.abspath(args.output_dir)
    argdict = vars(args)
    
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
    x_train = df_by_split['train'][feature_cols].values
    y_train = np.ravel(df_by_split['train'][outcome_col_name])

    x_test = df_by_split['test'][feature_cols].values
    y_test = np.ravel(df_by_split['test'][outcome_col_name])
    is_multiclass = len(np.unique(y_train)) > 2
    
    # Perform hyper_searcher search
    n_examples = x_train.shape[0]
    n_features = x_train.shape[1]
    

    splitter = Splitter(size=args.validation_size, random_state=args.random_seed, n_splits=args.n_splits, cols_to_group=args.key_cols_to_group_when_splitting)
    key_train = splitter.make_groups_from_df(df_by_split['train'][key_cols])
    # hyper_searcher search on validation over possible threshold values
    # Make sure all candidates at least provide
    # one instance of each class (positive and negative)
    yproba_class1_vals = list()
    for tr_inds, va_inds in splitter.split(x_train, groups=key_train):
        x_valid = x_train[va_inds]
        yproba_valid = x_valid
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
        np.percentile(nontrivial_thr_vals, 95),90)
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
        print("thr_grid = %.4f, %.4f, %.4f ... %.4f, %.4f" % 
                (thr_grid[0], thr_grid[1], thr_grid[2], thr_grid[-2], thr_grid[-1]))
    
        ## TODO find better way to do this fast
        # So we dont need to call fit at each thr value
    #linear_clf = make_pipeline(StandardScaler(), SVC(C=1e-7, random_state=args.random_seed, tol=1e-2,  kernel='linear',  probability=True, class_weight='balanced'))
    #linear_clf.fit(x_train, y_train)
    score_grid_SG = np.zeros((splitter.n_splits, thr_grid.size))
    for ss, (tr_inds, va_inds) in enumerate(splitter.split(x_train, y_train, groups=key_train)):
        x_tr = x_train[tr_inds].copy()
        y_tr = y_train[tr_inds].copy()
        x_va = x_train[va_inds]
        y_va = y_train[va_inds]
        
        # Use a linear classifier to classify based on scores and choose the best score as thresold
        #tmp_clf = ThresholdClassifier(linear_clf)
        #tmp_clf.fit(x_tr, y_tr)

        for gg, thr in enumerate(thr_grid):
            #tmp_clf = tmp_clf.set_params(threshold=thr)
            yhat = x_va > thr
            score_grid_SG[ss, gg] = calc_score_for_binary_predictions(y_va, yhat, scoring=args.threshold_scoring)
        ## TODO read off best average score
    avg_score_G = np.mean(score_grid_SG, axis=0)
    gg = np.argmax(avg_score_G)
        # Keep best scoring estimator 
    best_thr = thr_grid[gg]
    print("Chosen Threshold: %.4f" % best_thr)
    #best_clf = ThresholdClassifier(linear_clf,i threshold=best_thr)


    # Evaluation
    row_dict_list = list()
    extra_list = list()
    max_mews_score = np.vstack([x_train, x_test]).max()
    for split_name, x, y in [
            ('train', x_train, y_train),
            ('test', x_test, y_test)]:
        row_dict = dict(split_name=split_name, n_examples=x.shape[0], n_labels_positive=np.sum(y))
        row_dict['frac_labels_positive'] = np.sum(y) / x.shape[0]
        
        y_scores = x
        #y_pred_proba = y_scores/max_mews_score
        y_pred = (y_scores > best_thr)*1
        
        confusion_arr = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(confusion_arr, columns=[0,1], index=[0,1])
        cm_df.columns.name = 'Predicted label'
        cm_df.index.name = 'True label'
        
        accuracy = accuracy_score(y, y_pred)
        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        log2_loss = log_loss(y, y_scores/max_mews_score, normalize=True) / np.log(2)
        row_dict.update(dict(cross_entropy_base2=log2_loss, accuracy=accuracy, balanced_accuracy=balanced_accuracy))
        if not is_multiclass:
            f1 = f1_score(y, y_pred)

            avg_precision = average_precision_score(y, y_scores)
            roc_auc = roc_auc_score(y, y_scores)
            row_dict.update(dict(f1_score=f1, average_precision=avg_precision, AUROC=roc_auc))
            '''
            # Plots
            roc_fpr, roc_tpr, _ = roc_curve(y, y_scores)
            ax = plt.gca()
            ax.plot(roc_fpr, roc_tpr)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.savefig(os.path.join(fig_dir, '{split_name}_roc_curve_random_seed={random_seed}.png'.format(split_name=split_name, random_seed=str(args.random_seed))))
            plt.clf()

            pr_precision, pr_recall, _ = precision_recall_curve(y, y_scores)
            ax = plt.gca()
            ax.plot(pr_recall, pr_precision)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.savefig(os.path.join(fig_dir, '{split_name}_pr_curve_random_seed={random_seed}.png'.format(split_name=split_name, random_seed=str(args.random_seed))))
            plt.clf()
            '''

        row_dict_list.append(row_dict)

    perf_df = pd.DataFrame(row_dict_list)
    perf_df = perf_df.set_index('split_name')
    #print('saving to %s'%(os.path.join(fig_dir, 'performance_df_random_seed={random_seed}.csv'.format(random_seed=str(args.random_seed)))))
    perf_df.to_csv(os.path.join(fig_dir, 'mews_performance_df.csv'))
    best_thr_df = pd.DataFrame({'best_thr':best_thr}, index=[0])
    best_thr_df.to_csv(os.path.join(fig_dir, 'mews_best_threshold.csv'), index=False)


