import argparse
import ast
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve)

from utils_scoring import (THRESHOLD_SCORING_OPTIONS, calc_score_for_binary_predictions)

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

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

    parser.add_argument('--outcome_col_name', type=str, required=False)
    parser.add_argument('--key_cols_to_group_when_splitting', type=str,
            default=None, nargs='*')
    parser.add_argument('--scoring', type=str, default='roc_auc_score')
    parser.add_argument('--random_seed', type=int, default=8675309)
    parser.add_argument('--threshold_scoring', type=str,
            default=None, choices=[None, 'None'] + THRESHOLD_SCORING_OPTIONS)
    
    args=parser.parse_args()
    fig_dir = os.path.abspath(args.output_dir)
    
    # key[5:] strips the 'grid_' prefix from the argument
    argdict = vars(args)
    #raw_param_grid = {
    #    key[5:]: argdict[key] for key in argdict if key.startswith('grid_')}
    '''
    raw_param_grid = dict()
    for key, val  in argdict.items():
        if key.startswith('grid_'):
            raw_param_grid[key[5:]] = val
        elif key.startswith('filter_'):
            raw_param_grid[key] = val
    '''
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
        y_pred_proba = y_scores/max_mews_score
        y_pred = (y_pred_proba > 0.5)*1
        
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
            plt.savefig(os.path.join(fig_dir, '{split_name}_roc_curve_random_seed={random_seed}.png'.format(split_name=split_name, random_seed=str(args.random_seed))))
            plt.clf()

            pr_precision, pr_recall, _ = precision_recall_curve(y, y_pred_proba)
            ax = plt.gca()
            ax.plot(pr_recall, pr_precision)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.savefig(os.path.join(fig_dir, '{split_name}_pr_curve_random_seed={random_seed}.png'.format(split_name=split_name, random_seed=str(args.random_seed))))
            plt.clf()

        row_dict_list.append(row_dict)

    perf_df = pd.DataFrame(row_dict_list)
    perf_df = perf_df.set_index('split_name')
    #print('saving to %s'%(os.path.join(fig_dir, 'performance_df_random_seed={random_seed}.csv'.format(random_seed=str(args.random_seed)))))
    perf_df.to_csv(os.path.join(fig_dir, 'performance_df_random_seed={random_seed}.csv'.format(random_seed=str(args.random_seed))))


