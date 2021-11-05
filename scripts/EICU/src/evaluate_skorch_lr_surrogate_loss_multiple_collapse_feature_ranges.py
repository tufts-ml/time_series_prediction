'''
Evaluate performance of a single trained classifier on multiple patient-stay-slices
'''
import os
import numpy as np
import pandas as pd
from joblib import dump, load
import sys
sys.path.append(os.path.join(os.path.abspath('../'), 'src'))

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchLogisticRegression'))
from sklearn.preprocessing import StandardScaler
from SkorchLogisticRegression import SkorchLogisticRegression

#import LR model before importing other packages because joblib files act weird when certain packages are loaded
from feature_transformation import *
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score)
from utils import load_data_dict_json
import ast
import random
import pickle
import glob
import seaborn as sns
from split_dataset import Splitter

def get_all_precision_recalls(clf_models_dir, filename_aka):
    ''' Get the best model from training history'''
    training_files = glob.glob(os.path.join(clf_models_dir, filename_aka))
    precision_train_np = np.zeros(len(training_files))
    recall_train_np = np.zeros(len(training_files))
    precision_valid_np = np.zeros(len(training_files))
    recall_valid_np = np.zeros(len(training_files))
    precision_test_np = np.zeros(len(training_files))
    recall_test_np = np.zeros(len(training_files))
    
    for i, f in enumerate(training_files):
        try:
            training_hist_df = pd.read_csv(f)
            precision_train_np[i] = training_hist_df.precision_train.values[-1]
            recall_train_np[i] = training_hist_df.recall_train.values[-1]
            precision_valid_np[i] = training_hist_df.precision_valid.values[-1]
            recall_valid_np[i] = training_hist_df.recall_valid.values[-1]
            precision_test_np[i] = training_hist_df.precision_test.values[-1]
            recall_test_np[i] = training_hist_df.recall_test.values[-1]
        except:
            continue
        
    
    precision_train_unfiltered = precision_train_np
    precision_valid_unfiltered = precision_valid_np
    precision_test_unfiltered = precision_test_np
    recall_train_unfiltered = recall_train_np
    recall_valid_unfiltered = recall_valid_np
    recall_test_unfiltered = recall_test_np    
    

    return precision_train_unfiltered, precision_valid_unfiltered, precision_test_unfiltered, recall_train_unfiltered, recall_valid_unfiltered, recall_test_unfiltered, training_files
  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_models_dir', default=None, type=str,
                        help='Directory where classifier models are saved')
    parser.add_argument('--clf_train_test_split_dir', default=None, type=str,
                        help='Directory where the train-test split data for the classifier is saved')
    parser.add_argument('--outcome_column_name', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')

    
    
    args = parser.parse_args()
    
    clf_models_dir = args.clf_models_dir
    
    
    ## get the test patient id's
    # get the test set's csv and dict
    collapse_feature_data_dir = 'skorch_logistic_regression'
    collapse_feature_model_dirs = ['skorch_logistic_regression_small', 'skorch_logistic_regression']
    f, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
#     f_va, axs_va = plt.subplots(1, 1, figsize=(8, 8))
#     f_te, axs_te = plt.subplots(1, 1, figsize=(8, 8))
    sns.set_context("notebook", font_scale=1.25)
    sns.set_style("whitegrid")
    fontsize=12
    
    for mm, (collapse_feature_type, collapse_feature_model_dir) in enumerate([('original summary statistics',
                                                                               'skorch_logistic_regression_small'),
                                                                             ('improved summary statistics',
                                                                              'skorch_logistic_regression')]):
        
        
        if collapse_feature_model_dir=='skorch_logistic_regression_small':
            x_train_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'x_train_small.csv'))
            y_train_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_train_small.csv'))
            x_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'x_test_small.csv'))
            y_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_test_small.csv'))
            y_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'y_dict_small.json')
            x_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'x_dict_small.json')            
            
        else:
            x_train_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'x_train.csv'))
            y_train_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_train.csv'))
            x_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'x_test.csv'))
            y_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_test.csv'))
            y_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'y_dict.json')
            x_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'x_dict.json')
        
        
        scaler = pickle.load(open(os.path.join(clf_models_dir, collapse_feature_model_dir, 'scaler.pkl'), 'rb'))

        # import the y dict to get the id cols
        y_test_dict = load_data_dict_json(y_test_dict_file)
        x_test_dict = load_data_dict_json(x_test_dict_file)
        id_cols = parse_id_cols(y_test_dict)
        feature_cols = parse_feature_cols(x_test_dict)
        outcome_col = args.outcome_column_name

        # get performance metrics
        x_train = x_train_df[feature_cols].values.astype(np.float32)
        y_train = y_train_df[outcome_col].values

        x_test = x_test_df[feature_cols].values.astype(np.float32)
        y_test = y_test_df[outcome_col].values


        # get the validation data
        splitter = Splitter(size=0.25, random_state=41,
                            n_splits=1, 
                            cols_to_group='subject_id')
        # Assign training instances to splits by provided keys
        key_train = splitter.make_groups_from_df(x_train_df[id_cols])

        for ss, (tr_inds, va_inds) in enumerate(splitter.split(x_train, y_train, groups=key_train)):
            x_tr = x_train[tr_inds].copy()
            y_tr = y_train[tr_inds].copy()
            x_valid = x_train[va_inds]
            y_valid = y_train[va_inds]

        y_train = y_tr
        del(y_tr)


        # load the scaler
        x_train_transformed = scaler.transform(x_tr)
        x_valid_transformed = scaler.transform(x_valid)
        x_test_transformed = scaler.transform(x_test)
        
        curr_clf_models_dir = os.path.join(clf_models_dir, collapse_feature_model_dir)
        
        prcs_train, prcs_valid, prcs_test, recs_train, recs_valid, recs_test, tr_files = get_all_precision_recalls(curr_clf_models_dir, 'skorch_logistic_regression*cross_entropy_loss*warm_start=false*.csv')
        
        # choose model with max recall on validation given precision of 0.7 on train and 0.6 on validation
        min_prec_tr = 0.5
        min_prec_va = 0.4
        keep_inds = (prcs_train>min_prec_tr)&(prcs_valid>min_prec_va)
        
        if keep_inds.sum()==0:
            keep_inds = (prcs_train>min_prec_tr)&(prcs_valid>0.45)
        
        fracs_above_min_precision = (keep_inds).sum()/len(prcs_train)
        prcs_train = prcs_train[keep_inds]
        prcs_valid = prcs_valid[keep_inds]
        prcs_test = prcs_test[keep_inds]        
        recs_train = recs_train[keep_inds]
        recs_valid = recs_valid[keep_inds]
        recs_test = recs_test[keep_inds] 
        tr_files = np.array(tr_files)[keep_inds]        
        best_ind = np.argmax(recs_valid)
        best_model_fname = tr_files[best_ind] 
        
        ### select 1 classifier and plot its precision and recalls across many thresholds on train, valid and test


        skorch_lr_clf = SkorchLogisticRegression(n_features=x_test.shape[1])
        skorch_lr_clf.initialize()
        skorch_lr_clf.load_params(f_params=best_model_fname.replace('_perf.csv', 'params.pt'))

        y_train_pred_probas = skorch_lr_clf.predict_proba(x_train_transformed)[:,1]
        y_train_preds = y_train_pred_probas>=0.5        

        y_valid_pred_probas = skorch_lr_clf.predict_proba(x_valid_transformed)[:,1]
        y_valid_preds = y_valid_pred_probas>=0.5

        y_test_pred_probas = skorch_lr_clf.predict_proba(x_test_transformed)[:,1]
        y_test_preds = y_test_pred_probas>=0.5

        precision_train, recall_train, thresholds_pr_train = precision_recall_curve(y_train, y_train_pred_probas)
        precision_valid, recall_valid, thresholds_pr_valid = precision_recall_curve(y_valid, y_valid_pred_probas)
        precision_test, recall_test, thresholds_pr_test = precision_recall_curve(y_test, y_test_pred_probas)

        target_precisions = np.arange(0.1, 1, 0.01) 


        recalls_at_target_precisions_train, recalls_at_target_precisions_valid, recalls_at_target_precisions_test = [np.zeros(len(target_precisions)), np.zeros(len(target_precisions)), np.zeros(len(target_precisions))]
        for kk, target_precision in enumerate(target_precisions):
            keep_inds_tr = precision_train>=target_precision
            keep_inds_va = precision_valid>=target_precision
            keep_inds_te = precision_test>=target_precision

            recalls_at_target_precisions_train[kk] = max(recall_train[keep_inds_tr])
            recalls_at_target_precisions_valid[kk] = max(recall_valid[keep_inds_va])
            recalls_at_target_precisions_test[kk] = max(recall_test[keep_inds_te])


#         chosen_thresh_ind = (target_precisions>=0.599)&(target_precisions<=0.601)

        for jj, (split, precs, recs, chosen_prec, chosen_rec, f, axs) in enumerate([
            ('train', target_precisions, recalls_at_target_precisions_train, 
             prcs_train[best_ind], recs_train[best_ind], f, axs),
            ('valid', target_precisions, recalls_at_target_precisions_valid, 
             prcs_valid[best_ind], recs_valid[best_ind], f, axs),
            ('test', target_precisions, recalls_at_target_precisions_test, 
             prcs_test[best_ind], recs_test[best_ind], f, axs)]):
            axs[jj].plot(recs, precs, label=collapse_feature_type, linewidth=3, zorder=1)
            axs[jj].set_ylabel('Target precision', fontsize=fontsize+2)
            axs[jj].set_title('Recalls at target precision on %s'%split, fontsize=fontsize+4)
            xmin = -0.003
            xmax = 0.5
            ymin = 0.35
            ymax = 1.0
            xticks = np.arange(0.0, 1.0, 0.05)
            xticklabels = ['%.2f'%x for x in xticks]
            axs[jj].set_xticks(xticks)
            axs[jj].set_xticklabels(xticklabels)
            axs[jj].set_xlim([xmin, xmax])
            axs[jj].set_ylim([ymin, ymax])
            if jj==2:
                axs[jj].set_xlabel('Recall at target precision', fontsize=fontsize+2)

            chosen_thresh_ind = np.argmin(abs(chosen_prec - precs))

            if mm==1:
                axs[jj].plot(recs[chosen_thresh_ind], precs[chosen_thresh_ind], color='k', marker='+', mew=3, markersize=25,
                            label='chosen target_precision', zorder=2)   
            else:
                axs[jj].plot(recs[chosen_thresh_ind], precs[chosen_thresh_ind], color='k', marker='+', mew=3, markersize=25, zorder=2)                  
            if jj==2:
                axs[jj].legend(fontsize=fontsize-3)
    f.savefig('skorch_lr_multiple_collapse_feature_comparison_bce_%s.png'%split, bbox_inches='tight', pad_inches=0)
    from IPython import embed; embed()
        
        
        