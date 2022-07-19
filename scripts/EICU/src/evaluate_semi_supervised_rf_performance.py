'''
Evaluate performance of a single trained classifier on multiple patient-stay-slices
'''
import os
import numpy as np
import pandas as pd
from joblib import dump, load
import sys
import torch
sys.path.append(os.path.join(os.path.abspath('../'), 'src'))

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

#import LR model before importing other packages because joblib files act weird when certain packages are loaded
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score)
import random
from scipy.special import softmax
import glob
import argparse


def get_best_model_file(saved_model_files_aka):
    
    training_files = glob.glob(saved_model_files_aka)
    aucroc_per_fit_list = []
    auprc_per_fit_list = []
    for f in training_files: 
        perf_df = pd.read_csv(f)
        aucroc_per_fit_list.append(perf_df['valid_AUROC'].values[-1])
        auprc_per_fit_list.append(perf_df['valid_AUPRC'].values[-1])
    
    best_fit_ind = np.argmax(auprc_per_fit_list)
    best_model_file = training_files[best_fit_ind]    
    
    return best_model_file



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_models_dir', default=None, type=str,
                        help='Directory where classifier models are saved')
    parser.add_argument('--clf_train_test_split_dir', default=None, type=str,
                        help='Directory where the train-test split data for the classifier is saved')
    parser.add_argument('--tslice_folder', type=str, 
                        help='folder where features filtered by tslice are stored')
    parser.add_argument('--preproc_data_dir', type=str,
                        help='folder where the preprocessed data is stored')
    parser.add_argument('--outcome_column_name', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    parser.add_argument('--random_seed_list', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    parser.add_argument('--output_dir', default=' ', type=str,
                       help='name of outcome column in test dataframe')
    
    
    args = parser.parse_args()

    clf_models_dir = args.clf_models_dir
    ## get the test data
    x_train_np_filename = os.path.join(args.clf_train_test_split_dir, 'X_train_collapsed.npy')
    x_valid_np_filename = os.path.join(args.clf_train_test_split_dir, 'X_valid_collapsed.npy')
    x_test_np_filename = os.path.join(args.clf_train_test_split_dir, 'X_test_collapsed.npy')
    y_train_np_filename = os.path.join(args.clf_train_test_split_dir, 'y_train_collapsed.npy')
    y_valid_np_filename = os.path.join(args.clf_train_test_split_dir, 'y_valid_collapsed.npy')
    y_test_np_filename = os.path.join(args.clf_train_test_split_dir, 'y_test_collapsed.npy')    
    
    
    X_train_fs = np.load(x_train_np_filename)
    X_test = np.load(x_test_np_filename)
    y_train = np.load(y_train_np_filename)
    y_test = np.load(y_test_np_filename)
    X_valid = np.load(x_valid_np_filename)
    y_valid = np.load(y_valid_np_filename)
       
    perf_df = pd.DataFrame()
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    for perc_labelled in ['1.2', '3.7', '11.1', '33.3', '100']:

        saved_model_files_aka = os.path.join(clf_models_dir, 
                                             "final_perf_rf*perc_labelled=%s*.csv"%(perc_labelled))
        best_model_file = get_best_model_file(saved_model_files_aka).replace('.csv', '_trained_model.joblib').replace('final_perf_', '')
        print('Evaluating RF model with perc_labelled =%s \nfile=%s'%(perc_labelled, best_model_file))
        model = load(best_model_file)
        
        outputs = model.predict_proba(X_test)[:,1]
        targets = y_test
                
        
#         bootstrapping to get CI on metrics
        roc_auc_np = np.zeros(len(random_seed_list))
        avg_precision_np = np.zeros(len(random_seed_list))

        for k, seed in enumerate(random_seed_list):
            random.seed(int(seed))
            rnd_inds = random.sample(range(targets.shape[0]), int(0.8*targets.shape[0])) 
            curr_y_test = targets[rnd_inds]
#             curr_x_test = inputs[rnd_inds, :]
            curr_y_pred = np.argmax(outputs[rnd_inds], -1)
            curr_y_pred_proba = outputs[rnd_inds]

            roc_auc_np[k] = roc_auc_score(curr_y_test, curr_y_pred_proba)
            avg_precision_np[k] = average_precision_score(curr_y_test, curr_y_pred_proba)

        print('perc_labelled = %s, \nMedian ROC-AUC : %.3f'%(perc_labelled, np.percentile(roc_auc_np, 50)))
        print('Median average precision : %.3f'%np.percentile(avg_precision_np, 50))
        

        for prctile in prctile_vals:
            row_dict = dict()
            row_dict['model'] = 'Random Forest'
            row_dict['percentile'] = prctile
            row_dict['perc_labelled'] = perc_labelled
            row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
            row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)
            perf_df = perf_df.append(row_dict, ignore_index=True)      

    perf_csv = os.path.join(args.output_dir, 'RF_performance.csv')
    print('Saving Random Forest performance to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)
