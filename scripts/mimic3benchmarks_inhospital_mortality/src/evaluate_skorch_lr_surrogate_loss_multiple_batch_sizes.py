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
            training_hist_df = pd.read_csv(f)
            precision_train_np[i] = training_hist_df.precision_train.values[-1]
            recall_train_np[i] = training_hist_df.recall_train.values[-1]
            precision_valid_np[i] = training_hist_df.precision_valid.values[-1]
            recall_valid_np[i] = training_hist_df.recall_valid.values[-1]
            precision_test_np[i] = training_hist_df.precision_test.values[-1]
            recall_test_np[i] = training_hist_df.recall_test.values[-1]
        
    
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

    sl_rand_init_direct_min_precision_filename_aka = 'skorch_logistic_regression*surrogate_loss_tight*warm_start=false*incremental_min_precision=false*history.json'
    ### plot precision recall as a function of batch sizes for the random init case
    tr_files = glob.glob(os.path.join(clf_models_dir, sl_rand_init_direct_min_precision_filename_aka))
    batch_sizes = list() 
    for ii, tr_file in enumerate(tr_files): 
        batch_sizes.append(tr_file.split('batch_size=')[1].split('-scoring')[0]) 
    
    unique_batch_sizes = np.unique(batch_sizes)
    unique_batch_sizes = sorted(np.array(unique_batch_sizes).astype(int))
    precisions_train_per_batch_size = list()
    recalls_train_per_batch_size = list()
    precisions_valid_per_batch_size = list()
    recalls_valid_per_batch_size = list()
    precisions_test_per_batch_size = list()
    recalls_test_per_batch_size = list()
    
    for batch_size in unique_batch_sizes:
        
        curr_batch_size_fnames_aka = 'skorch_logistic_regression*batch_size=%s*surrogate_loss_tight*warm_start=false*incremental_min_precision=false*history.json'%str(batch_size)
        precisions_train_per_batch_size_direct_min_precision, precisions_valid_per_batch_size_direct_min_precision, precisions_test_per_batch_size_direct_min_precision, recalls_train_per_batch_size_direct_min_precision, recalls_valid_per_batch_size_direct_min_precision, recalls_test_per_batch_size_direct_min_precision, tr_files_per_batch_size_direct_min_precision = get_all_precision_recalls(clf_models_dir, curr_batch_size_fnames_aka.replace('history.json', '.csv'))
        
        precisions_train_per_batch_size.append(precisions_train_per_batch_size_direct_min_precision)
        recalls_train_per_batch_size.append(recalls_train_per_batch_size_direct_min_precision)
        precisions_valid_per_batch_size.append(precisions_valid_per_batch_size_direct_min_precision)
        recalls_valid_per_batch_size.append(recalls_valid_per_batch_size_direct_min_precision)
        precisions_test_per_batch_size.append(precisions_test_per_batch_size_direct_min_precision)
        recalls_test_per_batch_size.append(recalls_test_per_batch_size_direct_min_precision)
        
        
    
    f, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True) 
    sns.set_context("notebook", font_scale=1.25)
    sns.set_style("whitegrid")
    fontsize=12
    
    for jj, (split, precisions_per_batch_size, recalls_per_batch_size) in enumerate([('train', 
                                                                                      precisions_train_per_batch_size,
                                                                                      recalls_train_per_batch_size),
                                                                                    ('validation',
                                                                                    precisions_valid_per_batch_size,
                                                                                    recalls_valid_per_batch_size),
                                                                                    ('test',
                                                                                    precisions_test_per_batch_size,
                                                                                    recalls_test_per_batch_size)]):
    
        for ii in range(len(unique_batch_sizes)):  
            axs[jj].plot(precisions_per_batch_size[ii], recalls_per_batch_size[ii], '.',  
                     label='batch_size = '+ str(unique_batch_sizes[ii]))  
            axs[jj].set_xlabel('Precision', fontsize=fontsize) 
            axs[jj].set_ylabel('Recall', fontsize=fontsize) 
            if jj == 1:
                axs[jj].legend(bbox_to_anchor=(.8, .4), fontsize=fontsize)   
            axs[jj].set_title('Precision vs Recall on %s set'%(split), fontsize=fontsize+2)
            axs[jj].set_xticks(np.arange(0, 1.1, 0.1))
            axs[jj].set_xlim([0., 1.0])
            axs[jj].grid(True)
            
#     f.savefig('precision_recalls_batch_sizes.png', bbox_inches='tight', pad_inches=0) 

    f.savefig('precision_recalls_batch_sizes.pdf', bbox_inches='tight', pad_inches=0) 
    
    
    
    
 