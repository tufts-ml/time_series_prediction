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
    

    return precision_train_unfiltered, precision_valid_unfiltered, precision_test_unfiltered, recall_train_unfiltered, recall_valid_unfiltered, recall_test_unfiltered
  
    
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
    
    '''
    ## get the test patient id's
    # get the test set's csv and dict
    y_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_test.csv'))
    y_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'y_dict.json')
    x_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'x_test.csv'))
    x_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'x_dict.json')
    
    # import the y dict to get the id cols
    y_test_dict = load_data_dict_json(y_test_dict_file)
    x_test_dict = load_data_dict_json(x_test_dict_file)
    id_cols = parse_id_cols(y_test_dict)
    feature_cols = parse_feature_cols(x_test_dict)
    
    outcome_col = args.outcome_column_name
    y_test_ids_df = y_test_df[id_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)
    
    # get performance metrics
    x_test = x_test_df[feature_cols].values
    y_test = y_test_df[outcome_col].values

    # load the scaler
    scaler = pickle.load(open(os.path.join(clf_models_dir, 'scaler.pkl'), 'rb'))
    x_test_transformed = scaler.transform(x_test)
    '''

    
    
    bound_model_dirs = glob.glob(os.path.join(clf_models_dir, '*gamma_fp=*')) 
    
    old_bounds_dirs = [i for i in bound_model_dirs if 'delta' not in i]
    new_bounds_dirs = [i for i in bound_model_dirs if 'delta' in i]
    
    gamma_list = ['$\gamma_{fp}$=3, $\delta_{fp}$=0.009, $\epsilon_{fp}$=0.13, $m_{fp}$=64.8, $b_{fp}=4.0$',
                  '$\gamma_{fp}$=5, $\delta_{fp}$=0.015, $\epsilon_{fp}$=0.37, $m_{fp}$=18.8, $b_{fp}=2.8$', 
                  '$\gamma_{fp}$=7, $\delta_{fp}$=0.021, $\epsilon_{fp}$=0.73, $m_{fp}$=8.2, $b_{fp}=2.0$']
    gamma_colors = ['g', 'tab:orange', 'b', 'k', 'tab:purple']
    gamma_precision_train_list = []
    gamma_precision_valid_list = []
    gamma_precision_test_list = []
    
    gamma_recall_train_list = []
    gamma_recall_valid_list = []
    gamma_recall_test_list = []    
    
    gamma_chosen_precision_train_list = []
    gamma_chosen_precision_valid_list = []
    gamma_chosen_precision_test_list = []
    
    gamma_chosen_recall_train_list = []
    gamma_chosen_recall_valid_list = []
    gamma_chosen_recall_test_list = [] 
    
    
    f, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True) 
    for ii, bound_model_dir in enumerate(old_bounds_dirs):
        sl_rand_init_direct_min_precision_filename_aka = 'skorch_logistic_regression*surrogate_loss_tight*warm_start=false*incremental_min_precision=false*history.json'


        precisions_train_direct_min_precision, precisions_valid_direct_min_precision, precisions_test_direct_min_precision, recalls_train_direct_min_precision, recalls_valid_direct_min_precision, recalls_test_direct_min_precision = get_all_precision_recalls(bound_model_dir, sl_rand_init_direct_min_precision_filename_aka.replace('history.json', '.csv'))
    
        gamma_precision_train_list.append(precisions_train_direct_min_precision)
        gamma_precision_valid_list.append(precisions_valid_direct_min_precision)
        gamma_precision_test_list.append(precisions_test_direct_min_precision)
        
        gamma_recall_train_list.append(recalls_train_direct_min_precision)
        gamma_recall_valid_list.append(recalls_valid_direct_min_precision)
        gamma_recall_test_list.append(recalls_test_direct_min_precision)
        
        keep_inds = (precisions_train_direct_min_precision>=0.7)&(precisions_valid_direct_min_precision>=0.6)
        best_ind = np.argmax(recalls_valid_direct_min_precision[keep_inds])
        
        gamma_chosen_precision_train_list.append(precisions_train_direct_min_precision[keep_inds][best_ind])
        gamma_chosen_precision_valid_list.append(precisions_valid_direct_min_precision[keep_inds][best_ind])
        gamma_chosen_precision_test_list.append(precisions_test_direct_min_precision[keep_inds][best_ind])
        
        gamma_chosen_recall_train_list.append(recalls_train_direct_min_precision[keep_inds][best_ind])
        gamma_chosen_recall_valid_list.append(recalls_valid_direct_min_precision[keep_inds][best_ind])
        gamma_chosen_recall_test_list.append(recalls_test_direct_min_precision[keep_inds][best_ind])        
        
        
#         # plot the precision recalls for each bound
#         axs[0].plot(precisions_train_direct_min_precision, recalls_train_direct_min_precision, '.', label=gamma_list[ii])
#         axs[]
#         axs[1].plot(precisions_valid_direct_min_precision, recalls_valid_direct_min_precision, '.', label=gamma_list[ii])
#         axs[2].plot(precisions_test_direct_min_precision, recalls_test_direct_min_precision, '.', label=gamma_list[ii])

    for jj, (split, precisions_per_gamma, recalls_per_gamma,
             best_precision_per_gamma, best_recall_per_gamma) in enumerate([('train', 
                                                                            gamma_precision_train_list,
                                                                            gamma_recall_train_list,
                                                                            gamma_chosen_precision_train_list,
                                                                            gamma_chosen_recall_train_list),
                                                                           ('validation',
                                                                            gamma_precision_valid_list,
                                                                            gamma_recall_valid_list,
                                                                            gamma_chosen_precision_valid_list,
                                                                            gamma_chosen_recall_valid_list),
                                                                           ('test',
                                                                            gamma_precision_test_list,
                                                                            gamma_recall_test_list,
                                                                            gamma_chosen_precision_test_list,
                                                                            gamma_chosen_recall_test_list)]):

        for kk in range(len(gamma_list)):  
            axs[jj].plot(precisions_per_gamma[kk], recalls_per_gamma[kk], '.',  
                     label=gamma_list[kk], color=gamma_colors[kk], alpha=0.6)  
            axs[jj].plot(best_precision_per_gamma[kk], best_recall_per_gamma[kk], 'x', markersize=20, 
                         color=gamma_colors[kk], alpha=0.6)  
            
            axs[jj].set_xlabel('Precision') 
            axs[jj].set_ylabel('Recall') 
            if jj == 1:
                axs[jj].legend(bbox_to_anchor=(.4, .7), fontsize=8.5)   
            axs[jj].set_title('Precision vs Recall on %s set'%(split))
            axs[jj].set_xticks(np.arange(0, 1.1, 0.1))
            axs[jj].set_xlim([0., 1.0])
            axs[jj].set_ylim([0., 0.4])
            
            
      
    f.savefig('precision_recalls_multiple_bounds_old.png')
    
    
    
    gamma_list = ['$\gamma_{fp}$=2, $\delta_{fp}$=0.015, $\epsilon_{fp}$=0.3, $m_{fp}$=28.0, $b_{fp}=4.2$',
                  '$\gamma_{fp}$=2, $\delta_{fp}$=0.03, $\epsilon_{fp}$=0.6, $m_{fp}$=11.7, $b_{fp}=3.5$',
                  '$\gamma_{fp}$=2, $\delta_{fp}$=0.045, $\epsilon_{fp}$=0.9, $m_{fp}$=6.9, $b_{fp}=3.1$']
    gamma_colors = ['g', 'tab:orange', 'b']
    gamma_precision_train_list = []
    gamma_precision_valid_list = []
    gamma_precision_test_list = []
    
    gamma_recall_train_list = []
    gamma_recall_valid_list = []
    gamma_recall_test_list = []    
    
    gamma_chosen_precision_train_list = []
    gamma_chosen_precision_valid_list = []
    gamma_chosen_precision_test_list = []
    
    gamma_chosen_recall_train_list = []
    gamma_chosen_recall_valid_list = []
    gamma_chosen_recall_test_list = []   
    
    f, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True) 
    for ii, bound_model_dir in enumerate(new_bounds_dirs):
        sl_rand_init_direct_min_precision_filename_aka = 'skorch_logistic_regression*surrogate_loss_tight*warm_start=false*incremental_min_precision=false*history.json'


        precisions_train_direct_min_precision, precisions_valid_direct_min_precision, precisions_test_direct_min_precision, recalls_train_direct_min_precision, recalls_valid_direct_min_precision, recalls_test_direct_min_precision = get_all_precision_recalls(bound_model_dir, sl_rand_init_direct_min_precision_filename_aka.replace('history.json', '.csv'))
    
        gamma_precision_train_list.append(precisions_train_direct_min_precision)
        gamma_precision_valid_list.append(precisions_valid_direct_min_precision)
        gamma_precision_test_list.append(precisions_test_direct_min_precision)
        
        gamma_recall_train_list.append(recalls_train_direct_min_precision)
        gamma_recall_valid_list.append(recalls_valid_direct_min_precision)
        gamma_recall_test_list.append(recalls_test_direct_min_precision)
        
        keep_inds = (precisions_train_direct_min_precision>=0.7)&(precisions_valid_direct_min_precision>=0.6)
        best_ind = np.argmax(recalls_valid_direct_min_precision[keep_inds])
        
        gamma_chosen_precision_train_list.append(precisions_train_direct_min_precision[keep_inds][best_ind])
        gamma_chosen_precision_valid_list.append(precisions_valid_direct_min_precision[keep_inds][best_ind])
        gamma_chosen_precision_test_list.append(precisions_test_direct_min_precision[keep_inds][best_ind])
        
        gamma_chosen_recall_train_list.append(recalls_train_direct_min_precision[keep_inds][best_ind])
        gamma_chosen_recall_valid_list.append(recalls_valid_direct_min_precision[keep_inds][best_ind])
        gamma_chosen_recall_test_list.append(recalls_test_direct_min_precision[keep_inds][best_ind])        
        
        
#         # plot the precision recalls for each bound
#         axs[0].plot(precisions_train_direct_min_precision, recalls_train_direct_min_precision, '.', label=gamma_list[ii])
#         axs[]
#         axs[1].plot(precisions_valid_direct_min_precision, recalls_valid_direct_min_precision, '.', label=gamma_list[ii])
#         axs[2].plot(precisions_test_direct_min_precision, recalls_test_direct_min_precision, '.', label=gamma_list[ii])

    for jj, (split, precisions_per_gamma, recalls_per_gamma,
             best_precision_per_gamma, best_recall_per_gamma) in enumerate([('train', 
                                                                            gamma_precision_train_list,
                                                                            gamma_recall_train_list,
                                                                            gamma_chosen_precision_train_list,
                                                                            gamma_chosen_recall_train_list),
                                                                           ('validation',
                                                                            gamma_precision_valid_list,
                                                                            gamma_recall_valid_list,
                                                                            gamma_chosen_precision_valid_list,
                                                                            gamma_chosen_recall_valid_list),
                                                                           ('test',
                                                                            gamma_precision_test_list,
                                                                            gamma_recall_test_list,
                                                                            gamma_chosen_precision_test_list,
                                                                            gamma_chosen_recall_test_list)]):

        for kk in range(len(gamma_list)):  
            axs[jj].plot(precisions_per_gamma[kk], recalls_per_gamma[kk], '.',  
                     label=gamma_list[kk], color=gamma_colors[kk], alpha=0.6)  
            axs[jj].plot(best_precision_per_gamma[kk], best_recall_per_gamma[kk], 'x', markersize=20, 
                         color=gamma_colors[kk], alpha=0.6)  
            
            axs[jj].set_xlabel('Precision') 
            axs[jj].set_ylabel('Recall') 
            if jj == 1:
                axs[jj].legend(bbox_to_anchor=(.4, .7), fontsize=8.5)   
            axs[jj].set_title('Precision vs Recall on %s set'%(split))
            axs[jj].set_xticks(np.arange(0, 1.1, 0.1))
            axs[jj].set_xlim([0., 1.0])
            axs[jj].set_ylim([0., 0.4])
            
    f.savefig('precision_recalls_multiple_bounds_new.png')
    
    
    from IPython import embed; embed()
        
    
    '''
    for ii, (method, prcs_train, recs_train, prcs_valid, recs_valid, prcs_test, recs_test) in enumerate([
        ('direct min precision', 
         precisions_train_direct_min_precision,
         recalls_train_direct_min_precision,
         precisions_valid_direct_min_precision,
         recalls_valid_direct_min_precision,
         precisions_test_direct_min_precision,
         recalls_test_direct_min_precision),
    ]):
        
        min_prec_tr = 0.7
        min_prec_va = 0.6
        keep_inds = (prcs_train>min_prec_tr)&(prcs_valid>min_prec_va)
        fracs_above_min_precision = (keep_inds).sum()/len(prcs_train)
        prcs_train = prcs_train[keep_inds]
        prcs_valid = prcs_valid[keep_inds]
        prcs_test = prcs_test[keep_inds]        
        recs_train = recs_train[keep_inds]
        recs_valid = recs_valid[keep_inds]
        recs_test = recs_test[keep_inds]        
        
        best_ind = np.argmax(recs_valid)
        
#         max_recall = max(recs[keep_inds])
        print('\nMethod - %s'%method)
        print('=================================================')
        print('Frac hypers achieving above %.4f on training set : %.5f'%(min_prec_tr, fracs_above_min_precision))
        print('Precision on train/valid/test with best model :')
        print('--------------------------------------------------')
        print('Train : %.5f'%prcs_train[best_ind])
        print('Valid : %.5f'%prcs_valid[best_ind])
        print('Test : %.5f'%prcs_test[best_ind])
        print('Recall on train/valid/test with best model :')
        print('--------------------------------------------------')
        print('Train : %.5f'%recs_train[best_ind])
        print('Valid : %.5f'%recs_valid[best_ind])
        print('Test : %.5f'%recs_test[best_ind])       
    
    
    
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
        precisions_train_per_batch_size_direct_min_precision, precisions_valid_per_batch_size_direct_min_precision, precisions_test_per_batch_size_direct_min_precision, recalls_train_per_batch_size_direct_min_precision, recalls_valid_per_batch_size_direct_min_precision, recalls_test_per_batch_size_direct_min_precision = get_all_precision_recalls(clf_models_dir, curr_batch_size_fnames_aka.replace('history.json', '.csv'))
        
        precisions_train_per_batch_size.append(precisions_train_per_batch_size_direct_min_precision)
        recalls_train_per_batch_size.append(recalls_train_per_batch_size_direct_min_precision)
        precisions_valid_per_batch_size.append(precisions_valid_per_batch_size_direct_min_precision)
        recalls_valid_per_batch_size.append(recalls_valid_per_batch_size_direct_min_precision)
        precisions_test_per_batch_size.append(precisions_test_per_batch_size_direct_min_precision)
        recalls_test_per_batch_size.append(recalls_test_per_batch_size_direct_min_precision)
        
        
    
    f, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True) 
    
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
            axs[jj].set_xlabel('Precision') 
            axs[jj].set_ylabel('Recall') 
            if jj == 1:
                axs[jj].legend(bbox_to_anchor=(.8, .4), fontsize=10)   
            axs[jj].set_title('Precision vs Recall on %s set'%(split))
            axs[jj].set_xticks(np.arange(0, 1.1, 0.1))
            axs[jj].set_xlim([0., 1.0])
            
    f.savefig('precision_recalls_batch_sized.png') 
    
    
    
    from IPython import embed; embed()
    '''
    
    
 