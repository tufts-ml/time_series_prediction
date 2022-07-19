'''
Evaluate performance of a single trained classifier on multiple patient-stay-slices
'''
import os
import numpy as np
import pandas as pd
from joblib import dump, load
import sys
sys.path.append(os.path.join(os.path.abspath('../'), 'src'))

from collections import OrderedDict
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
  

    
def plot_all_models_training_plots(clf_models_dir, all_models_history_files_aka, plt_name):
    
    metrics = ['precision', 'recall', 'auprc', 'auroc', 'loss']
    f, axs = plt.subplots(len(metrics), 1, figsize=(8,8), sharex=True)
    sns.set_context("notebook", font_scale=1.5)
    alpha=0.3
    all_models_history_files = glob.glob(os.path.join(clf_models_dir, all_models_history_files_aka))

    for f_ind, model_history_file in enumerate(all_models_history_files):
        training_hist_df = pd.DataFrame(json.load(open(model_history_file))) 
        try:
            if training_hist_df['precision_train'].values[-1]>0.7:
                for i, metric in enumerate(metrics): 
                    # plot epochs vs precision on train and validation
                    if (metric == 'fpu_bound'):
                        axs[i].plot(training_hist_df.epoch, training_hist_df['fpu_bound_train'], color='r', 
                                    label='FP upper bound', alpha=alpha)
                        axs[i].plot(training_hist_df.epoch, training_hist_df['fp_train'], color='b', label='FP train', alpha=alpha)
                    elif (metric == 'tpl_bound'):
                        axs[i].plot(training_hist_df.epoch, training_hist_df['tpl_bound_train'], color='r', 
                                    label='TP lower bound', alpha=alpha)
                        axs[i].plot(training_hist_df.epoch, training_hist_df['tp_train'], color='b', label='TP train', alpha=alpha)            
                    else:
                        try:
                            axs[i].plot(training_hist_df.epoch, training_hist_df['%s_train'%metric], color='b', 
                                        label='train', alpha=alpha)
                            axs[i].plot(training_hist_df.epoch, training_hist_df['%s_valid'%metric], color='r', 
                                        label='valid', alpha=alpha)
                            if (metric=='precision')|(metric=='recall'):
                                yticks = np.arange(0, 1.1, 0.2)
                                yticklabels = ['%.2f'%ii for ii in yticks]
                                axs[i].set_yticks(yticks)
                                axs[i].set_yticklabels(yticklabels)

                        except:
                            axs[i].plot(training_hist_df.epoch, training_hist_df['train_%s'%metric], color='b', 
                                        label='%s(train)'%metric, alpha=alpha)
                            axs[i].plot(training_hist_df.epoch, training_hist_df['valid_%s'%metric], color='r', 
                                        label='%s(valid)'%metric, alpha=alpha)

                        axs[i].set_ylabel(metric, fontsize=16)
                    if f_ind == 0:
                        axs[i].legend(loc='lower right', fontsize=13)
                        axs[i].grid(True)   
        except:
            continue
            
    for i in range(len(metrics)):
        handles, labels = axs[i].get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        axs[i].legend(by_label.values(), by_label.keys())
    axs[i].set_xlabel('epochs', fontsize=15)
    axs[i].set_xlim([0, 200])
#     plt.suptitle(plt_name)
    f.savefig(plt_name+'.png')    
    
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

    
    
#     bound_model_dirs = glob.glob(os.path.join(clf_models_dir, '*gamma_fp=*')) 
    
    precision_train_list = []
    precision_valid_list = []
    precision_test_list = []
    
    recall_train_list = []
    recall_valid_list = []
    recall_test_list = []    
    
    chosen_precision_train_list = []
    chosen_precision_valid_list = []
    chosen_precision_test_list = []
    
    chosen_recall_train_list = []
    chosen_recall_valid_list = []
    chosen_recall_test_list = [] 
    
    
    
    bce_plus_threshold_filename_aka = 'rnn_per_tstep*cross_entropy_loss*history.json'
    sl_rand_init_direct_min_precision_filename_aka = 'rnn_per_tstep*surrogate_loss_tight*history.json'
    
    methods_filenames_list = [bce_plus_threshold_filename_aka, sl_rand_init_direct_min_precision_filename_aka]
    methods_names_list = ['BCE + threshold search', 'Sigmoid Bounds']
    methods_colors = ['r', 'g']
    
    # plot the training plots for the rnn with surrogate loss
    plot_all_models_training_plots(clf_models_dir, sl_rand_init_direct_min_precision_filename_aka, 'rnn_surrogate_loss_training_plots')
    
    f, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True) 
    for ii, method_filename in enumerate(methods_filenames_list):


        precisions_train, precisions_valid, precisions_test, recalls_train, recalls_valid, recalls_test = get_all_precision_recalls(clf_models_dir, method_filename.replace('history.json', '.csv'))
    
        precision_train_list.append(precisions_train)
        precision_valid_list.append(precisions_valid)
        precision_test_list.append(precisions_test)
        
        recall_train_list.append(recalls_train)
        recall_valid_list.append(recalls_valid)
        recall_test_list.append(recalls_test)
        
        keep_inds = (precisions_train>=0.7)
        best_ind = np.argmax(recalls_valid[keep_inds])
        
        chosen_precision_train_list.append(precisions_train[keep_inds][best_ind])
        chosen_precision_valid_list.append(precisions_valid[keep_inds][best_ind])
        chosen_precision_test_list.append(precisions_test[keep_inds][best_ind])
        
        chosen_recall_train_list.append(recalls_train[keep_inds][best_ind])
        chosen_recall_valid_list.append(recalls_valid[keep_inds][best_ind])
        chosen_recall_test_list.append(recalls_test[keep_inds][best_ind])        

    for jj, (split, precisions, recalls,
             best_precision, best_recall) in enumerate([('train', 
                                                                            precision_train_list,
                                                                            recall_train_list,
                                                                            chosen_precision_train_list,
                                                                            chosen_recall_train_list),
                                                                           ('validation',
                                                                            precision_valid_list,
                                                                            recall_valid_list,
                                                                            chosen_precision_valid_list,
                                                                            chosen_recall_valid_list),
                                                                           ('test',
                                                                            precision_test_list,
                                                                            recall_test_list,
                                                                            chosen_precision_test_list,
                                                                            chosen_recall_test_list)]):

        for kk in range(len(methods_names_list)):  
            axs[jj].plot(precisions[kk], recalls[kk], '.',  
                     label=methods_names_list[kk], color=methods_colors[kk], alpha=0.6)  
            axs[jj].plot(best_precision[kk], best_recall[kk], 'x', markersize=20, 
                         color=methods_colors[kk], alpha=0.6)  
            
            axs[jj].set_xlabel('Precision') 
            axs[jj].set_ylabel('Recall') 
            if jj == 1:
                axs[jj].legend(bbox_to_anchor=(.4, .7), fontsize=8.5)   
            axs[jj].set_title('Precision vs Recall on %s set'%(split))
            axs[jj].set_xticks(np.arange(0, 1.1, 0.1))
            axs[jj].set_xlim([0., 1.0])
            axs[jj].set_ylim([0., 0.4])
            
            
      
    f.savefig('rnn_precision_recalls_sl_vs_bce_plus_threshold.png')
    
    from IPython import embed; embed()