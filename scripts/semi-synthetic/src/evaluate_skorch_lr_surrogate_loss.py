'''
Evaluate performance of a single trained classifier on multiple patient-stay-slices
'''
import os
import numpy as np
import pandas as pd
from joblib import dump, load
import sys
sys.path.append(os.path.join(os.path.abspath('../'), 'src'))

import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score)
import ast
import random
import pickle
import glob
import argparse
import json
import seaborn as sns

def get_best_model(clf_models_dir, filename_aka):
    ''' Get the best model from training history'''
    
    training_files = glob.glob(os.path.join(clf_models_dir, filename_aka))
    valid_losses_np = np.zeros(len(training_files))
    precision_valid_np = np.zeros(len(training_files))
    recall_valid_np = np.zeros(len(training_files))
    precision_train_np = np.zeros(len(training_files))
    
    for i, f in enumerate(training_files):
        training_hist_df = pd.DataFrame(json.load(open(f)))
        
        # get the model with lowest validation loss 
        valid_losses_np[i] = training_hist_df.valid_loss.values[-1]
        precision_valid_np[i] = training_hist_df.precision_valid.values[-1]
        recall_valid_np[i] = training_hist_df.recall_valid.values[-1]
        precision_train_np[i] = training_hist_df.precision_train.values[-1]
        
    precision_valid_np[np.isnan(precision_valid_np)]=0
    precision_train_np[np.isnan(precision_train_np)]=0
    recall_valid_np[np.isnan(recall_valid_np)]=0
    best_model_ind = np.argmax(recall_valid_np)
    
    return training_files[best_model_ind]
    

def get_best_model_after_threshold_search(clf_models_dir, filename_aka, return_all_precision_recalls=False):
    ''' Get the best model from training history'''
    training_files = glob.glob(os.path.join(clf_models_dir, filename_aka))
    precision_train_np = np.zeros(len(training_files))
    recall_train_np = np.zeros(len(training_files))
    precision_valid_np = np.zeros(len(training_files))
    recall_valid_np = np.zeros(len(training_files))
    
    for i, f in enumerate(training_files):
        training_hist_df = pd.read_csv(f)
        precision_train_np[i] = training_hist_df.precision_train.values[-1]
        recall_train_np[i] = training_hist_df.recall_train.values[-1]
        precision_valid_np[i] = training_hist_df.precision_valid.values[-1]
        recall_valid_np[i] = training_hist_df.recall_valid.values[-1]
        
    precision_train_unfiltered = precision_train_np
    precision_valid_unfiltered = precision_valid_np
    recall_train_unfiltered = recall_train_np
    recall_valid_unfiltered = recall_valid_np
    
# #     get model with max recall at precision >=0.8
    keep_inds = precision_train_np>=0.75
    if keep_inds.sum()>0:
        training_files = np.array(training_files)[keep_inds]
        precision_train_np = precision_train_np[keep_inds]
        recall_train_np = recall_train_np[keep_inds]
        precision_valid_np = precision_valid_np[keep_inds]
        recall_valid_np = recall_valid_np[keep_inds]
    

    best_model_ind = np.argmax(recall_valid_np)
    
    if return_all_precision_recalls:
        return training_files[best_model_ind], precision_train_unfiltered, precision_valid_unfiltered, recall_train_unfiltered, recall_valid_unfiltered
    else:
        return training_files[best_model_ind]


    
def plot_best_model_training_plots(best_model_history_file, plt_name):
    
    metrics = ['precision', 'recall', 'bce_loss', 'surr_loss', 'fpu_bound', 'tpl_bound']
    training_hist_df = pd.DataFrame(json.load(open(best_model_history_file))) 
    f, axs = plt.subplots(len(metrics), 1, figsize=(8,8), sharex=True)
    
    for i, metric in enumerate(metrics): 
        # plot epochs vs precision on train and validation
        if (metric == 'fpu_bound'):
            axs[i].plot(training_hist_df.epoch, training_hist_df['fpu_bound_train'], color='r', label='FP upper bound')
            axs[i].plot(training_hist_df.epoch, training_hist_df['fp_train'], color='b', label='FP train')
        elif (metric == 'tpl_bound'):
            axs[i].plot(training_hist_df.epoch, training_hist_df['tpl_bound_train'], color='r', label='TP lower bound')
            axs[i].plot(training_hist_df.epoch, training_hist_df['tp_train'], color='b', label='TP train')            
        else:
            try:
                axs[i].plot(training_hist_df.epoch, training_hist_df['%s_train'%metric], color='b', label='%s(train)'%metric)
#                 axs[i].plot(training_hist_df.epoch, training_hist_df['%s_valid'%metric], color='k', label='%s(validation)'%metric) 
#                 axs[i].set_ylim([0.1, 1])
            except:
                axs[i].plot(training_hist_df.epoch, training_hist_df['train_%s'%metric], color='b', label='%s(train)'%metric)
#                 axs[i].plot(training_hist_df.epoch, training_hist_df['valid_%s'%metric], color='k', label='%s(validation)'%metric)             
            axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].grid(True)   
    axs[i].set_xlabel('epochs')
    plt.suptitle(plt_name)
    f.savefig(plt_name+'.png')
    
    
def plot_all_models_training_plots(clf_models_dir, all_models_history_files_aka, plt_name):
    
    metrics = ['precision', 'recall'] 
    f, axs = plt.subplots(len(metrics), 1, figsize=(8,8), sharex=True)
    sns.set_context("notebook", font_scale=1.25)
    alpha=0.3
    all_models_history_files = glob.glob(os.path.join(clf_models_dir, all_models_history_files_aka))
    for f_ind, model_history_file in enumerate(all_models_history_files):
        training_hist_df = pd.DataFrame(json.load(open(model_history_file))) 
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
          
                axs[i].set_ylabel(metric)
            if f_ind == 0:
                axs[i].legend(loc='upper left')
                axs[i].grid(True)   
    axs[i].set_xlabel('epochs')
    axs[i].set_xlim([0, 1000])
#     plt.suptitle(plt_name)
    f.savefig(plt_name+'.png')
    f.savefig(plt_name+'.pdf', bbox_inches='tight', pad_inches=0)
    
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
    # get the best model minimizining bce loss
    bce_filename_aka = 'skorch_logistic_regression*cross_entropy_loss*history.json'
    bce_plus_thresh_filename_aka = 'skorch_logistic_regression*cross_entropy_loss*perf.csv'

    best_model_file_bce_training_hist = get_best_model(clf_models_dir, bce_filename_aka)
    best_model_file_bce = best_model_file_bce_training_hist.replace('history.json', '.txt')

    # plot training plots of best model
    plot_best_model_training_plots(best_model_file_bce_training_hist, 'logistic_regression_minimizing_cross_entropy')
    
    best_model_file_bce_plus_thresh = get_best_model_after_threshold_search(clf_models_dir, bce_plus_thresh_filename_aka)


    # get the best model minimizining surrogate loss
    sl_with_bce_init_filename_aka = 'skorch_logistic_regression*surrogate_loss_tight*warm_start=true*history.json'
    sl_with_bce_init_csv_aka = 'skorch_logistic_regression*surrogate_loss_tight*warm_start=true*perf.csv'

    best_model_file_sl_with_bce_init = get_best_model_after_threshold_search(clf_models_dir, sl_with_bce_init_csv_aka)
    best_model_file_training_hist_sl_with_bce_init = best_model_file_sl_with_bce_init.replace('_perf.csv', 'history.json')
    
    
    # plot training plots of best model
    plot_best_model_training_plots(best_model_file_training_hist_sl_with_bce_init, 'logistic_regression_minimizing_surrogate_loss_tight_bce_init')
    '''
    
    # get the best model minimizining surrogate loss
    sl_rand_init_filename_aka = 'skorch_logistic_regression*surrogate_loss_tight*warm_start=false*history.json'
    sl_rand_init_csv_aka = 'skorch_logistic_regression*surrogate_loss_tight*warm_start=false*perf.csv'

    best_model_file_sl_rand_init, precision_train_np, precision_valid_np, recall_train_np, recall_valid_np = get_best_model_after_threshold_search(clf_models_dir, sl_rand_init_csv_aka, return_all_precision_recalls=True)
    best_model_file_training_hist_sl_rand_init = best_model_file_sl_rand_init.replace('_perf.csv', 'history.json')
    
    
    # plot training plots of best model
    plot_best_model_training_plots(best_model_file_training_hist_sl_rand_init, 'logistic_regression_minimizing_surrogate_loss_tight_rand_init')
    
    plot_all_models_training_plots(clf_models_dir, sl_rand_init_filename_aka, 'logistic_regression_minimizing_surrogate_loss_tight_rand_init_all_models')
    
    # get the best model minimizining surrogate loss
    sl_loose_rand_init_filename_aka = 'skorch_logistic_regression*surrogate_loss_loose*warm_start=false*history.json'
    sl_loose_rand_init_csv_aka = 'skorch_logistic_regression*surrogate_loss_loose*warm_start=false*perf.csv'

    best_model_file_sl_loose_rand_init, precision_train_loose_np, _, recall_train_loose_np, _ = get_best_model_after_threshold_search(clf_models_dir, sl_loose_rand_init_csv_aka, return_all_precision_recalls=True)
    best_model_file_training_hist_sl_loose_rand_init = best_model_file_sl_loose_rand_init.replace('_perf.csv', 'history.json')
    
    
    # plot training plots of best model
    plot_best_model_training_plots(best_model_file_training_hist_sl_loose_rand_init, 'logistic_regression_minimizing_surrogate_loss_loose_rand_init')
    
    plot_all_models_training_plots(clf_models_dir, sl_loose_rand_init_filename_aka, 'logistic_regression_minimizing_surrogate_loss_loose_rand_init_all_models')
    '''
    print('Performance of LR minimizing BCE :')
    f = open(best_model_file_bce, "r")
    print(f.read())
    bce_plus_thresh_perf_df = pd.read_csv(best_model_file_bce_plus_thresh)
    print('Performance of LR minimizing BCE + threshold search:')
    print(bce_plus_thresh_perf_df) 
    sl_with_bce_init_perf_df = pd.read_csv(best_model_file_sl_with_bce_init)
    print('Performance of LR with surrogate objective and warm start with BCE:')
    print(sl_with_bce_init_perf_df)
    '''
    sl_with_rand_init_perf_df = pd.read_csv(best_model_file_sl_rand_init)
    print('Performance of LR with surrogate objective (tight)and random init:')
    print(sl_with_rand_init_perf_df)
    
    sl_loose_with_rand_init_perf_df = pd.read_csv(best_model_file_sl_loose_rand_init)
    print('Performance of LR with surrogate objective (loose) and random init:')
    print(sl_loose_with_rand_init_perf_df)
    
    
    # plot he boxplot of precision recall
    f, axs = plt.subplots(1, 1, figsize=(8,8))
    sns.set_context("notebook", font_scale=1.25)
    xticks = [0.5, 1.5, 3.5, 4.5]
    axs.boxplot([precision_train_np, precision_valid_np, recall_train_np, recall_valid_np], positions=xticks, widths=(0.3, 0.3, 0.3, 0.3)) 
    
    xticklabels = ['precision_train', 'precision_valid', 'recall_train', 'recall_valid']
    axs.set_xticks(xticks)
    axs.set_xticklabels(xticklabels, rotation=20) 
    axs.set_title('Semi-synthetic Dataset Precision and Recall Over all Hyperparameters', fontsize=14)
    f.savefig('precision_recall_boxplot.png')  
    f.savefig('precision_recall_boxplot.pdf', bbox_inches='tight', pad_inches=0)
    
    # plot he boxplot of precision recall
    f, axs = plt.subplots(1, 1, figsize=(8,8))
    axs.boxplot([precision_train_loose_np, recall_train_loose_np]) 
    xticks = [1, 2] 
    xticklabels = ['precision', 'recall']
    axs.set_xticks(xticks)
    axs.set_xticklabels(xticklabels) 
    axs.set_title('Semi-synthetic Dataset Precision and Recall Over all Hyperparameters', fontsize=14)
    f.savefig('precision_recall_loose_boxplot.png') 
        
 