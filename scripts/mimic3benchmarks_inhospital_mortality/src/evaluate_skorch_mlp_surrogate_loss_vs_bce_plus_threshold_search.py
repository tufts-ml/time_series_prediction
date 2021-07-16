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
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchMLP'))
from sklearn.preprocessing import StandardScaler
from SkorchMLP import SkorchMLP

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
    

def get_best_model_after_threshold_search(clf_models_dir, filename_aka):
    ''' Get the best model from training history'''
    training_files = glob.glob(os.path.join(clf_models_dir, filename_aka))
    valid_losses_np = np.zeros(len(training_files))
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
    precision_valid_np[np.isnan(precision_valid_np)]=0
    recall_valid_np[np.isnan(recall_valid_np)]=0
    
#     from IPython import embed; embed()
#     get model with max recall at precision >=0.9
    keep_inds = precision_train_np>=0.9
    training_files = np.array(training_files)[keep_inds]
    precision_valid_np = precision_valid_np[keep_inds]
    recall_valid_np = recall_valid_np[keep_inds]
    
    best_model_ind = np.argmax(recall_valid_np)
#     best_model_ind = np.argmin(valid_losses_np)
    
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
                axs[i].plot(training_hist_df.epoch, training_hist_df['%s_valid'%metric], color='k', label='%s(validation)'%metric) 
#                 axs[i].set_ylim([0.1, 1])
            except:
                axs[i].plot(training_hist_df.epoch, training_hist_df['train_%s'%metric], color='b', label='%s(train)'%metric)
                axs[i].plot(training_hist_df.epoch, training_hist_df['valid_%s'%metric], color='k', label='%s(validation)'%metric)             
            axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].grid(True)   
    axs[i].set_xlabel('epochs')
    plt.suptitle(plt_name)
    f.savefig(plt_name+'.png')
#     from IPython import embed; embed()

def plot_all_models_training_plots(clf_models_dir, all_models_history_files_aka, plt_name):
    
    metrics = ['precision', 'recall'] 
    f, axs = plt.subplots(len(metrics), 1, figsize=(8,8), sharex=True)
    sns.set_context("notebook", font_scale=1.5)
    alpha=0.3
    all_models_history_files = glob.glob(os.path.join(clf_models_dir, all_models_history_files_aka))

    for f_ind, model_history_file in enumerate(all_models_history_files):
        training_hist_df = pd.DataFrame(json.load(open(model_history_file))) 
        
        if training_hist_df['precision_train'].values[-1]>0.9:
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
    axs[i].set_xlabel('epochs', fontsize=15)
    axs[i].set_xlim([0, 200])
#     plt.suptitle(plt_name)
    f.savefig(plt_name+'.png')
    f.savefig(plt_name+'.pdf', bbox_inches='tight', pad_inches=0)
    
def plot_all_models_precision_recall(clf_models_dir, all_models_history_files_aka, plt_name):
    
    metrics = ['precision', 'recall'] 
    f, axs = plt.subplots(len(metrics), 1, figsize=(8,8), sharex=True)
    sns.set_context("notebook", font_scale=1.5)
    alpha=0.3
    all_models_history_files = glob.glob(os.path.join(clf_models_dir, all_models_history_files_aka))
        
    precision_train, precision_valid, recall_train, recall_valid = [[], [], [], []]
    for f_ind, model_history_file in enumerate(all_models_history_files):
        training_hist_df = pd.DataFrame(json.load(open(model_history_file))) 
        precision_train.append(training_hist_df['precision_train'].values)
        precision_valid.append(training_hist_df['precision_valid'].values)
        recall_train.append(training_hist_df['recall_train'].values)
        recall_valid.append(training_hist_df['recall_valid'].values)
    
    precision_train_np = np.stack(precision_train)
    precision_valid_np = np.stack(precision_valid)  
    recall_train_np = np.stack(recall_train)
    recall_valid_np = np.stack(recall_valid) 
    epochs = training_hist_df.epoch.values
    
    keep_inds = (precision_train_np[:,-1]>0.9)&(precision_valid_np[:,-1]>0.8)
    keep_indices = np.flatnonzero(keep_inds)
    best_index = np.argmax(recall_valid_np[keep_indices, -1])
    best_ind = keep_indices[best_index]
    
    yticks = np.arange(0, 1.1, 0.2)
    yticklabels = ['%.2f'%ii for ii in yticks]
    fontsize=16
    
    axs[0].plot(epochs, precision_train_np[keep_inds].T, color="0.8")
    axs[0].plot(epochs, precision_valid_np[keep_inds].T, color="0.8")
    axs[0].plot(epochs, precision_train_np[best_ind].T, color="r", label='train (best model)')
    axs[0].plot(epochs, precision_valid_np[best_ind].T, color="b", label='valid (best model)')
    axs[0].set_ylabel('precision', fontsize=fontsize)
    axs[0].grid(True)
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels(yticklabels)
    axs[0].legend(prop={'size': 12})
    
    axs[1].plot(epochs, recall_train_np[keep_inds].T, color="0.8")
    axs[1].plot(epochs, recall_valid_np[keep_inds].T, color="0.8")
    axs[1].plot(epochs, recall_train_np[best_ind].T, color="r", label='train (best model)')
    axs[1].plot(epochs, recall_valid_np[best_ind].T, color="b", label='valid (best_model)')    
    axs[1].set_xlabel('epochs', fontsize=fontsize)
    axs[1].set_ylabel('recall', fontsize=fontsize)
    axs[1].set_xlim([0, 200])
    axs[1].grid(True)
    axs[1].set_yticks(yticks)
    axs[1].set_yticklabels(yticklabels)
    axs[1].legend(prop={'size': 12})
    
    f.savefig(plt_name+'.png')
    f.savefig(plt_name+'.pdf', bbox_inches='tight', pad_inches=0)
    
    from IPython import embed; embed()
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
    
    
# # #     get model with max recall at precision >=0.8
#     keep_inds = precision_train_np>=0.75
#     if keep_inds.sum()>0:
#         training_files = np.array(training_files)[keep_inds]
#         precision_train_np = precision_train_np[keep_inds]
#         recall_train_np = recall_train_np[keep_inds]
#         precision_valid_np = precision_valid_np[keep_inds]
#         recall_valid_np = recall_valid_np[keep_inds]
    

#     best_model_ind = np.argmax(recall_valid_np)
    

    return precision_train_unfiltered, precision_valid_unfiltered, precision_test_unfiltered, recall_train_unfiltered, recall_valid_unfiltered, recall_test_unfiltered
  
def make_precision_recall_boxplots(precision_train_np, precision_valid_np, precision_test_np, recall_train_np, recall_valid_np, recall_test_np, plt_name, title_str=''):
    # plot he boxplot of precision recall
    f, axs = plt.subplots(1, 1, figsize=(8,8))
    sns.set_context("notebook", font_scale=1.25)
    xticks = [0.5, 1.0, 1.5, 3.5, 4.0, 4.5]
    axs.boxplot([precision_train_np, precision_valid_np, precision_test_np, recall_train_np, recall_valid_np, recall_test_np], positions=xticks, widths=(0.3, 0.3, 0.3, 0.3, 0.3, 0.3)) 
    
    xticklabels = ['precision_train', 'precision_valid', 'precision_test', 'recall_train', 'recall_valid', 'recall_test']
    axs.set_xticks(xticks)
    axs.set_xticklabels(xticklabels, rotation=20) 
    
    yticks = np.arange(0, 1.1, 0.1)
    yticklabels = ['%.2f'%ii for ii in yticks]
    axs.set_yticks(yticks)
    axs.set_yticklabels(yticklabels) 
    axs.set_ylim([0, 1])
    axs.grid(True)
    axs.set_title('Precision and Recall Over all Hyperparameters '+ title_str , fontsize=14)
    f.savefig(plt_name+'.png')  
#     f.savefig('precision_recall_boxplot.pdf', bbox_inches='tight', pad_inches=0)


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
    
#     # get performance metrics
    x_test = x_test_df[feature_cols].values
    y_test = y_test_df[outcome_col].values

    # load the scaler
    scaler = pickle.load(open(os.path.join(clf_models_dir, 'scaler.pkl'), 'rb'))
    x_test_transformed = scaler.transform(x_test)
    '''
    # get the best model minimizining bce loss
    bce_filename_aka = 'skorch_mlp*cross_entropy_loss*n_hiddens=32*history.json'
    bce_plus_thresh_filename_aka = 'skorch_mlp*cross_entropy_loss*n_hiddens=32*perf.csv'

    best_model_file_bce = get_best_model(clf_models_dir, bce_filename_aka)
    best_model_file_bce = best_model_file_bce.replace('history.json', '.txt')
    
    
    best_model_file_bce_plus_thresh = get_best_model_after_threshold_search(clf_models_dir, bce_plus_thresh_filename_aka)

    # plot training plots of best model
    plot_best_model_training_plots(best_model_file_bce_plus_thresh.replace('_perf.csv', 'history.json'), 'MLP_minimizing_cross_entropy')


    # get the best model minimizining surrogate loss
    sl_filename_aka = 'skorch_mlp*surrogate_loss_tight*n_hiddens=32*history.json'
    sl_csv_aka = 'skorch_mlp*surrogate_loss_tight*n_hiddens=32*perf.csv'

#             best_model_file_sl = get_best_model(clf_models_dir, sl_filename_aka)
    best_model_file_sl = get_best_model_after_threshold_search(clf_models_dir, sl_csv_aka)
    best_model_file_training_hist = best_model_file_sl.replace('_perf.csv', 'history.json')

    # plot training plots of best model
    plot_best_model_training_plots(best_model_file_training_hist, 'MLP_minimizing_surrogate_loss_tight')

    # print the bce , bce+threshold search, surrogate loss results
#     bce_perf_df = pd.read_csv(best_model_file_bce)
    print('Performance of MLP minimizing BCE :')
#     print(bce_perf_df)
    f = open(best_model_file_bce, "r")
    print(f.read())
    bce_plus_thresh_perf_df = pd.read_csv(best_model_file_bce_plus_thresh)
    print('Performance of MLP minimizing BCE + threshold search:')
    print(bce_plus_thresh_perf_df) 
    sl_perf_df = pd.read_csv(best_model_file_sl)
    print('Performance of MLP with surrogate objective:')
    print(sl_perf_df)
    '''
    
    sl_rand_init_incremental_min_precision_filename_aka = 'skorch_mlp*surrogate_loss_tight*warm_start=false*incremental_min_precision=true*history.json'
    
    plot_all_models_precision_recall(clf_models_dir, sl_rand_init_incremental_min_precision_filename_aka, 
                                     'skorch_mlp_incremental_min_precision')
    
#     plot_all_models_training_plots(clf_models_dir, sl_rand_init_incremental_min_precision_filename_aka, 'skorch_mlp_incremental_min_precision')

    precisions_train_incremental_min_precision, precisions_valid_incremental_min_precision, precisions_test_incremental_min_precision, recalls_train_incremental_min_precision, recalls_valid_incremental_min_precision,  recalls_test_incremental_min_precision= get_all_precision_recalls(clf_models_dir, sl_rand_init_incremental_min_precision_filename_aka.replace('history.json', '.csv'))
    
    make_precision_recall_boxplots(precisions_train_incremental_min_precision, precisions_valid_incremental_min_precision, precisions_test_incremental_min_precision, recalls_train_incremental_min_precision, recalls_valid_incremental_min_precision, recalls_test_incremental_min_precision, 'mlp_precision_recall_boxplot_incremental_min_precision', '(Ramp Up)')
    
    sl_rand_init_direct_min_precision_filename_aka = 'skorch_mlp*surrogate_loss_tight*warm_start=false*incremental_min_precision=false*history.json'
    
#     plot_all_models_training_plots(clf_models_dir, sl_rand_init_direct_min_precision_filename_aka, 'skorch_mlp_direct_min_precision')
    
    precisions_train_direct_min_precision, precisions_valid_direct_min_precision, precisions_test_direct_min_precision, recalls_train_direct_min_precision, recalls_valid_direct_min_precision, recalls_test_direct_min_precision = get_all_precision_recalls(clf_models_dir, sl_rand_init_direct_min_precision_filename_aka.replace('history.json', '.csv'))
    
   
    make_precision_recall_boxplots(precisions_train_direct_min_precision, precisions_valid_direct_min_precision, precisions_test_direct_min_precision, recalls_train_direct_min_precision, recalls_valid_direct_min_precision, recalls_test_direct_min_precision, 'mlp_precision_recall_boxplot_direct_min_precision', '(Direct)')
  
    sl_bce_perturb_filename_aka = 'skorch_mlp*surrogate_loss_tight*warm_start=true*history.json'
    
#     plot_all_models_training_plots(clf_models_dir, sl_bce_perturb_filename_aka, 'skorch_mlp_bce_perturb')
    
    precisions_train_bce_perturb, precisions_valid_bce_perturb, precisions_test_bce_perturb, recalls_train_bce_perturb, recalls_valid_bce_perturb, recalls_test_bce_perturb = get_all_precision_recalls(clf_models_dir, sl_bce_perturb_filename_aka.replace('history.json', '.csv'))
    
    
    make_precision_recall_boxplots(precisions_train_bce_perturb, precisions_valid_bce_perturb, precisions_test_bce_perturb, recalls_train_bce_perturb, recalls_valid_bce_perturb, recalls_test_bce_perturb, 'mlp_precision_recall_boxplot_bce_perturb', '(BCE + Perturbation)')
    

    bce_plus_thresh_filename_aka = 'skorch_mlp*cross_entropy_loss*warm_start=false*history.json'
    
#     plot_all_models_training_plots(clf_models_dir, bce_plus_thresh_filename_aka, 'skorch_mlp_bce_plus_thresh')
    
    precisions_train_bce_plus_thresh, precisions_valid_bce_plus_thresh, precisions_test_bce_plus_thresh, recalls_train_bce_plus_thresh, recalls_valid_bce_plus_thresh, recalls_test_bce_plus_thresh = get_all_precision_recalls(clf_models_dir, bce_plus_thresh_filename_aka.replace('history.json', '.csv'))
    
    
    make_precision_recall_boxplots(precisions_train_bce_plus_thresh, precisions_valid_bce_plus_thresh, precisions_test_bce_plus_thresh, recalls_train_bce_plus_thresh, recalls_valid_bce_plus_thresh, recalls_test_bce_plus_thresh, 'lr_precision_recall_boxplot_bce_plus_thresh', '(BCE + Threshold Search)')

    
    sl_loose_filename_aka = 'skorch_mlp*surrogate_loss_loose*warm_start=false*history.json'
    
#     plot_all_models_training_plots(clf_models_dir, sl_loose_filename_aka, 'skorch_mlp_sl_loose')
    
    precisions_train_sl_loose, precisions_valid_sl_loose, precisions_test_sl_loose, recalls_train_sl_loose, recalls_valid_sl_loose, recalls_test_sl_loose = get_all_precision_recalls(clf_models_dir, sl_loose_filename_aka.replace('history.json', '.csv'))
    
    
    make_precision_recall_boxplots(precisions_train_sl_loose, precisions_valid_sl_loose, precisions_test_sl_loose, recalls_train_sl_loose, recalls_valid_sl_loose, recalls_test_sl_loose, 'mlp_precision_recall_boxplot_sl_loose', '(Surrogate Loss Hinge Bound)')

    for ii, (method, prcs_train, recs_train, prcs_valid, recs_valid, prcs_test, recs_test) in enumerate([
        ('direct min precision', 
         precisions_train_direct_min_precision,
         recalls_train_direct_min_precision,
         precisions_valid_direct_min_precision,
         recalls_valid_direct_min_precision,
         precisions_test_direct_min_precision,
         recalls_test_direct_min_precision),
        ('incremental min precision',
         precisions_train_incremental_min_precision,
         recalls_train_incremental_min_precision,
         precisions_valid_incremental_min_precision,
         recalls_valid_incremental_min_precision,
         precisions_test_incremental_min_precision,
         recalls_test_incremental_min_precision),
        ('bce + perturbaton',
         precisions_train_bce_perturb,
         recalls_train_bce_perturb,
         precisions_valid_bce_perturb,
         recalls_valid_bce_perturb,
         precisions_test_bce_perturb,         
         recalls_test_bce_perturb),
        ('bce + threshold search',
         precisions_train_bce_plus_thresh,
         recalls_train_bce_plus_thresh,
         precisions_valid_bce_plus_thresh,
         recalls_valid_bce_plus_thresh, 
         precisions_test_bce_plus_thresh,
         recalls_test_bce_plus_thresh),
        ('Surrogate Loss (Hinge Bound)',
         precisions_train_sl_loose,
         recalls_train_sl_loose,
         precisions_valid_sl_loose,
         recalls_valid_sl_loose, 
         precisions_test_sl_loose,
         recalls_test_sl_loose)
    ]):
        
        keep_inds = (prcs_train>0.9)&(prcs_valid>0.8)
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
        print('Frac hypers achieving above 0.9 on training set : %.5f'%(fracs_above_min_precision))
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
        
    from IPython import embed; embed()
    
 