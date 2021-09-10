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

#import LR model before importing other packages because joblib files act weird when certain packages are loaded
from feature_transformation import *
from merge_features_all_tslices import merge_data_dicts
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score)
from utils import load_data_dict_json
import ast
from filter_admissions_by_tslice import get_preprocessed_data
import random
import pickle
import glob
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchLogisticRegression'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchMLP'))
from sklearn.preprocessing import StandardScaler
from SkorchLogisticRegression import SkorchLogisticRegression
from SkorchMLP import SkorchMLP

def get_best_model(clf_models_dir, filename_aka):
    ''' Get the best model from training history'''
    
    training_files = glob.glob(os.path.join(clf_models_dir, filename_aka))
    valid_losses_np = np.zeros(len(training_files))
    precision_valid_np = np.zeros(len(training_files))
    recall_valid_np = np.zeros(len(training_files))
    
    for i, f in enumerate(training_files):
        training_hist_df = pd.DataFrame(json.load(open(f)))
        
        # get the model with lowest validation loss 
        valid_losses_np[i] = training_hist_df.valid_loss.values[-1]
        precision_valid_np[i] = training_hist_df.precision_valid.values[-1]
        recall_valid_np[i] = training_hist_df.recall_valid.values[-1]
    
    precision_valid_np[np.isnan(precision_valid_np)]=0
    recall_valid_np[np.isnan(recall_valid_np)]=0
    best_model_ind = np.argmax(recall_valid_np)
#     best_model_ind = np.argmin(valid_losses_np)
    
    
    return training_files[best_model_ind]
    

def get_best_model_after_threshold_search(clf_models_dir, filename_aka):
    ''' Get the best model from training history'''
    
    training_files = glob.glob(os.path.join(clf_models_dir, filename_aka))
    valid_losses_np = np.zeros(len(training_files))
    precision_valid_np = np.zeros(len(training_files))
    recall_valid_np = np.zeros(len(training_files))
    
    for i, f in enumerate(training_files):
        training_hist_df = pd.read_csv(f)
        
        # get the model with lowest validation loss 
        precision_valid_np[i] = training_hist_df.precision_train.values[-1]
        recall_valid_np[i] = training_hist_df.recall_train.values[-1]
    
    precision_valid_np[np.isnan(precision_valid_np)]=0
    recall_valid_np[np.isnan(recall_valid_np)]=0
    best_model_ind = np.argmax(recall_valid_np)
#     best_model_ind = np.argmin(valid_losses_np)
    
    
    return training_files[best_model_ind]    
    
    
def plot_best_model_training_plots(best_model_history_file, plt_name):
    
    metrics = ['precision', 'recall', 'loss']
    training_hist_df = pd.DataFrame(json.load(open(best_model_history_file))) 
    f, axs = plt.subplots(len(metrics), 1, figsize=(8,8), sharex=True)
    
    
    for i, metric in enumerate(metrics): 
        # plot epochs vs precision on train and validation
        try:
            axs[i].plot(training_hist_df.epoch, training_hist_df['%s_train'%metric], label='%s(train)'%metric)
            axs[i].plot(training_hist_df.epoch, training_hist_df['%s_valid'%metric], label='%s(validation)'%metric) 
            axs[i].set_ylim([0, 1])
        except:
            axs[i].plot(training_hist_df.epoch, training_hist_df['train_%s'%metric], label='%s(train)'%metric)
            axs[i].plot(training_hist_df.epoch, training_hist_df['valid_%s'%metric], label='%s(validation)'%metric)             
        axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].grid(True)   
    axs[i].set_xlabel('epochs')
    plt.suptitle(plt_name)
    f.savefig(plt_name+'.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_models_dir', default=None, type=str,
                        help='Directory where classifier models are saved')
    parser.add_argument('--clf_train_test_split_dir', default=None, type=str,
                        help='Directory where the train-test split data for the classifier is saved')
    parser.add_argument('--outcome_column_name', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    parser.add_argument('--random_seed_list', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    parser.add_argument('--output_dir', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    
    
    args = parser.parse_args()
    

    
    ## get the test patient id's
    # get the test set's csv and dict
    y_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_test.csv.gz'))
    x_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'x_test.csv.gz'))
    x_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'x_dict.json')
    y_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'y_dict.json')
    
    # import the y dict to get the id cols
    x_test_dict = load_data_dict_json(x_test_dict_file)
    y_test_dict = load_data_dict_json(y_test_dict_file)
    id_cols = parse_id_cols(y_test_dict)
    
    outcome_col = args.outcome_column_name
    y_test_ids_df = y_test_df[id_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)
    
    
    # get the bin edges for lengths of stay
    stay_lengths = y_test_df['stay_length']
    bin_length = 48
    los_bin_edges = np.arange(0, 198, bin_length)
    los_bin_edges = np.append(los_bin_edges, stay_lengths.max())
    
    # parse feature columns
    feature_cols = parse_feature_cols(x_test_dict)
    
    clf_models_dir = args.clf_models_dir
    
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    perf_df = pd.DataFrame()
    for p in range(len(los_bin_edges)-1):
        bin_start = los_bin_edges[p]
        bin_end = los_bin_edges[p+1]
        
        # filter out stay lengths within the current bin
        keep_inds = (stay_lengths>=bin_start)&(stay_lengths<=bin_end)
        
        if keep_inds.sum()==0:
            continue
        
#         mews_score_col = parse_feature_cols(mews_data_dict['schema'])
        x_test = x_test_df.loc[keep_inds, feature_cols].values
        y_test = y_test_df.loc[keep_inds, outcome_col].values
        
        # load the scaler
        scaler = pickle.load(open(os.path.join(clf_models_dir, 'skorch_logistic_regression', 'scaler.pkl'), 'rb'))
        x_test_transformed = scaler.transform(x_test)
        
        # load classifier
        if p==0:
            
            # load best LR minimizing BCE loss
            skorch_lr_bce = SkorchLogisticRegression(n_features=x_test.shape[1])
            skorch_lr_bce.initialize()
            
            # get the best model minimizining bce loss
#             bce_filename_aka = 'skorch_logistic_regression*cross_entropy_loss*history.json'
            bce_filename_aka = 'skorch_logistic_regression*cross_entropy_loss*_perf.csv'
            curr_model_dir = os.path.join(clf_models_dir, 'skorch_logistic_regression')
#             best_model_file_bce = get_best_model(curr_model_dir, bce_filename_aka)
            best_model_file_bce = get_best_model_after_threshold_search(curr_model_dir, bce_filename_aka)
            
            # plot training plots of best model
#             plot_best_model_training_plots(best_model_file_bce, 'logistic_regression_minimizing_cross_entropy')
            
            best_model_prefix_bce = best_model_file_bce.split('/')[-1].replace('_perf.csv', '') 
            
            best_model_perf_csv = os.path.join(curr_model_dir, best_model_prefix_bce+'_perf.csv')
            
            lr_best_model_perf_df = pd.read_csv(best_model_perf_csv)
            lr_thr = float(lr_best_model_perf_df.threshold)
            
            skorch_lr_bce.load_params(f_params=os.path.join(curr_model_dir,
                                                  best_model_prefix_bce+'params.pt'))
            print('Evaluating with saved model : %s'%(os.path.join(curr_model_dir, best_model_prefix_bce)))
            
            
            # get the best MLP model minimizining bce loss
#             bce_filename_aka = 'skorch_mlp*cross_entropy_loss*history.json'
            bce_filename_aka = 'skorch_mlp*cross_entropy_loss*_perf.csv'
            curr_model_dir = os.path.join(clf_models_dir, 'skorch_mlp')
#             best_model_file_bce = get_best_model(curr_model_dir, bce_filename_aka)
            best_model_file_bce = get_best_model_after_threshold_search(curr_model_dir, bce_filename_aka)
    
            # plot training plots of best model
#             plot_best_model_training_plots(best_model_file_bce, 'MLP_minimizing_cross_entropy')
            
            best_model_prefix_bce = best_model_file_bce.split('/')[-1].replace('_perf.csv', '')   
            best_model_n_hiddens_bce = int(best_model_prefix_bce.split('n_hiddens=')[-1]) 
            
            best_model_perf_csv = os.path.join(curr_model_dir, best_model_prefix_bce+'_perf.csv')
            
            mlp_best_model_perf_df = pd.read_csv(best_model_perf_csv)            
            mlp_thr = float(mlp_best_model_perf_df.threshold)
            
            # load model minimizing BCE loss
            skorch_mlp_bce = SkorchMLP(n_features=x_test.shape[1],
                                                    n_hiddens=best_model_n_hiddens_bce)
            skorch_mlp_bce.initialize()
            
            skorch_mlp_bce.load_params(f_params=os.path.join(curr_model_dir,
                                                  best_model_prefix_bce+'params.pt'))
            print('Evaluating with saved model : %s'%(os.path.join(curr_model_dir, best_model_prefix_bce)))
            
            # load the random forest
            rf_model_file = os.path.join(clf_models_dir, 'random_forest', 'random_forest_trained_model.joblib')
            rf_clf = load(rf_model_file)
            rf_thr = rf_clf.threshold
            
        for model_name, model_clf, model_thr in [('logistic regression', skorch_lr_bce, lr_thr), ('MLP', skorch_mlp_bce, mlp_thr),
                                                ('random forest', rf_clf, rf_thr)]:
            print('Evaluating %s on stay lengths between %.1f and %.1f hours'%(model_name, bin_start, bin_end))
            roc_auc_np = np.zeros(len(random_seed_list))
            balanced_accuracy_np = np.zeros(len(random_seed_list))
            log_loss_np = np.zeros(len(random_seed_list))
            avg_precision_np = np.zeros(len(random_seed_list))
            precision_score_np = np.zeros(len(random_seed_list))
            recall_score_np = np.zeros(len(random_seed_list))
            pos_label_ratio_np = np.zeros(len(random_seed_list))
            for k, seed in enumerate(random_seed_list):
                random.seed(int(seed))
                rnd_inds = random.sample(range(x_test.shape[0]), int(0.8*x_test.shape[0])) 
                curr_y_test = y_test[rnd_inds]  
                # because the rf classifier is a pipeline object with the scaler object appended, we dont need the scaled version of the inputs
                if model_name == 'random forest': 
                    curr_x_test = x_test[rnd_inds, :]
                else:
                    curr_x_test = x_test_transformed[rnd_inds, :]
                
#                 chosen_threshold = float(best_model_perf_df.threshold)
                y_pred = (model_clf.predict_proba(curr_x_test)[:,1]>=model_thr)*1
                y_pred_proba = model_clf.predict_proba(curr_x_test)[:, 1]
                
                if np.sum(curr_y_test)>0:
                    roc_auc_np[k] = roc_auc_score(curr_y_test, y_pred_proba)
                    balanced_accuracy_np[k] = balanced_accuracy_score(curr_y_test, y_pred)
                    log_loss_np[k] = log_loss(curr_y_test, y_pred_proba, normalize=True) / np.log(2)
                    avg_precision_np[k] = average_precision_score(curr_y_test, y_pred_proba)
                    precision_score_np[k] = precision_score(curr_y_test, y_pred)
                    recall_score_np[k] = recall_score(curr_y_test, y_pred)
                    pos_label_ratio_np[k] = curr_y_test.sum()/len(y_test)
                else:
                    roc_auc_np[k] = np.nan
                    balanced_accuracy_np[k] = np.nan
                    log_loss_np[k] = np.nan
                    avg_precision_np[k] = np.nan
                    precision_score_np[k] = np.nan
                    recall_score_np[k] = np.nan
                    pos_label_ratio_np[k] = np.nan
                
            print('Median AUROC : %.3f'%np.median(roc_auc_np))
            print('Median average precision : %.3f'%np.median(avg_precision_np))
            print('Median precision score : %.3f'%np.median(precision_score_np))
            print('Median recall score : %.3f'%np.median(recall_score_np))
            print('Median balanced accuracy score : %.3f'%np.median(balanced_accuracy_np))

            for prctile in prctile_vals:
                row_dict = dict()
                row_dict['model'] = model_name
                row_dict['percentile'] = prctile
                row_dict['bin_start'] = bin_start
                row_dict['bin_end'] = bin_end
                row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
                row_dict['N'] = len(x_test_transformed)
                row_dict['balanced_accuracy'] = np.percentile(balanced_accuracy_np, prctile)
                row_dict['log_loss'] = np.percentile(log_loss_np, prctile)
                row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)
                row_dict['precision_score'] = np.percentile(precision_score_np, prctile)
                row_dict['recall_score'] = np.percentile(recall_score_np, prctile)
                row_dict['positive_labels_ratio'] = np.percentile(pos_label_ratio_np, prctile)#curr_y_test.sum()/len(curr_y_test)
                perf_df = perf_df.append(row_dict, ignore_index=True)        
    
    
    # plot the performance
    perf_measures = ['roc_auc', 'average_precision', 'precision_score', 'recall_score', 'positive_labels_ratio']
    fontsize=18
    suffix = '_HUF'
    for perf_measure in perf_measures:
        f, axs = plt.subplots(figsize=[10,8])
        model_colors=['r', 'b', 'g', 'k', 'c']
        for p, model in enumerate(perf_df.model.unique()):
            inds = (perf_df.model==model)
            cur_df = perf_df.loc[inds, :].copy()
            x = perf_df.bin_start.unique()
            y = cur_df.loc[cur_df.percentile==50, perf_measure].values
            marker_sizes = cur_df.loc[cur_df.percentile==50, 'N'].values * 500/cur_df['N'].max()
            y_err  = np.zeros((2, len(x)))
            y_err[0,:] = y - cur_df.loc[cur_df.percentile==5, perf_measure].values
            y_err[1,:] = cur_df.loc[cur_df.percentile==95, perf_measure].values - y
            axs.errorbar(x=x, y=y, yerr=y_err, label=model, fmt='.-', linewidth=3, color=model_colors[p])
            axs.scatter(x, y, s=marker_sizes, color=model_colors[p])
        
        
        axs.set_xlabel('length of stay', fontsize=fontsize)
        fig_aka='perf_%s_vs_length_of_stay%s.pdf'%(perf_measure,suffix)

        if perf_measure == 'roc_auc':
            axs.set_ylim([0.48, 1])
        elif perf_measure == 'average_precision':
            axs.set_ylim([0, 0.9])
        elif perf_measure == 'precision_score':
            axs.set_ylim([0, 0.9])
        elif perf_measure == 'recall_score':
            axs.set_ylim([0, 0.9])

        axs.grid(True)
        axs.set_ylabel(perf_measure, fontsize=fontsize)
        
        if perf_measure != 'positive_labels_ratio':
            axs.legend(fontsize=fontsize-2, loc='upper left')
        axs.tick_params(labelsize=fontsize)
        fig_name = os.path.join(args.output_dir, fig_aka)
        
        # set the xticks in days
        axs.set_xticks([0, 48, 96, 144, 192])
        axs.set_xticklabels(['0-2 days', '2-4 days', '4-6 days', '6-8 days', '>8 days'])
        
        f.savefig(fig_name)
        print('Saved results to %s'%fig_name)
    
    perf_csv = os.path.join(args.output_dir, 'performance_over_lengths_of_stays.csv')
    print('Saving performance over stay lengths to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)
