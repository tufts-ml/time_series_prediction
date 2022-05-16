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
    sns.set_context("notebook", font_scale=1.25)
    alpha=0.3
    all_models_history_files = glob.glob(os.path.join(clf_models_dir, all_models_history_files_aka))
    for f_ind, model_history_file in enumerate(all_models_history_files):
        training_hist_df = pd.DataFrame(json.load(open(model_history_file))) 
        try:
            if training_hist_df['precision_train'].values[-1]>0.2:
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
#                         axs[i].set_ylim([0, 2500])
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
#                         axs[i].grid(True) 
        

            
        except:
            continue
#     for ax in axs:
#         ax.legend(loc='upper left')
#         ax.grid(True) 
    axs[i].set_xlabel('epochs')
    axs[i].set_xlim([0, 500])
#     plt.suptitle(plt_name)
    f.savefig(plt_name+'.png')
#     f.savefig(plt_name+'.pdf', bbox_inches='tight', pad_inches=0)
    

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
    x_train_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_train.csv'))
    
#     x_valid_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'x_valid.csv'))
#     y_valid_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_valid.csv'))
    
    x_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'x_test.csv'))
    y_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_test.csv'))
    y_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'y_dict.json')
    x_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'x_dict.json')
    
    # import the y dict to get the id cols
    y_test_dict = load_data_dict_json(y_test_dict_file)
    x_test_dict = load_data_dict_json(x_test_dict_file)
    id_cols = parse_id_cols(y_test_dict)
    feature_cols = parse_feature_cols(x_test_dict)
    outcome_col = args.outcome_column_name
#     # get performance metrics
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
    scaler = pickle.load(open(os.path.join(clf_models_dir, 'scaler.pkl'), 'rb'))
    x_train_transformed = scaler.transform(x_tr)
    x_valid_transformed = scaler.transform(x_valid)
    x_test_transformed = scaler.transform(x_test)

    # load model minimizing BCE loss
#     skorch_lr_bce = SkorchLogisticRegression(n_features=x_test.shape[1])
#     skorch_lr_bce.initialize()
    
    
    '''
    sl_rand_init_incremental_min_precision_filename_aka = 'skorch_logistic_regression*surrogate_loss_tight*warm_start=false*incremental_min_precision=true*history.json'
    
#     plot_all_models_training_plots(clf_models_dir, sl_rand_init_incremental_min_precision_filename_aka, 'skorch_lr_incremental_min_precision')
    
    precisions_train_incremental_min_precision, precisions_valid_incremental_min_precision, precisions_test_incremental_min_precision, recalls_train_incremental_min_precision, recalls_valid_incremental_min_precision,  recalls_test_incremental_min_precision= get_all_precision_recalls(clf_models_dir, sl_rand_init_incremental_min_precision_filename_aka.replace('history.json', '.csv'))
    
    make_precision_recall_boxplots(precisions_train_incremental_min_precision, precisions_valid_incremental_min_precision, precisions_test_incremental_min_precision, recalls_train_incremental_min_precision, recalls_valid_incremental_min_precision, recalls_test_incremental_min_precision, 'lr_precision_recall_boxplot_incremental_min_precision', '(Ramp Up)')
    '''
    
    sl_rand_init_direct_min_precision_filename_aka = 'skorch_logistic_regression*surrogate_loss_tight*warm_start=false*incremental_min_precision=false*history.json'
    
#     plot_all_models_training_plots(clf_models_dir, sl_rand_init_direct_min_precision_filename_aka, 'skorch_lr_direct_min_precision')
    
    precisions_train_direct_min_precision, precisions_valid_direct_min_precision, precisions_test_direct_min_precision, recalls_train_direct_min_precision, recalls_valid_direct_min_precision, recalls_test_direct_min_precision, training_files_direct_min_precision = get_all_precision_recalls(clf_models_dir, sl_rand_init_direct_min_precision_filename_aka.replace('history.json', '.csv'))
    
    
#     make_precision_recall_boxplots(precisions_train_direct_min_precision, precisions_valid_direct_min_precision, precisions_test_direct_min_precision, recalls_train_direct_min_precision, recalls_valid_direct_min_precision, recalls_test_direct_min_precision, 'lr_precision_recall_boxplot_direct_min_precision', '(Direct)')
    
    '''
    sl_bce_perturb_filename_aka = 'skorch_logistic_regression*surrogate_loss_tight*warm_start=true*history.json'
    
    plot_all_models_training_plots(clf_models_dir, sl_bce_perturb_filename_aka, 'skorch_lr_bce_perturb')
    
    precisions_train_bce_perturb, precisions_valid_bce_perturb, precisions_test_bce_perturb, recalls_train_bce_perturb, recalls_valid_bce_perturb, recalls_test_bce_perturb = get_all_precision_recalls(clf_models_dir, sl_bce_perturb_filename_aka.replace('history.json', '.csv'))
    
    
    make_precision_recall_boxplots(precisions_train_bce_perturb, precisions_valid_bce_perturb, precisions_test_bce_perturb, recalls_train_bce_perturb, recalls_valid_bce_perturb, recalls_test_bce_perturb, 'lr_precision_recall_boxplot_bce_perturb', '(BCE + Perturbation)')
    '''

    
    bce_plus_thresh_filename_aka = 'skorch_logistic_regression*cross_entropy_loss*warm_start=false*history.json'
    
#     plot_all_models_training_plots(clf_models_dir, bce_plus_thresh_filename_aka, 'skorch_lr_bce_plus_thresh')
    
    precisions_train_bce_plus_thresh, precisions_valid_bce_plus_thresh, precisions_test_bce_plus_thresh, recalls_train_bce_plus_thresh, recalls_valid_bce_plus_thresh, recalls_test_bce_plus_thresh, training_files_bce_plus_thresh = get_all_precision_recalls(clf_models_dir, bce_plus_thresh_filename_aka.replace('history.json', '.csv'))
    
    
#     make_precision_recall_boxplots(precisions_train_bce_plus_thresh, precisions_valid_bce_plus_thresh, precisions_test_bce_plus_thresh, recalls_train_bce_plus_thresh, recalls_valid_bce_plus_thresh, recalls_test_bce_plus_thresh, 'lr_precision_recall_boxplot_bce_plus_thresh', '(BCE + Threshold Search)')
    

    sl_loose_filename_aka = 'skorch_logistic_regression*surrogate_loss_loose*warm_start=false*history.json'
    
#     plot_all_models_training_plots(clf_models_dir, sl_loose_filename_aka, 'skorch_lr_sl_loose')
    
    precisions_train_sl_loose, precisions_valid_sl_loose, precisions_test_sl_loose, recalls_train_sl_loose, recalls_valid_sl_loose, recalls_test_sl_loose, training_files_sl_loose = get_all_precision_recalls(clf_models_dir, sl_loose_filename_aka.replace('history.json', '.csv'))
    
    
#     make_precision_recall_boxplots(precisions_train_sl_loose, precisions_valid_sl_loose, precisions_test_sl_loose, recalls_train_sl_loose, recalls_valid_sl_loose, recalls_test_sl_loose, 'lr_precision_recall_boxplot_sl_loose', '(Surrogate Loss Hinge Bound)')
    
#     from IPython import embed; embed()
    best_files_dict = dict()
    best_perf_dict = dict()
    for ii, (method, prcs_train, recs_train, prcs_valid, recs_valid, prcs_test, recs_test, tr_files) in enumerate([
        ('direct min precision', 
         precisions_train_direct_min_precision,
         recalls_train_direct_min_precision,
         precisions_valid_direct_min_precision,
         recalls_valid_direct_min_precision,
         precisions_test_direct_min_precision,
         recalls_test_direct_min_precision,
         training_files_direct_min_precision),
#         ('incremental min precision',
#          precisions_train_incremental_min_precision,
#          recalls_train_incremental_min_precision,
#          precisions_valid_incremental_min_precision,
#          recalls_valid_incremental_min_precision,
#          precisions_test_incremental_min_precision,
#          recalls_test_incremental_min_precision),
#         ('bce + perturbaton',
#          precisions_train_bce_perturb,
#          recalls_train_bce_perturb,
#          precisions_valid_bce_perturb,
#          recalls_valid_bce_perturb,
#          precisions_test_bce_perturb,         
#          recalls_test_bce_perturb),
        ('bce + threshold search',
         precisions_train_bce_plus_thresh,
         recalls_train_bce_plus_thresh,
         precisions_valid_bce_plus_thresh,
         recalls_valid_bce_plus_thresh, 
         precisions_test_bce_plus_thresh,
         recalls_test_bce_plus_thresh,
         training_files_bce_plus_thresh),
        ('Surrogate Loss (Hinge Bound)',
         precisions_train_sl_loose,
         recalls_train_sl_loose,
         precisions_valid_sl_loose,
         recalls_valid_sl_loose, 
         precisions_test_sl_loose,
         recalls_test_sl_loose,
         training_files_sl_loose)
    ]):
        
        min_prec_tr = 0.7
        min_prec_va = 0.6
        keep_inds = (prcs_train>min_prec_tr)&(prcs_valid>min_prec_va)
        
        if keep_inds.sum()==0:
            keep_inds = (prcs_train>min_prec_tr)&(prcs_valid>0.7)
        
        fracs_above_min_precision = (keep_inds).sum()/len(prcs_train)
        prcs_train = prcs_train[keep_inds]
        prcs_valid = prcs_valid[keep_inds]
        prcs_test = prcs_test[keep_inds]        
        recs_train = recs_train[keep_inds]
        recs_valid = recs_valid[keep_inds]
        recs_test = recs_test[keep_inds] 
        tr_files = np.array(tr_files)[keep_inds]
        
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
        print('--------------------------------------------------')
        print('best training file : %s'%tr_files[best_ind])
        best_files_dict[method] = tr_files[best_ind]
        best_perf_dict[method] = dict()
        best_perf_dict[method]['precision_train'] = prcs_train[best_ind]
        best_perf_dict[method]['precision_valid'] = prcs_valid[best_ind]
        best_perf_dict[method]['precision_test'] = prcs_test[best_ind]
        best_perf_dict[method]['recall_train'] = recs_train[best_ind]
        best_perf_dict[method]['recall_valid'] = recs_valid[best_ind]
        best_perf_dict[method]['recall_test'] = recs_test[best_ind]

    
    ## get the 5th, 50th and 95th percentile of recall scores
    random_seed_list = [111, 412, 5318, 90, 101, 8491, 8213, 1721, 1, 58, 892, 55, 623, 199, 1829, 902, 1992, 24, 8]  
    
    for ii, (method, best_model_fname, thr) in enumerate([
        ('Sigmoid bound', best_files_dict['direct min precision'], .5),
        ('BCE + Threshold search', best_files_dict['bce + threshold search'], .9696),
        ('Hinge Bound', best_files_dict['Surrogate Loss (Hinge Bound)'], .5)
    ]):
        
        skorch_lr_clf = SkorchLogisticRegression(n_features=x_test.shape[1])
        skorch_lr_clf.initialize()
        skorch_lr_clf.load_params(f_params=os.path.join(clf_models_dir,
                                                        best_model_fname.replace('_perf.csv', 'params.pt')))

        y_train_pred_probas = skorch_lr_clf.predict_proba(x_train_transformed)[:,1]
        y_train_preds = y_train_pred_probas>=thr       
        
        y_test_pred_probas = skorch_lr_clf.predict_proba(x_test_transformed)[:,1]
        y_test_preds = y_test_pred_probas>=thr
        
        
        precisions_train_np, precisions_test_np = np.zeros(len(random_seed_list)), np.zeros(len(random_seed_list))
        recalls_train_np, recalls_test_np = np.zeros(len(random_seed_list)), np.zeros(len(random_seed_list))
        for k, seed in enumerate(random_seed_list):
            rnd_inds_tr = random.sample(range(x_train_transformed.shape[0]), int(0.8*x_train_transformed.shape[0])) 
            precisions_train_np[k] = precision_score(y_train[rnd_inds_tr], y_train_preds[rnd_inds_tr])
            recalls_train_np[k] = recall_score(y_train[rnd_inds_tr], y_train_preds[rnd_inds_tr])
            
            rnd_inds_te = random.sample(range(x_test.shape[0]), int(0.8*x_test.shape[0])) 
            precisions_test_np[k] = precision_score(y_test[rnd_inds_te], y_test_preds[rnd_inds_te])
            recalls_test_np[k] = recall_score(y_test[rnd_inds_te], y_test_preds[rnd_inds_te])           
        
        print('Method : %s'%method)
        train_perf_dict = {'precision_5' : np.percentile(precisions_train_np, 5),
                             'precision_50' : np.percentile(precisions_train_np, 50),
                             'precision_95' : np.percentile(precisions_train_np, 95),
                             'recall_5' : np.percentile(recalls_train_np, 5),
                             'recall_50' : np.percentile(recalls_train_np, 50),
                             'recall_95' : np.percentile(recalls_train_np, 95),}

        
        test_perf_dict = {'precision_5' : np.percentile(precisions_test_np, 5),
                             'precision_50' : np.percentile(precisions_test_np, 50),
                             'precision_95' : np.percentile(precisions_test_np, 95),
                             'recall_5' : np.percentile(recalls_test_np, 5),
                             'recall_50' : np.percentile(recalls_test_np, 50),
                             'recall_95' : np.percentile(recalls_test_np, 95),}
        print('Training set performance : ')
        print(train_perf_dict)
        
        print('Test set performance : ')
        print(test_perf_dict)
        
        
    
    
    
    ### select 1 classifier and plot its precision and recalls across many thresholds on train, valid and test
    f_tr, axs_tr = plt.subplots(1, 1, figsize=(8, 8))
    f_va, axs_va = plt.subplots(1, 1, figsize=(8, 8))
    f_te, axs_te = plt.subplots(1, 1, figsize=(8, 8))
    sns.set_context("notebook", font_scale=1.75)
    sns.set_style("whitegrid")
    fontsize=12
    for ii, (method, best_model_fname, model_color, chosen_prec_recall_dict) in enumerate([
        ('Sigmoid bound', best_files_dict['direct min precision'], 'r', best_perf_dict['direct min precision']),
        ('BCE + Threshold search', best_files_dict['bce + threshold search'], 'b', best_perf_dict['bce + threshold search']),
        ('Hinge Bound', best_files_dict['Surrogate Loss (Hinge Bound)'], 'g', best_perf_dict['Surrogate Loss (Hinge Bound)'])
    ]):
        
        skorch_lr_clf = SkorchLogisticRegression(n_features=x_test.shape[1])
        skorch_lr_clf.initialize()
        skorch_lr_clf.load_params(f_params=os.path.join(clf_models_dir,
                                                        best_model_fname.replace('_perf.csv', 'params.pt')))

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
             chosen_prec_recall_dict['precision_train'], chosen_prec_recall_dict['recall_train'], f_tr, axs_tr),
            ('valid', target_precisions, recalls_at_target_precisions_valid, 
             chosen_prec_recall_dict['precision_valid'], chosen_prec_recall_dict['recall_valid'], f_va, axs_va),
            ('test', target_precisions, recalls_at_target_precisions_test, 
             chosen_prec_recall_dict['precision_test'], chosen_prec_recall_dict['recall_test'], f_te, axs_te)]):
            axs.plot(recs, precs, color=model_color, label=method, linewidth=3, zorder=1)
#             axs[jj].plot(chosen_rec, chosen_prec, color=model_color, marker = 'x', markersize=8)
            axs.set_ylabel('Target precision', fontsize=fontsize+2)
            axs.set_title('Recalls at target precision (%s set)'%split, fontsize=fontsize+4)
            xmin = -0.003
            xmax = 0.35
            ymin = 0.35
            ymax = 1.0
            xticks = np.arange(0.0, 1.0, 0.05)
            xticklabels = ['%.2f'%x for x in xticks]
            axs.set_xticks(xticks)
            axs.set_xticklabels(xticklabels)
            axs.set_xlim([xmin, xmax])
            axs.set_ylim([ymin, ymax])
            axs.set_xlabel('Recall at target precision', fontsize=fontsize+2)
            
            chosen_thresh_ind = np.argmin(abs(chosen_prec - precs))
            
            if ii==2:
                axs.plot(recs[chosen_thresh_ind], precs[chosen_thresh_ind], color='k', marker='+', mew=3, markersize=25,
                            label='selected operating point', zorder=2)   
            else:
                axs.plot(recs[chosen_thresh_ind], precs[chosen_thresh_ind], color='k', marker='+', mew=3, markersize=25, zorder=2)                  
            
            axs.legend(fontsize=fontsize)
#     f.savefig('skorch_lr_recalls_at_various_target_precisions.png', pad_inches=0)
        f.savefig('skorch_lr_recalls_at_various_target_precisions_%s.pdf'%split, bbox_inches='tight', pad_inches=0)
        f.savefig('skorch_lr_recalls_at_various_target_precisions_%s.png'%split, bbox_inches='tight', 
                  pad_inches=0.5)
    
    
    from IPython import embed; embed()
    ## plot the PR curve
    f, axs = plt.subplots(2, 2, figsize=(15, 15))
    sns.set_context("notebook", font_scale=1.25)
    sns.set_style("whitegrid")
    
    for ii, (method, best_model_fname, model_color) in enumerate([('Sigmoid bound', 
                                                                   best_files_dict['direct min precision'], 
                                                                   'r'),
                                                                  ('BCE', 
                                                                   best_files_dict['bce + threshold search'], 
                                                                   'b'),
                                                                  ('Hinge bound', 
                                                                   best_files_dict['Surrogate Loss (Hinge Bound)'], 
                                                                   'g')]):
        
        skorch_lr_clf = SkorchLogisticRegression(n_features=x_test.shape[1])
        skorch_lr_clf.initialize()
        skorch_lr_clf.load_params(f_params=os.path.join(clf_models_dir,
                                                        best_model_fname.replace('_perf.csv', 'params.pt')))
        
        y_valid_pred_probas = skorch_lr_clf.predict_proba(x_valid_transformed)[:,1]
        y_valid_preds = y_valid_pred_probas>=0.5
        
        y_test_pred_probas = skorch_lr_clf.predict_proba(x_test_transformed)[:,1]
        y_test_preds = y_test_pred_probas>=0.5
        
        #compute roc curve
        fpr_valid, tpr_valid, thresholds_auc_valid = roc_curve(y_valid, y_valid_pred_probas)
        precision_valid, recall_valid, thresholds_pr_valid = precision_recall_curve(y_valid, y_valid_pred_probas)
        roc_valid = roc_auc_score(y_valid, y_valid_pred_probas)
        ap_valid = average_precision_score(y_valid, y_valid_pred_probas)
        
        fpr_test, tpr_test, thresholds_auc_test = roc_curve(y_test, y_test_pred_probas)
        precision_test, recall_test, thresholds_pr_test = precision_recall_curve(y_test, y_test_pred_probas)  
        roc_test = roc_auc_score(y_test, y_test_pred_probas)
        ap_test = average_precision_score(y_test, y_test_pred_probas)
        
                
        # get the thresholds, precisions, recalls, tprs satisfying the validation precision greater than 0.8 on validation
        select_inds_pr_valid = precision_valid>=0.8
        select_thresholds_valid = thresholds_pr_valid[select_inds_pr_valid[:-1]]
        select_precision_valid = precision_valid[select_inds_pr_valid]
        select_recall_valid = recall_valid[select_inds_pr_valid]
        select_inds_auc_valid = fpr_valid<=0.2
        select_thresholds_auc_valid = thresholds_auc_valid[select_inds_auc_valid]
        select_tpr_valid = tpr_valid[select_inds_auc_valid]
        select_fpr_valid = fpr_valid[select_inds_auc_valid]        
        
        
        select_inds_pr_test = precision_test>=0.8
        select_thresholds_test = thresholds_pr_test[select_inds_pr_test[:-1]]
        select_precision_test = precision_test[select_inds_pr_test]
        select_recall_test = recall_test[select_inds_pr_test]
        select_inds_auc_test = fpr_test<=0.2
        select_thresholds_auc_test = thresholds_auc_test[select_inds_auc_test]
        select_tpr_test = tpr_test[select_inds_auc_test]
        select_fpr_test = fpr_test[select_inds_auc_test]   
        
        
        
        # get the threshold, fpr, tpr of max recall with precision greater than 0.8 on validation
        if method=='BCE':
            best_ind_pr_valid = np.argmax(select_recall_valid)
            best_precision_valid = select_precision_valid[best_ind_pr_valid]
            best_recall_valid = select_recall_valid[best_ind_pr_valid]
            best_threshold_valid = select_thresholds_valid[best_ind_pr_valid]
            
            best_ind_auc_valid = np.argmax(select_tpr_valid)
            best_tpr_valid = select_tpr_valid[best_ind_auc_valid]
            best_fpr_valid = select_fpr_valid[best_ind_auc_valid]
            best_threshold_auc_valid = select_thresholds_auc_valid[best_ind_auc_valid]            
            
        else:
            keep_inds = (thresholds_pr_valid>=0.50)&(precision_valid[:-1]>=0.8)
            keep_thresholds_valid = thresholds_pr_valid[keep_inds]
            keep_precision_valid = precision_valid[:-1][keep_inds]
            keep_recall_valid = recall_valid[:-1][keep_inds]
            best_ind_pr_valid = 0
            best_precision_valid = keep_precision_valid[best_ind_pr_valid]
            best_recall_valid = keep_recall_valid[best_ind_pr_valid] 
            best_threshold_valid = keep_thresholds_valid[best_ind_pr_valid]
            
#             keep_inds_auc = (thresholds_auc_valid>=0.50)&(fpr_valid<=0.2)
#             keep_thresholds_auc_valid = thresholds_auc_valid[keep_inds_auc]
#             keep_fpr_valid = fpr_valid[keep_inds_auc]
#             keep_tpr_valid = tpr_valid[keep_inds_auc]
#             best_ind_auc_valid = np.argmax(keep_tpr_valid)
#             best_fpr_valid = keep_fpr_valid[best_ind_auc_valid]
#             best_tpr_valid = keep_tpr_valid[best_ind_auc_valid] 
#             best_threshold_auc_valid = keep_thresholds_auc_valid[best_ind_auc_valid]
            chosen_ind_auc_valid = np.argmax(select_tpr_valid)
            best_tpr_valid = select_tpr_valid[chosen_ind_auc_valid]
            best_fpr_valid = select_fpr_valid[chosen_ind_auc_valid]
        
        chosen_ind_test = np.argmax(select_recall_test)
        best_recall_test = select_recall_test[chosen_ind_test]
        best_precision_test = select_precision_test[chosen_ind_test]
        
        chosen_ind_auc_test = np.argmax(select_tpr_test)
        best_tpr_test = select_tpr_test[chosen_ind_auc_test]
        best_fpr_test = select_fpr_test[chosen_ind_auc_test]
        
        fontsize=12
        axs[0, 0].plot(fpr_valid, tpr_valid, c=model_color, label=method +' (AUROC : %.2f)'%roc_valid)
        axs[0, 0].set_xlabel('False Positive Rate', fontsize=fontsize)
        axs[0, 0].set_ylabel('True Positive Rate', fontsize=fontsize)
        axs[0, 0].set_title('ROC (Valid)')
        axs[0, 0].legend(fontsize=fontsize)
        axs[0, 0].plot(select_fpr_valid, select_tpr_valid, linewidth=8, c=model_color, alpha=0.5)
        if ii==2:
            axs[0, 0].plot(best_fpr_valid, best_tpr_valid, 'kx', markersize=10, label='chosen threshold')
        else:
            axs[0, 0].plot(best_fpr_valid, best_tpr_valid, 'kx', markersize=10)        
        
        
        axs[1, 0].plot(fpr_test, tpr_test, c=model_color, label=method +' (AUROC : %.2f)'%roc_test)
        axs[1, 0].set_xlabel('False Positive Rate', fontsize=fontsize)
        axs[1, 0].set_ylabel('True Positive Rate', fontsize=fontsize)
        axs[1, 0].set_title('ROC (Test)')   
        axs[1, 0].legend(fontsize=fontsize)
        axs[1, 0].plot(select_fpr_test, select_tpr_test, linewidth=8, c=model_color, alpha=0.5)
        if ii==2:
            axs[1, 0].plot(best_fpr_test, best_tpr_test, 'kx', markersize=10, label='chosen threshold')
        else:
            axs[1, 0].plot(best_fpr_test, best_tpr_test, 'kx', markersize=10) 
        
        
        axs[0, 1].plot(recall_valid, precision_valid, c=model_color, label=method +' (AUPRC : %.2f)'%ap_valid)
        axs[0, 1].set_xlabel('Recall', fontsize=fontsize)
        axs[0, 1].set_ylabel('Precision', fontsize=fontsize)
        axs[0, 1].set_title('Precision Recall Curve (Valid)') 
        axs[0, 1].plot(select_recall_valid, select_precision_valid, linewidth=8, c=model_color, alpha=0.5)
        if ii==2:
            axs[0, 1].plot(best_recall_valid, best_precision_valid, 'kx', markersize=10, label='chosen threshold')
        else:
            axs[0, 1].plot(best_recall_valid, best_precision_valid, 'kx', markersize=10)

        axs[0, 1].legend(fontsize=fontsize)
        
        
        axs[1, 1].plot(recall_test, precision_test, c=model_color, label=method +' (AUPRC : %.2f)'%ap_test)
        axs[1, 1].set_xlabel('Recall', fontsize=fontsize)
        axs[1, 1].set_ylabel('Precision', fontsize=fontsize)
        axs[1, 1].set_title('Precision Recall Curve (Test)') 
        axs[1, 1].plot(select_recall_test, select_precision_test, linewidth=8, c=model_color, alpha=0.5)
        if ii==2:
            axs[1, 1].plot(best_recall_test, best_precision_test, 'kx', markersize=10, label='chosen threshold')
        else:
            axs[1, 1].plot(best_recall_test, best_precision_test, 'kx', markersize=10)
        axs[1, 1].legend(fontsize=fontsize)
    
    f.savefig('roc_prc_all_methods.pdf', bbox_inches='tight', pad_inches=0)
    f.savefig('roc_prc_all_methods.png')
    
    
    from IPython import embed; embed()
    
