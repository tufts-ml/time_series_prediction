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
    

def plot_best_model_training_plots(best_model_history_file, plt_name):
    
    metrics = ['precision', 'recall', 'loss']
    training_hist_df = pd.DataFrame(json.load(open(best_model_history_file))) 
    f, axs = plt.subplots(len(metrics), 1, figsize=(8,8), sharex=True)
    
    
    for i, metric in enumerate(metrics): 
        # plot epochs vs precision on train and validation
        try:
            axs[i].plot(training_hist_df.epoch, training_hist_df['%s_train'%metric], label='%s(train)'%metric)
            axs[i].plot(training_hist_df.epoch, training_hist_df['%s_valid'%metric], label='%s(validation)'%metric) 
            axs[i].set_ylim([0.1, 1])
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
    parser.add_argument('--collapsed_tslice_folder', type=str, 
                        help='folder where collapsed features from each tslice are stored')
    parser.add_argument('--tslice_folder', type=str, 
                        help='folder where features filtered by tslice are stored')
    parser.add_argument('--evaluation_tslices', type=str,
                        help='evaluation tslices separated by spaces')
    parser.add_argument('--preproc_data_dir', type=str,
                        help='folder where the preprocessed data is stored')
    parser.add_argument('--outcome_column_name', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    parser.add_argument('--random_seed_list', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    parser.add_argument('--output_dir', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    
    
    args = parser.parse_args()
    
#     models = ['logistic_regression']
#     clf_models_dict = dict.fromkeys(models)
#     for model in models:
#         clf_model_file = os.path.join(args.clf_models_dir, model, '%s_trained_model.joblib'%model)
#         clf_model = load(clf_model_file)
#         clf_models_dict[model] = clf_model
    
    clf_models_dir = args.clf_models_dir
    
    ## get the test patient id's
    # get the test set's csv and dict
    y_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_test.csv'))
    y_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'y_dict.json')
    
    # import the y dict to get the id cols
    y_test_dict = load_data_dict_json(y_test_dict_file)
    id_cols = parse_id_cols(y_test_dict)
    
    tslice_folders = os.path.join(args.tslice_folder, 'TSLICE=')
    collapsed_tslice_folders = os.path.join(args.collapsed_tslice_folder, 'TSLICE=')
    outcome_col = args.outcome_column_name
    tslices_list = args.evaluation_tslices.split(' ')
    y_test_ids_df = y_test_df[id_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)
    
    
    # evaluate lr and rf performance on tslices with percentiles
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    perf_df = pd.DataFrame()
    
    for p, tslice in enumerate(tslices_list):
        tslice_folder = tslice_folders + tslice
        collapsed_tslice_folder = collapsed_tslice_folders + tslice
        # get test set collapsed labs and vitals
        collapsed_features_df = pd.read_csv(os.path.join(collapsed_tslice_folder,
                                                         'CollapsedFeaturesPerSequence.csv'))
        outcomes_df = pd.read_csv(os.path.join(tslice_folder,
                                               'outcomes_filtered_%s_hours.csv'%tslice))
        test_features_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder,
                                                              'Spec_CollapsedFeaturesPerSequence.json'))

        test_features_df = pd.merge(collapsed_features_df, y_test_ids_df, on=id_cols)

        test_outcomes_df = pd.merge(test_features_df[id_cols], outcomes_df, on=id_cols, how='inner')

    #     # get performance metrics
        feature_cols = parse_feature_cols(test_features_dict)
        x_test = test_features_df[feature_cols].values
        y_test = test_outcomes_df[outcome_col].values

        # load the scaler
        scaler = pickle.load(open(os.path.join(clf_models_dir, 'scaler.pkl'), 'rb'))
        x_test_transformed = scaler.transform(x_test)
        
        
        # load classifier
        if p==0:
            
            # load model minimizing BCE loss
            skorch_lr_bce = SkorchLogisticRegression(n_features=x_test.shape[1])
            skorch_lr_bce.initialize()
            
            # get the best model minimizining bce loss
            bce_filename_aka = 'skorch_logistic_regression*cross_entropy_loss*history.json'
            best_model_file_bce = get_best_model(clf_models_dir, bce_filename_aka)
            
            # plot training plots of best model
            plot_best_model_training_plots(best_model_file_bce, 'logistic_regression_minimizing_cross_entropy')
            
            best_model_prefix_bce = best_model_file_bce.split('/')[-1].replace('history.json', '') 
            
#             best_model_prefix_bce = 'skorch_logistic_regression_lr=0.005-weight_decay=0.01-batch_size=-1-scoring=cross_entropy_loss-seed=111'
            skorch_lr_bce.load_params(f_params=os.path.join(clf_models_dir,
                                                  best_model_prefix_bce+'params.pt'))
            print('Evaluating with saved model : %s'%(os.path.join(clf_models_dir, best_model_prefix_bce)))
            
            # load model minimizing tight surrogate loss
            skorch_lr_sl = SkorchLogisticRegression(n_features=x_test.shape[1])
            skorch_lr_sl.initialize()
            
            # get the best model minimizining surrogate loss
            sl_filename_aka = 'skorch_logistic_regression*surrogate_loss_tight*history.json'
            best_model_file_sl = get_best_model(clf_models_dir, sl_filename_aka)

            # plot training plots of best model
            plot_best_model_training_plots(best_model_file_sl, 'logistic_regression_minimizing_surrogate_loss_tight')

            best_model_prefix_sl = best_model_file_sl.split('/')[-1].replace('history.json', '') 
            
#             best_model_prefix_sl = 'skorch_logistic_regression_lr=0.005-weight_decay=0.01-batch_size=-1-scoring=surrogate_loss_tight-seed=111'
            skorch_lr_sl.load_params(f_params=os.path.join(clf_models_dir,
                                                  best_model_prefix_sl+'params.pt'))
            print('Evaluating with saved model : %s'%(os.path.join(clf_models_dir, best_model_prefix_sl)))            
            
            from IPython import embed; embed()
        for loss_name, model_clf in [('cross_entropy_loss', skorch_lr_bce), ('surrogate_loss_tight', skorch_lr_sl)]:
            print('Evaluating skorch lr trained minimizing %s on tslice=%s'%(loss_name, tslice))
            roc_auc_np = np.zeros(len(random_seed_list))
            balanced_accuracy_np = np.zeros(len(random_seed_list))
            log_loss_np = np.zeros(len(random_seed_list))
            avg_precision_np = np.zeros(len(random_seed_list))
            precision_score_np = np.zeros(len(random_seed_list))
            recall_score_np = np.zeros(len(random_seed_list))
            for k, seed in enumerate(random_seed_list):
                random.seed(int(seed))
                rnd_inds = random.sample(range(x_test.shape[0]), int(0.8*x_test.shape[0])) 
                curr_y_test = y_test[rnd_inds]           
                curr_x_test = x_test_transformed[rnd_inds, :]
                y_pred = model_clf.predict(curr_x_test)
                y_pred_proba = model_clf.predict_proba(curr_x_test)[:, 1]

                roc_auc_np[k] = roc_auc_score(curr_y_test, y_pred_proba)
                balanced_accuracy_np[k] = balanced_accuracy_score(curr_y_test, y_pred)
                log_loss_np[k] = log_loss(curr_y_test, y_pred_proba, normalize=True) / np.log(2)
                avg_precision_np[k] = average_precision_score(curr_y_test, y_pred_proba)
                precision_score_np[k] = precision_score(curr_y_test, y_pred)
                recall_score_np[k] = recall_score(curr_y_test, y_pred)

            print('Median AUROC : %.3f'%np.median(roc_auc_np))
            print('Median average precision : %.3f'%np.median(avg_precision_np))
            print('Median precision score : %.3f'%np.median(precision_score_np))
            print('Median recall score : %.3f'%np.median(recall_score_np))
            print('Median balanced accuracy score : %.3f'%np.median(balanced_accuracy_np))

            for prctile in prctile_vals:
                row_dict = dict()
                row_dict['model'] = 'skorch_lr_%s'%loss_name
                row_dict['percentile'] = prctile
                row_dict['tslice'] = tslice
                row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
                row_dict['balanced_accuracy'] = np.percentile(balanced_accuracy_np, prctile)
                row_dict['log_loss'] = np.percentile(log_loss_np, prctile)
                row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)
                row_dict['precision_score'] = np.percentile(precision_score_np, prctile)
                row_dict['recall_score'] = np.percentile(recall_score_np, prctile)

                perf_df = perf_df.append(row_dict, ignore_index=True)        
                
    perf_csv = os.path.join(args.output_dir, 'skorch_lr_pertslice_performance.csv')
    print('Saving lr, rf per-tslice performance to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)
