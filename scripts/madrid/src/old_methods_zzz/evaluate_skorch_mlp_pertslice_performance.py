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
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchMLP'))
from sklearn.preprocessing import StandardScaler
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
    recall_valid_np[np.isnan(recall_valid_np)] = 0
    
    best_model_ind = np.argmax(recall_valid_np)
#     best_model_ind = np.argmin(valid_losses_np)


    return training_files[best_model_ind]
    
#     # keep only models that match 0.35 precision on validation
#     thresh=0.6
#     keep_inds = precision_valid_np>=thresh
#     precision_above_thresh = precision_valid_np[keep_inds]
#     recall_above_thresh = recall_valid_np[keep_inds]
    
#     # pick the model with the best recall among those
#     training_files_above_thresh = np.array(training_files)[keep_inds]
#     best_model_ind = np.argmax(recall_above_thresh)
    
#     return training_files_above_thresh[best_model_ind]
    
    
    
    

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
    parser.add_argument('--include_medications', default='True', type=str,
                       help='temporary flag to add/not add medictaions')
    
    
    args = parser.parse_args()
    
    ## get the test patient id's
    # get the test set's csv and dict
    y_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_test.csv.gz'))
    y_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'y_dict.json')
    
    # import the y dict to get the id cols
    y_test_dict = load_data_dict_json(y_test_dict_file)
    id_cols = parse_id_cols(y_test_dict)
    
    tslice_folders = os.path.join(args.tslice_folder, 'TSLICE=')
    collapsed_tslice_folders = os.path.join(args.collapsed_tslice_folder, 'TSLICE=')
    outcome_col = args.outcome_column_name
    tslices_list = args.evaluation_tslices.split(' ')
    y_test_ids_df = y_test_df[id_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)
    
    # get demographics csv and data_dict
    # for each patient get their vitals, labs, demographics
    _,_,_,_,demographics_df, demographics_data_dict,_,_,_,_ = get_preprocessed_data(args.preproc_data_dir)
    
    clf_models_dir = args.clf_models_dir
    
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    perf_df = pd.DataFrame()
    for p, tslice in enumerate(tslices_list):
        tslice_folder = tslice_folders + tslice
        collapsed_tslice_folder = collapsed_tslice_folders + tslice
        # get test set collapsed labs and vitals
        collapsed_vitals_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'CollapsedVitalsPerSequence.csv.gz'))
        collapsed_labs_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'CollapsedLabsPerSequence.csv.gz'))
        mews_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'MewsScoresPerSequence.csv.gz'))
        outcomes_df = pd.read_csv(os.path.join(tslice_folder,
                                               'clinical_deterioration_outcomes_filtered_%s_hours.csv.gz'%tslice))
        collapsed_vitals_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_CollapsedVitalsPerSequence.json'))
        collapsed_labs_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_CollapsedLabsPerSequence.json'))
        mews_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_MewsScoresPerSequence.json'))
        test_vitals_df = pd.merge(collapsed_vitals_df, y_test_ids_df, on=id_cols)
        test_labs_df = pd.merge(collapsed_labs_df, y_test_ids_df, on=id_cols)
        test_mews_df = pd.merge(mews_df, y_test_ids_df, on=id_cols)
        
        if args.include_medications=='True':
            collapsed_medications_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'CollapsedMedicationsPerSequence.csv.gz'))
            collapsed_medications_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_CollapsedMedicationsPerSequence.json'))
            test_medications_df = pd.merge(collapsed_medications_df, y_test_ids_df, on=id_cols)
            
            # merge them
            test_collapsed_features_df = pd.merge(pd.merge(test_vitals_df, test_labs_df, 
                                                           on=id_cols, how='inner'),
                                                  test_medications_df,on=id_cols, how='inner')
            data_dicts_list = [collapsed_vitals_data_dict, 
                               collapsed_labs_data_dict,
                               collapsed_medications_data_dict,
                               demographics_data_dict]
            
        else :
            test_collapsed_features_df = pd.merge(test_vitals_df, test_labs_df, 
                                                           on=id_cols, how='inner')
            
            data_dicts_list = [collapsed_vitals_data_dict,
                               collapsed_labs_data_dict,
                               demographics_data_dict]
        
        test_features_df = pd.merge(test_collapsed_features_df, demographics_df, on=id_cols)
        if p==0:
            test_features_dict = merge_data_dicts(data_dicts_list)

        test_outcomes_df = pd.merge(test_features_df[id_cols], outcomes_df, on=id_cols, how='inner')

    #     # get performance metrics
        feature_cols = parse_feature_cols(test_features_dict['schema'])
        mews_score_col = parse_feature_cols(mews_data_dict['schema'])
        x_test = test_features_df[feature_cols].values
        y_test = test_outcomes_df[outcome_col].values
        
        # load the scaler
        scaler = pickle.load(open(os.path.join(clf_models_dir, 'scaler.pkl'), 'rb'))
        x_test_transformed = scaler.transform(x_test)
        # load classifier
        if p==0:
            
            # get the best model minimizining bce loss
            bce_filename_aka = 'skorch_mlp*cross_entropy_loss*history.json'
            best_model_file_bce = get_best_model(clf_models_dir, bce_filename_aka)
            
            # plot training plots of best model
            plot_best_model_training_plots(best_model_file_bce, 'MLP_minimizing_cross_entropy')
            
            best_model_prefix_bce = best_model_file_bce.split('/')[-1].replace('history.json', '')   
            best_model_n_hiddens_bce = int(best_model_prefix_bce.split('n_hiddens=')[-1]) 
            
            
            # load model minimizing BCE loss
            skorch_mlp_bce = SkorchMLP(n_features=x_test.shape[1],
                                                    n_hiddens=best_model_n_hiddens_bce)
            skorch_mlp_bce.initialize()
            
            skorch_mlp_bce.load_params(f_params=os.path.join(clf_models_dir,
                                                  best_model_prefix_bce+'params.pt'))
            print('Evaluating with saved model : %s'%(os.path.join(clf_models_dir, best_model_prefix_bce)))
            
            from IPython import embed; embed()
            # get the best model minimizining surrogate loss
            sl_filename_aka = 'skorch_mlp*surrogate_loss_tight*history.json'
            best_model_file_sl = get_best_model(clf_models_dir, sl_filename_aka)
            
            # plot training plots of best model
            plot_best_model_training_plots(best_model_file_sl, 'MLP_minimizing_surrogate_loss_tight')            
            
            best_model_prefix_sl = best_model_file_sl.split('/')[-1].replace('history.json', '') 
            best_model_n_hiddens_sl = int(best_model_prefix_sl.split('n_hiddens=')[-1])
            
            # load model minimizing tight surrogate loss
            skorch_mlp_sl = SkorchMLP(n_features=x_test.shape[1],
                                                   n_hiddens=best_model_n_hiddens_sl)
            skorch_mlp_sl.initialize()
            
            skorch_mlp_sl.load_params(f_params=os.path.join(clf_models_dir,
                                                  best_model_prefix_sl+'params.pt'))
            print('Evaluating with saved model : %s'%(os.path.join(clf_models_dir, best_model_prefix_sl)))            
        
            from IPython import embed; embed()
        for loss_name, model_clf in [('cross_entropy_loss', skorch_mlp_bce), ('surrogate_loss_tight', skorch_mlp_sl)]:
            print('Evaluating skorch mlp trained minimizing %s on tslice=%s'%(loss_name, tslice))
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
                row_dict['model'] = 'skorch_mlp_%s'%loss_name
                row_dict['percentile'] = prctile
                row_dict['tslice'] = tslice
                row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
                row_dict['balanced_accuracy'] = np.percentile(balanced_accuracy_np, prctile)
                row_dict['log_loss'] = np.percentile(log_loss_np, prctile)
                row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)
                row_dict['precision_score'] = np.percentile(precision_score_np, prctile)
                row_dict['recall_score'] = np.percentile(recall_score_np, prctile)

                perf_df = perf_df.append(row_dict, ignore_index=True)        
                
    perf_csv = os.path.join(args.output_dir, 'skorch_mlp_pertslice_performance.csv')
    print('Saving mlp per-tslice performance to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)