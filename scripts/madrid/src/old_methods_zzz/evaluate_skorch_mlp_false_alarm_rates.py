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
                             roc_auc_score, roc_curve, precision_recall_curve, precision_score)
from utils import load_data_dict_json
import ast
from filter_admissions_by_tslice import get_preprocessed_data
import random
import pickle
import glob
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchMLP'))
from sklearn.preprocessing import StandardScaler
from SkorchMLP import SkorchMLP
import datetime

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
    parser.add_argument('--collapsed_tslice_folder', type=str, 
                        help='folder where collapsed features from each tslice are stored')
    parser.add_argument('--tslice_folder', type=str, 
                        help='folder where features filtered by tslice are stored')
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
    y_train_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_train.csv.gz'))
    
    y_dict_file = os.path.join(args.clf_train_test_split_dir, 'y_dict.json')
    
    # import the y dict to get the id cols
    y_dict = load_data_dict_json(y_dict_file)
    id_cols = parse_id_cols(y_dict)
    
    tslice_folders = os.path.join(args.tslice_folder, 'TSLICE=')
    collapsed_tslice_folders = os.path.join(args.collapsed_tslice_folder, 'TSLICE=')
    outcome_col = args.outcome_column_name
    y_test_ids_df = y_test_df[id_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)
    y_train_ids_df = y_train_df[id_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)
    
    # get demographics csv and data_dict
    # for each patient get their vitals, labs, demographics
    _,_,_,_,demographics_df, demographics_data_dict,_,_,_,_ = get_preprocessed_data(args.preproc_data_dir)
    
    clf_models_dir = args.clf_models_dir
    
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    perf_df = pd.DataFrame()
    tslices_list = ["90%"]
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
        train_vitals_df = pd.merge(collapsed_vitals_df, y_train_ids_df, on=id_cols)
        train_labs_df = pd.merge(collapsed_labs_df, y_train_ids_df, on=id_cols)
        train_mews_df = pd.merge(mews_df, y_train_ids_df, on=id_cols)        
        
        
        if args.include_medications=='True':
            collapsed_medications_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'CollapsedMedicationsPerSequence.csv.gz'))
            collapsed_medications_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_CollapsedMedicationsPerSequence.json'))
            test_medications_df = pd.merge(collapsed_medications_df, y_test_ids_df, on=id_cols)
            train_medications_df = pd.merge(collapsed_medications_df, y_train_ids_df, on=id_cols)
            
            # merge them
            test_collapsed_features_df = pd.merge(pd.merge(test_vitals_df, test_labs_df, 
                                                           on=id_cols, how='inner'),
                                                  test_medications_df,on=id_cols, how='inner')
            
            train_collapsed_features_df = pd.merge(pd.merge(train_vitals_df, train_labs_df, 
                                                           on=id_cols, how='inner'),
                                                  train_medications_df,on=id_cols, how='inner')
            data_dicts_list = [collapsed_vitals_data_dict, 
                               collapsed_labs_data_dict,
                               collapsed_medications_data_dict,
                               demographics_data_dict]
            
        else :
            test_collapsed_features_df = pd.merge(test_vitals_df, test_labs_df, 
                                                           on=id_cols, how='inner')

            train_collapsed_features_df = pd.merge(train_vitals_df, train_labs_df, 
                                                           on=id_cols, how='inner')            
            
            data_dicts_list = [collapsed_vitals_data_dict,
                               collapsed_labs_data_dict,
                               demographics_data_dict]
        
        test_features_df = pd.merge(test_collapsed_features_df, demographics_df, on=id_cols, how='inner')
        train_features_df = pd.merge(train_collapsed_features_df, demographics_df, on=id_cols, how='inner')
        
        if p==0:
            features_dict = merge_data_dicts(data_dicts_list)
        
        test_outcomes_df = pd.merge(test_features_df[id_cols], outcomes_df, on=id_cols, how='inner')
        train_outcomes_df = pd.merge(train_features_df[id_cols], outcomes_df, on=id_cols, how='inner')
        
        # get the features in blocks of 24 hours start from the first admisssion ever recorded until the last admission
        split_list = ['train', 'test']
        split_features_df_list = [train_features_df, test_features_df]
        split_outcomes_df_list = [train_outcomes_df, test_outcomes_df]
        
        for split, split_features_df, split_outcomes_df in zip(split_list, split_features_df_list, split_outcomes_df_list):
        
            split_features_df_sorted = split_features_df.sort_values(by='admission_timestamp')
            first_admission_timestamp = pd.to_datetime(split_features_df_sorted['admission_timestamp'].values[0])
            last_admission_timestamp = pd.to_datetime(split_features_df_sorted['admission_timestamp'].values[-1])
            total_days = np.ceil((last_admission_timestamp - first_admission_timestamp).total_seconds()/(3600*24)).astype(int)

            timestamp_bins = [first_admission_timestamp]
            bin_length = 168
#             bin_length = 192
            for day in range(total_days):
                timestamp_bins.append(timestamp_bins[day] + datetime.timedelta(hours=bin_length))

            split_features_df_per_bin_list = []
            split_outcomes_df_per_bin_list = []
            admissions_per_bin_np = np.zeros(len(timestamp_bins)-1)
            deteriorations_per_bin_np = np.zeros(len(timestamp_bins)-1)
            alarms_per_bin_np = np.zeros(len(timestamp_bins)-1)
            true_alarms_per_bin_np = np.zeros(len(timestamp_bins)-1)
            false_alarms_per_bin_np = np.zeros(len(timestamp_bins)-1)
            missed_alarms_per_bin_np = np.zeros(len(timestamp_bins)-1)


            admission_timestamps = pd.to_datetime(split_features_df['admission_timestamp'])
            clinical_deterioration_timestamps = pd.to_datetime(split_outcomes_df['clinical_deterioration_timestamp'])

            feature_cols = parse_feature_cols(features_dict['schema'])


            for bin_ind in range(len(timestamp_bins)-1):
                curr_bin_split_outcomes_df = split_outcomes_df[(clinical_deterioration_timestamps>=timestamp_bins[bin_ind]) &
                                                             (clinical_deterioration_timestamps<=timestamp_bins[bin_ind+1])]
                curr_bin_split_features_df = pd.merge(curr_bin_split_outcomes_df[id_cols], split_features_df, on=id_cols, how='inner')

                split_features_df_per_bin_list.append(curr_bin_split_features_df)
                split_outcomes_df_per_bin_list.append(curr_bin_split_outcomes_df)

                if len(curr_bin_split_features_df)>0:
                    # get performance metrics
                    x_split = curr_bin_split_features_df[feature_cols].values
                    y_split = curr_bin_split_outcomes_df[outcome_col].values

                    # load the scaler
                    scaler = pickle.load(open(os.path.join(clf_models_dir, 'scaler.pkl'), 'rb'))
                    x_split_transformed = scaler.transform(x_split)

                    # load classifier
                    if bin_ind==0:

                        # get the best model minimizining bce loss
                        bce_filename_aka = 'skorch_mlp*cross_entropy_loss*history.json'
                        best_model_file_bce = get_best_model(clf_models_dir, bce_filename_aka)

                        # plot training plots of best model
                        plot_best_model_training_plots(best_model_file_bce, 'MLP_minimizing_cross_entropy')

                        best_model_prefix_bce = best_model_file_bce.split('/')[-1].replace('history.json', '')   
                        best_model_n_hiddens_bce = int(best_model_prefix_bce.split('n_hiddens=')[-1]) 
                        # load model minimizing BCE loss
                        skorch_mlp_bce = SkorchMLP(n_features=x_split.shape[1],
                                                                n_hiddens=best_model_n_hiddens_bce)
                        skorch_mlp_bce.initialize()

                        skorch_mlp_bce.load_params(f_params=os.path.join(clf_models_dir,
                                                              best_model_prefix_bce+'params.pt'))
                        print('Evaluating with saved model : %s'%(os.path.join(clf_models_dir, best_model_prefix_bce)))       


                    for loss_name, model_clf in [('cross_entropy_loss', skorch_mlp_bce)]:
                        print('Evaluating skorch lr trained minimizing %s on tslice=%s between timestamps %s and %s'%(loss_name,
                                                                                                                      tslice,
                                                                                                                      timestamp_bins[bin_ind],
                                                                                                                      timestamp_bins[bin_ind+1]))

                        y_pred = model_clf.predict(x_split_transformed).flatten()
                        y_pred_proba = model_clf.predict_proba(x_split_transformed)[:, 1]

                        TP = np.sum(np.logical_and(y_split == 1, y_pred == 1))
                        FP = np.sum(np.logical_and(y_split == 0, y_pred == 1))
                        TN = np.sum(np.logical_and(y_split == 0, y_pred == 0))
                        FN = np.sum(np.logical_and(y_split == 1, y_pred == 0))

                        admissions_per_bin_np[bin_ind] = len(y_split)
                        deteriorations_per_bin_np[bin_ind] = y_split.sum()
                        alarms_per_bin_np[bin_ind] = y_pred.sum()
                        true_alarms_per_bin_np[bin_ind] = TP
                        false_alarms_per_bin_np[bin_ind] = FP
                        missed_alarms_per_bin_np[bin_ind] = FN


            f, axs = plt.subplots(2, 2, figsize=(20, 12))
            axs_list = axs.flatten()

            bins = np.arange(1, 250, 15)
            xtick_labels = [str(i) for i in bins]
            axs_list[0].hist(admissions_per_bin_np, bins = bins)
            axs_list[0].set_title('Distribution of admissions in every %d hours'%bin_length)
            axs_list[0].set_xlabel('Number of admissions')
            axs_list[0].set_ylabel('Counts')
            axs_list[0].set_xticks(bins)
            axs_list[0].set_xticklabels(xtick_labels)

            bins = np.arange(1, 20, 1)
            xtick_labels = [str(i) for i in bins]
            axs_list[1].hist(deteriorations_per_bin_np, bins = bins)
            axs_list[1].set_title('Distribution of deteriorations in every %d hours'%bin_length)
            axs_list[1].set_xlabel('Number of deteriorations')
            axs_list[1].set_ylabel('Counts')            
            axs_list[1].set_xticks(bins)
            axs_list[1].set_xticklabels(xtick_labels)

            bins = np.arange(1, 10, 1)
            xtick_labels = [str(i) for i in bins]
            axs_list[2].hist(alarms_per_bin_np, bins = bins)
            axs_list[2].set_title('Distribution of alarms in every %d hours'%bin_length)
            axs_list[2].set_xlabel('Number of alarms')
            axs_list[2].set_ylabel('Counts')
            axs_list[2].set_xticks(bins)
            axs_list[2].set_xticklabels(xtick_labels)

            bins = np.arange(1, 10, 1)
            xtick_labels = [str(i) for i in bins]
            axs_list[3].hist(false_alarms_per_bin_np, bins = bins)
            axs_list[3].set_title('Distribution of false alarms in every %d hours'%bin_length)
            axs_list[3].set_xlabel('Number of false alarms')
            axs_list[3].set_ylabel('Counts')
            axs_list[3].set_xticks(bins)
            axs_list[3].set_xticklabels(xtick_labels)
            
            plt.suptitle('Prospective Evaluation of Alarm Counts with MLP on %s Set \n (Start date : %s, End date : %s)'%(split.capitalize(),
                                                                                                             first_admission_timestamp,
                                                                                                             last_admission_timestamp),
                         fontsize=22)
            f.savefig('skorch_mlp_%s_alarm_distributions.png'%split)



                




       
               
