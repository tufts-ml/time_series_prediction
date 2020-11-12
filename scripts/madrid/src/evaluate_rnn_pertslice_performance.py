'''
Evaluate performance of a single trained classifier on multiple patient-stay-slices
'''
import os
import numpy as np
import pandas as pd
from joblib import dump, load
import sys
import torch
import skorch
sys.path.append(os.path.join(os.path.abspath('../'), 'src'))

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'rnn'))

#import LR model before importing other packages because joblib files act weird when certain packages are loaded
from feature_transformation import *
from merge_features_all_tslices import merge_data_dicts, get_all_features_data
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve)
from utils import load_data_dict_json
from dataset_loader import TidySequentialDataCSVLoader
from RNNBinaryClassifier import RNNBinaryClassifier
import ast
from filter_admissions_by_tslice import get_preprocessed_data
import random
from impute_missing_values import get_time_since_last_observed_features
from train_rnn_on_patient_stay_slice_sequences import normalize_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_models_dir', default=None, type=str,
                        help='Directory where classifier models are saved')
    parser.add_argument('--clf_train_test_split_dir', default=None, type=str,
                        help='Directory where the train-test split data for the classifier is saved')
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
    parser.add_argument('--output_dir', default=' ', type=str,
                       help='name of outcome column in test dataframe')
    
    
    args = parser.parse_args()

    
    ## get the test patient id's
    # get the test set's csv and dict
    y_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_test.csv'))
    y_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'y_dict.json')
    
    # get the x test feature columns
    x_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'x_dict.json')
    x_test_dict = load_data_dict_json(x_test_dict_file)
    feature_cols_with_mask_features = parse_feature_cols(x_test_dict)
    
    # import the y dict to get the id cols
    y_test_dict = load_data_dict_json(y_test_dict_file)
    id_cols = parse_id_cols(y_test_dict)

    tslice_folders = os.path.join(args.tslice_folder, 'TSLICE=')
    outcome_col_name = args.outcome_column_name
    tslices_list = args.evaluation_tslices.split(' ')
    y_test_ids_df = y_test_df[id_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)

    # get demographics csv and data_dict
    # for each patient get their vitals, labs, demographics
    _,labs_data_dict,_,vitals_data_dict, _, demographics_data_dict,_,_ = get_preprocessed_data(args.preproc_data_dir)
    time_col = parse_time_col(vitals_data_dict)

    # prctile_vals = [5, 50, 95]
    # random_seed_list = args.random_seed_list.split(' ')
    # perf_df = pd.DataFrame()
    
#    clf_models_dir = os.path.join(args.clf_models_dir, 'current_best_model')
    clf_models_dir=args.clf_models_dir

    # predict on each tslice
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    perf_df = pd.DataFrame()
    
    print('Evaluating  at tslices stored in : %s'%tslice_folders)
    for p, tslice in enumerate(tslices_list):
        tslice_folder = tslice_folders + tslice
        # get test set labs and vitals
        vitals_df = pd.read_csv(os.path.join(tslice_folder, 'vitals_before_icu_filtered_%s_hours.csv'%tslice))
        labs_df = pd.read_csv(os.path.join(tslice_folder, 'labs_before_icu_filtered_%s_hours.csv'%tslice))
        demographics_df = pd.read_csv(os.path.join(tslice_folder, 'demographics_before_icu_filtered_%s_hours.csv'%tslice))
        outcomes_df = pd.read_csv(os.path.join(tslice_folder,
                                               'clinical_deterioration_outcomes_filtered_%s_hours.csv'%tslice))
        test_vitals_df = pd.merge(vitals_df, y_test_ids_df, on=id_cols)
        test_labs_df = pd.merge(labs_df, y_test_ids_df, on=id_cols)
        test_demographics_df = pd.merge(demographics_df, y_test_ids_df, on=id_cols)

        # merge the labs, vitals and demographics to get a single features table
        test_features_df,test_features_dict = get_all_features_data(test_labs_df, labs_data_dict, 
                                                            test_vitals_df, vitals_data_dict, 
                                                            test_demographics_df, demographics_data_dict)


        test_outcomes_df = pd.merge(outcomes_df, y_test_ids_df, on=id_cols, how='inner')

        test_features_df.sort_values(by=id_cols+[time_col], inplace=True)
        test_outcomes_df.sort_values(by=id_cols, inplace=True)
        
        print('Adding missing values mask as features...')
        feature_cols = parse_feature_cols(test_features_dict)
        for feature_col in feature_cols:
            test_features_df.loc[:, 'mask_'+feature_col] = (~test_features_df[feature_col].isna())*1.0
            
        print('Adding time since last missing value is observed as features...')
        test_features_df = get_time_since_last_observed_features(test_features_df, id_cols, time_col, feature_cols)
        
        # impute missing values in the test features
        test_features_df = test_features_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()

#         for feature_col in feature_cols:
#             test_features_df[feature_col].fillna(test_features_df[feature_col].mean(), inplace=True)  
 
        
#         test_features_df = test_features_df[id_cols + [time_col] + feature_cols_with_mask_features].copy()
        
        # load test data with TidySequentialDataLoader
        test_vitals = TidySequentialDataCSVLoader(
            x_csv_path=test_features_df,
            y_csv_path=test_outcomes_df,
            x_col_names=feature_cols_with_mask_features,
            idx_col_names=id_cols,
            y_col_name=outcome_col_name,
            y_label_type='per_sequence'
        )    

        # predict on test data
        x_test, y_test = test_vitals.get_batch_data(batch_id=0)
        
        F = x_test.shape[2]
        # impute features with training set means
        per_feature_means = np.load(os.path.join(clf_models_dir, 'per_feature_means.npy'))
        for f in range(F):
            nan_inds = np.isnan(x_test[:,:,f])
            x_test[:,:,f][nan_inds] = per_feature_means[f]
            
        
        per_feature_scaling = np.load(os.path.join(clf_models_dir, 'per_feature_scaling.npy'))
        for f in range(F):
            x_test[:,:,f] = x_test[:,:,f]/per_feature_scaling[f]
        
        # load classifier
        if p==0:
            rnn = RNNBinaryClassifier(module__rnn_type='GRU',
                              module__n_layers=2,
                              module__n_hiddens=128,
                              module__n_inputs=x_test.shape[-1])
            rnn.initialize()
            best_model_prefix = 'hiddens=128-layers=2-lr=0.001-dropout=0.3-weight_decay=0-seed=1111'
            rnn.load_params(f_params=os.path.join(clf_models_dir,
                                                  best_model_prefix+'params.pt'),
                            f_optimizer=os.path.join(clf_models_dir,
                                                     best_model_prefix+'optimizer.pt'),
                            f_history=os.path.join(clf_models_dir,
                                                   best_model_prefix+'history.json'))
            print('Evaluating with saved model : %s'%(os.path.join(clf_models_dir, best_model_prefix)))

        print('Evaluating rnn on tslice=%s'%(tslice))
        roc_auc_np = np.zeros(len(random_seed_list))
        balanced_accuracy_np = np.zeros(len(random_seed_list))
        log_loss_np = np.zeros(len(random_seed_list))
        avg_precision_np = np.zeros(len(random_seed_list))
        for k, seed in enumerate(random_seed_list):
            random.seed(int(seed))
            rnd_inds = random.sample(range(x_test.shape[0]), int(0.8*x_test.shape[0])) 
            curr_y_test = y_test[rnd_inds]
            curr_x_test = x_test[rnd_inds, :]
            y_pred = np.argmax(rnn.predict_proba(curr_x_test), -1)
            y_pred_proba = rnn.predict_proba(curr_x_test)[:, 1]
            y_score = y_pred_proba

            roc_auc_np[k] = roc_auc_score(curr_y_test, y_score)
            balanced_accuracy_np[k] = balanced_accuracy_score(curr_y_test, y_pred)
            log_loss_np[k] = log_loss(curr_y_test, y_pred_proba, normalize=True) / np.log(2)
            avg_precision_np[k] = average_precision_score(curr_y_test, y_score)
        
        print('tslice : %s, ROC-AUC : %.2f'%(tslice, np.percentile(roc_auc_np, 50)))
        
        for prctile in prctile_vals:
            row_dict = dict()
            row_dict['model'] = 'RNN'
            row_dict['percentile'] = prctile
            row_dict['tslice'] = tslice
            row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
            row_dict['balanced_accuracy'] = np.percentile(balanced_accuracy_np, prctile)
            row_dict['log_loss'] = np.percentile(log_loss_np, prctile)
            row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)

            perf_df = perf_df.append(row_dict, ignore_index=True)      
    
    
    perf_csv = os.path.join(args.output_dir, 'rnn_pertslice_performance.csv')
    print('Saving RNN per-tslice performance to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)
