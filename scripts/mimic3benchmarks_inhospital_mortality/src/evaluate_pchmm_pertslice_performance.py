'''
Evaluate performance of a single trained classifier on multiple patient-stay-slices
'''
import os
import numpy as np
import pandas as pd
from joblib import dump, load
import sys
import torch
sys.path.append(os.path.join(os.path.abspath('../'), 'src'))

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'rnn'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'PC-VAE'))

#import LR model before importing other packages because joblib files act weird when certain packages are loaded
from feature_transformation import *
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve)
from utils import load_data_dict_json
from dataset_loader import TidySequentialDataCSVLoader
# from RNNBinaryClassifier import RNNBinaryClassifier
import ast
import random
# from impute_missing_values_and_normalize_data import get_time_since_last_observed_features
from pcvae.datasets.toy import toy_line, custom_dataset
from pcvae.models.hmm import HMM
from scipy.special import softmax


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
    
    features_dict = load_data_dict_json(os.path.join(args.preproc_data_dir, 'Spec_FeaturesPerTimestep.json')) 
    time_col = 'hours_in'
    
#     normalization_estimates_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'normalization_estimates.csv'))
    
#     clf_models_dir = os.path.join(args.clf_models_dir, 'current_best_model')
    clf_models_dir=args.clf_models_dir

    # predict on each tslice
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    perf_df = pd.DataFrame()
    
    print('Evaluating  at tslices stored in : %s'%tslice_folders)
    for p, tslice in enumerate(tslices_list):
        tslice_folder = tslice_folders + tslice

        outcomes_df = pd.read_csv(os.path.join(tslice_folder,
                                               'outcomes_filtered_%s_hours.csv'%tslice))
    
        features_df = pd.read_csv(os.path.join(tslice_folder,
                                               'features_before_death_filtered_%s_hours.csv'%tslice))
        
        test_outcomes_df = pd.merge(outcomes_df, y_test_ids_df, on=id_cols, how='inner')
        test_features_df = pd.merge(features_df, y_test_ids_df, on=id_cols, how='inner')
        test_features_df.sort_values(by=id_cols+[time_col], inplace=True)
        test_outcomes_df.sort_values(by=id_cols, inplace=True)
        
#         print('Adding missing values mask as features...')
        feature_cols = parse_feature_cols(features_dict)
#         for feature_col in feature_cols:
#             test_features_df.loc[:, 'mask_'+feature_col] = (~test_features_df[feature_col].isna())*1.0
            
#         print('Adding time since last missing value is observed as features...')
#         test_features_df = get_time_since_last_observed_features(test_features_df, id_cols, time_col, feature_cols)
        
        # impute missing values in the test features
#         test_features_df = test_features_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()

#         for feature_col in feature_cols:
#             test_features_df[feature_col].fillna(test_features_df[feature_col].mean(), inplace=True)  
        
        
        # normalize the data
#         for feature_col in feature_cols_with_mask_features:
#             feat_ind = normalization_estimates_df.feature==feature_col 
#             numerator_scaling = np.asarray(normalization_estimates_df[feat_ind]['numerator_scaling'])
#             denominator_scaling = np.asarray(normalization_estimates_df[feat_ind]['denominator_scaling'])
#             test_features_df[feature_col] = (test_features_df[feature_col] - numerator_scaling)/denominator_scaling
            
#             if test_features_df[feature_col].isna().sum()>0:
#                 test_features_df[feature_col]=numerator_scaling[0]
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
        
        N,T,F = x_test.shape
        
        # load the test data into dataset object for evaluation
        x_test = np.expand_dims(x_test, 1)
        key_list = ['train', 'valid', 'test']
        data_dict = dict.fromkeys(key_list)
        data_dict['train'] = (x_test[:2], y_test[:2])
        data_dict['valid'] = (x_test[:2], y_test[:2])
        data_dict['test'] = (x_test, y_test)
        data = custom_dataset(data_dict=data_dict)
        
        
        # load classifier
        n_states = 12 
        model = HMM(states=n_states,
                    observation_dist='NormalWithMissing',
                    predictor_dist='Categorical')

        model.build(data)
        model_filename = 'pchmm-lr=0.001-seed=890-batch_size=128-weights.h5'
        model.model.load_weights(os.path.join(args.clf_models_dir, model_filename))
        
        
        x_test, y_test = data.test().numpy()
        # get the beliefs of the test set
        z_test = model.hmm_model.predict(x_test)
        y_test_pred_proba = softmax(model._predictor(z_test).distribution.logits.numpy(), axis=1)
        
        print('Evaluating PCHMM on tslice=%s'%(tslice))
        roc_auc_np = np.zeros(len(random_seed_list))
        balanced_accuracy_np = np.zeros(len(random_seed_list))
        log_loss_np = np.zeros(len(random_seed_list))
        avg_precision_np = np.zeros(len(random_seed_list))
        for k, seed in enumerate(random_seed_list):
            random.seed(int(seed))
            rnd_inds = random.sample(range(x_test.shape[0]), int(0.8*x_test.shape[0])) 
            curr_y_test = y_test[rnd_inds]
            curr_x_test = x_test[rnd_inds, :]
            curr_y_pred = np.argmax(y_test_pred_proba[rnd_inds], -1)
            curr_y_pred_proba = y_test_pred_proba[rnd_inds]
            
            roc_auc_np[k] = roc_auc_score(curr_y_test, curr_y_pred_proba)
            balanced_accuracy_np[k] = balanced_accuracy_score(np.argmax(curr_y_test,-1), curr_y_pred)
            log_loss_np[k] = log_loss(curr_y_test, curr_y_pred_proba, normalize=True) / np.log(2)
            avg_precision_np[k] = average_precision_score(curr_y_test, curr_y_pred_proba)
        
        print('tslice : %s, ROC-AUC : %.2f'%(tslice, np.percentile(roc_auc_np, 50)))
        
        for prctile in prctile_vals:
            row_dict = dict()
            row_dict['model'] = 'PCHMM'
            row_dict['percentile'] = prctile
            row_dict['tslice'] = tslice
            row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
            row_dict['balanced_accuracy'] = np.percentile(balanced_accuracy_np, prctile)
            row_dict['log_loss'] = np.percentile(log_loss_np, prctile)
            row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)

            perf_df = perf_df.append(row_dict, ignore_index=True)      
    
    
    perf_csv = os.path.join(args.output_dir, 'pchmm_pertslice_performance.csv')
    print('Saving PCHMM per-tslice performance to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)