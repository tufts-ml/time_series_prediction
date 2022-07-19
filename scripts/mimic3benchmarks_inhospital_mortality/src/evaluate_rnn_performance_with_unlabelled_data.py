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

#import LR model before importing other packages because joblib files act weird when certain packages are loaded
from feature_transformation import *
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve, recall_score, precision_score)
from utils import load_data_dict_json
from dataset_loader import TidySequentialDataCSVLoader
from RNNBinaryClassifier import RNNBinaryClassifier
import ast
import random
from impute_missing_values_and_normalize_data import get_time_since_last_observed_features
import glob


def get_best_model(saved_model_files_aka):
    
    training_files = glob.glob(saved_model_files_aka)
    aucroc_per_fit_list = []
    
    for i, tr_file in enumerate(training_files):
        train_df = pd.DataFrame(json.load(open(tr_file))) 
        aucroc_per_fit_list.append(train_df.aucroc_score_valid.values[-1])
    
    aucroc_per_fit_np = np.array(aucroc_per_fit_list)
    aucroc_per_fit_np[np.isnan(aucroc_per_fit_np)]=0
    
    best_model_ind = np.argmax(aucroc_per_fit_np)
    best_model_file = training_files[best_model_ind] 
    
    # get parameters of best model
    model_params = best_model_file.split('/')[-1].split('-')
    hiddens_param = [f for f in model_params if 'hiddens' in f][0] 
    layers_param = [f for f in model_params if 'layers' in f][0]
    n_hiddens = int(hiddens_param.split('=')[-1])
    n_layers = int(layers_param.split('=')[-1]) 
    
    # make the plots
    best_model_train_df = pd.DataFrame(json.load(open(best_model_file))) 
    train_loss = best_model_train_df.train_loss
    valid_loss = best_model_train_df.valid_loss
    train_auroc = best_model_train_df.aucroc_score_train
    valid_auroc = best_model_train_df.aucroc_score_valid
    epochs = best_model_train_df.epoch
    
    f, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True) 
    axs[0].plot(epochs, train_loss, label='loss (train)')
    axs[0].plot(epochs, valid_loss, label='loss (valid)')
    axs[0].legend()
    axs[1].plot(epochs, train_auroc, label = 'ROC (Train)')
    axs[1].plot(epochs, valid_auroc, label = 'ROC (Valid)')
    axs[1].set_xlabel('Epochs')
    axs[1].legend()
    axs[0].set_title(best_model_file.split('/')[-1])
    
    f.savefig('rnn_best_model_training_plots_n_layers=%s.png'%str(n_layers))
    return best_model_file, n_hiddens, n_layers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_models_dir', default=None, type=str,
                        help='Directory where classifier models are saved')
    parser.add_argument('--clf_train_test_split_dir', default=None, type=str,
                        help='Directory where the train-test split data for the classifier is saved')
    parser.add_argument('--tslice_folder', type=str, 
                        help='folder where features filtered by tslice are stored')
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
    x_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'x_dict_imputed.json')
    x_test_dict = load_data_dict_json(x_test_dict_file)
    feature_cols_with_mask_features = parse_feature_cols(x_test_dict)
    
    # import the y dict to get the id cols
    y_test_dict = load_data_dict_json(y_test_dict_file)
    id_cols = parse_id_cols(y_test_dict)

    tslice_folders = os.path.join(args.tslice_folder, 'TSLICE=')
    outcome_col_name = args.outcome_column_name
    tslices_list = ['24']
    y_test_ids_df = y_test_df[id_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)
    
    features_dict = load_data_dict_json(os.path.join(args.preproc_data_dir, 'Spec_FeaturesPerTimestep.json')) 
    time_col = 'hours_in'
    
    clf_models_dir=args.clf_models_dir
    
    normalization_estimates_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'normalization_estimates.csv'))
    
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
        
        print('Adding missing values mask as features...')
        feature_cols = parse_feature_cols(features_dict)
        for feature_col in feature_cols:
            test_features_df.loc[:, 'mask_'+feature_col] = (~test_features_df[feature_col].isna())*1.0
            
        print('Adding time since last missing value is observed as features...')
        test_features_df = get_time_since_last_observed_features(test_features_df, id_cols, time_col, feature_cols)
        
        # impute missing values in the test features
        test_features_df = test_features_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()

        for feature_col in feature_cols:
            test_features_df[feature_col].fillna(test_features_df[feature_col].mean(), inplace=True)  
        
        
        # normalize the data
        for feature_col in feature_cols_with_mask_features:
            feat_ind = normalization_estimates_df.feature==feature_col 
            numerator_scaling = np.asarray(normalization_estimates_df[feat_ind]['numerator_scaling'])
            denominator_scaling = np.asarray(normalization_estimates_df[feat_ind]['denominator_scaling'])
            test_features_df[feature_col] = (test_features_df[feature_col] - numerator_scaling)/denominator_scaling
            
            if test_features_df[feature_col].isna().sum()>0:
                test_features_df[feature_col]=numerator_scaling[0]
    
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
        
        for perc_labelled in ['1', '5', '10', '15', '20', '50', '70', '75', '90', '100']:
            for layers in ['2']:
                saved_model_files_aka = os.path.join(clf_models_dir, '*layers=%s-*perc_labelled=%shistory.json'%(layers, perc_labelled))

                best_model_file, n_hiddens, n_layers = get_best_model(saved_model_files_aka)
                rnn = RNNBinaryClassifier(module__rnn_type='GRU',
                                  module__n_layers=n_layers,
                                  module__n_hiddens=n_hiddens,
                                  module__n_inputs=x_test.shape[-1])
                rnn.initialize()
                rnn.load_params(f_params=best_model_file.replace('history.json', 'params.pt'),
                                f_optimizer=best_model_file.replace('history.json', 'optimizer.pt'),
                                f_history=best_model_file)
                print('Evaluating with saved model : %s'%(best_model_file))

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

                print('layers : %s, perc labelled : %s, ROC-AUC : %.4f'%(layers, perc_labelled, np.percentile(roc_auc_np, 50)))
                print('layers : %s, perc labelled : %s, AUPRC : %.4f'%(layers, perc_labelled, np.percentile(avg_precision_np,
                                                                                                            50)))
                for prctile in prctile_vals:
                    row_dict = dict()
                    row_dict['model'] = 'GRU-layers=%s'%layers
                    row_dict['perc_labelled'] = perc_labelled
                    row_dict['percentile'] = prctile
                    row_dict['tslice'] = tslice
                    row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
                    row_dict['balanced_accuracy'] = np.percentile(balanced_accuracy_np, prctile)
                    row_dict['log_loss'] = np.percentile(log_loss_np, prctile)
                    row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)
                    row_dict['n_layers'] = layers

                    perf_df = perf_df.append(row_dict, ignore_index=True)      

    perf_csv = os.path.join(args.output_dir, 'rnn_performance_with_unlabelled_examples.csv')
    print('Saving RNN with unlabelled examples performance to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)