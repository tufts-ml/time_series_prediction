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
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'PC-HMM'))

#import LR model before importing other packages because joblib files act weird when certain packages are loaded
from feature_transformation import *
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve, recall_score, precision_score)
from utils import load_data_dict_json
from dataset_loader import TidySequentialDataCSVLoader
# from RNNBinaryClassifier import RNNBinaryClassifier
import ast
import random
# from impute_missing_values_and_normalize_data import get_time_since_last_observed_features
from pcvae.datasets.toy import toy_line, custom_dataset
from pcvae.models.hmm import HMM
from scipy.special import softmax
import glob


def plot_best_model_training_plots(best_model_file):
    train_perf_df = pd.read_csv(best_model_file)
    f, axs = plt.subplots(4,1)
    axs[0].plot(train_perf_df.epochs, train_perf_df.hmm_model_loss, label = 'train hmm loss')
    axs[0].plot(train_perf_df.epochs, train_perf_df.val_hmm_model_loss, label = 'val hmm loss')
    axs[0].legend()
    
    axs[1].plot(train_perf_df.epochs, train_perf_df.predictor_loss, label = 'train predictor loss')
    axs[1].plot(train_perf_df.epochs, train_perf_df.val_predictor_loss, label = 'val predictor loss')
    axs[1].legend()
    
    axs[2].plot(train_perf_df.epochs, train_perf_df.loss, label = 'train loss')
    axs[2].plot(train_perf_df.epochs, train_perf_df.val_loss, label = 'val loss')
    axs[2].legend()
    
    axs[3].plot(train_perf_df.epochs, train_perf_df.predictor_AUC, label = 'train AUC')
    axs[3].plot(train_perf_df.epochs, train_perf_df.val_predictor_AUC, label = 'val AUC')
    axs[3].legend()
    
    f.savefig('semi-supervised-best-model-training-plots.png')

# def get_best_model_file(saved_model_files_aka):
    
#     training_files = glob.glob(saved_model_files_aka)
#     aucroc_per_fit_list = []
#     loss_per_fit_list = []
    
#     for i, training_file in enumerate(training_files):
#         train_perf_df = pd.read_csv(training_file)
#         aucroc_per_fit_list.append(train_perf_df['val_predictor_AUC'].values[-1])
#         curr_lamb = int(training_file.split('lamb=')[1].replace('.csv', ''))
#         loss_per_fit_list.append((train_perf_df['val_predictor_loss'].values[-1])/curr_lamb)

#     aucroc_per_fit_np = np.array(aucroc_per_fit_list)
#     aucroc_per_fit_np[np.isnan(aucroc_per_fit_np)]=0

#     loss_per_fit_np = np.array(loss_per_fit_list)
#     loss_per_fit_np[np.isnan(loss_per_fit_np)]=10^8

# #    best_model_ind = np.argmax(aucroc_per_fit_np)
#     best_model_ind = np.argmin(loss_per_fit_np)

#     best_model_file = training_files[best_model_ind]
#     plot_best_model_training_plots(best_model_file)
# #     
#     # get the number of states of best file
#     best_fit_params = best_model_file.split('-')
#     n_states_param = [s for s in best_fit_params if 'n_states' in s][0]
#     n_states = int(n_states_param.split('=')[-1])
    
# #     from IPython import embed; embed()
#     return best_model_file, n_states


def get_best_model_file(saved_model_files_aka):
    
    training_files = glob.glob(saved_model_files_aka)
    aucroc_per_fit_list = []
    auprc_per_fit_list = []
#     loss_per_fit_list = []
    
    for i, training_file in enumerate(training_files):
        train_perf_df = pd.read_csv(training_file)
        auprc_per_fit_list.append(train_perf_df['valid_AUPRC'].values[-1])
        curr_lamb = int(training_file.split('lamb=')[1].replace('.csv', ''))
#         loss_per_fit_list.append((train_perf_df['val_predictor_loss'].values[-1])/curr_lamb)

    auprc_per_fit_np = np.array(auprc_per_fit_list)
    auprc_per_fit_np[np.isnan(auprc_per_fit_np)]=0

#     loss_per_fit_np = np.array(loss_per_fit_list)
#     loss_per_fit_np[np.isnan(loss_per_fit_np)]=10^8

#    best_model_ind = np.argmax(aucroc_per_fit_np)
    best_model_ind = np.argmax(auprc_per_fit_np)

    best_model_file = training_files[best_model_ind]
#     plot_best_model_training_plots(best_model_file)
#     
    # get the number of states of best file
    best_fit_params = best_model_file.split('-')
    n_states_param = [s for s in best_fit_params if 'n_states' in s][0]
    n_states = int(n_states_param.split('=')[-1])
    
#     from IPython import embed; embed()
    return best_model_file, n_states



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
    x_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'x_dict.json')
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
        
        for perc_labelled in ['5', '10', '20', '30', '60', '90']:
            for states in ['5', '10', '30', '60', '90']:
                for imputation_strategy in ["ffill", "mean", "no_imp"]:
                        
                    saved_model_files_aka = os.path.join(clf_models_dir, 
                                                         "final_perf_*semi-supervised-pchmm*imputation_strategy=%s-*perc_labelled=%s-*n_states=%s-*lamb=*.csv"%(imputation_strategy, perc_labelled, states))
                    best_model_file, n_states = get_best_model_file(saved_model_files_aka)
                    best_model_weights = best_model_file.replace('.csv', '-weights.h5').replace('final_perf_', '')

                    # load classifier
                    model = HMM(states=n_states,
                                observation_dist='NormalWithMissing',
                                predictor_dist='Categorical')

                    model.build(data)
                    model.model.load_weights(best_model_weights)


                    x_test, y_test = data.test().numpy()
                    # get the beliefs of the test set
                    z_test = model.hmm_model.predict(x_test)
                    y_test_pred_proba = model._predictor.predict(z_test)

                    print('Evaluating PCHMM with imputation = %s perc_labelled =%s and states =%s with model %s'%(imputation_strategy, perc_labelled, states, best_model_file))

                    # bootstrapping to get CI on metrics
                    roc_auc_np = np.zeros(len(random_seed_list))
#                     balanced_accuracy_np = np.zeros(len(random_seed_list))
                    log_loss_np = np.zeros(len(random_seed_list))
                    avg_precision_np = np.zeros(len(random_seed_list))
#                     precision_np = np.zeros(len(random_seed_list))
#                     recall_np = np.zeros(len(random_seed_list))

                    for k, seed in enumerate(random_seed_list):
                        random.seed(int(seed))
                        rnd_inds = random.sample(range(x_test.shape[0]), int(0.8*x_test.shape[0])) 
                        curr_y_test = y_test[rnd_inds]
                        curr_x_test = x_test[rnd_inds, :]
                        curr_y_pred = np.argmax(y_test_pred_proba[rnd_inds], -1)
                        curr_y_pred_proba = y_test_pred_proba[rnd_inds]

                        roc_auc_np[k] = roc_auc_score(curr_y_test, curr_y_pred_proba)
#                         balanced_accuracy_np[k] = balanced_accuracy_score(np.argmax(curr_y_test,-1), curr_y_pred)
#                         precision_np[k] = precision_score(np.argmax(curr_y_test,-1), curr_y_pred)
#                         recall_np[k] = recall_score(np.argmax(curr_y_test,-1), curr_y_pred)
                        log_loss_np[k] = log_loss(curr_y_test, curr_y_pred_proba, normalize=True) / np.log(2)
                        avg_precision_np[k] = average_precision_score(curr_y_test, curr_y_pred_proba)


                    print('imputation = %s, perc_labelled = %s, states = %s, ROC-AUC : %.2f'%(imputation_strategy, perc_labelled, states, np.percentile(roc_auc_np, 50)))
                    print('Median average precision : %.3f'%np.median(avg_precision_np))
            #         print('Median precision score : %.3f'%np.median(precision_np))
            #         print('Median recall score : %.3f'%np.median(recall_np))
            #         print('Median balanced accuracy score : %.3f'%np.median(balanced_accuracy_np))

                    for prctile in prctile_vals:
                        row_dict = dict()
                        row_dict['model'] = 'PCHMM-n_states=%s'%(states)
                        row_dict['percentile'] = prctile
        #                     row_dict['lambda'] = lamb
                        row_dict['perc_labelled'] = perc_labelled
                        row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
#                         row_dict['balanced_accuracy'] = np.percentile(balanced_accuracy_np, prctile)
                        row_dict['log_loss'] = np.percentile(log_loss_np, prctile)
                        row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)
#                         row_dict['precision'] = np.percentile(precision_np, prctile)
#                         row_dict['recall'] = np.percentile(recall_np, prctile)
                        row_dict['n_states'] = states
                        row_dict['imputation_strategy'] = imputation_strategy

                        perf_df = perf_df.append(row_dict, ignore_index=True)      

    
    perf_csv = os.path.join(args.output_dir, 'semi_supervised_pchmm_performance.csv')
    print('Saving semi-supervised PCHMM performance to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)
