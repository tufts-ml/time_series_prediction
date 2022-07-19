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
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'MixMatch', 'MixMatch-pytorch'))

#import LR model before importing other packages because joblib files act weird when certain packages are loaded
from feature_transformation import *
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score)

from utils import load_data_dict_json
from dataset_loader import TidySequentialDataCSVLoader
# from RNNBinaryClassifier import RNNBinaryClassifier
import ast
import random
# from impute_missing_values_and_normalize_data import get_time_since_last_observed_features
from scipy.special import softmax
import glob
import torch.utils.data as data
from RNNBinaryClassifierModule import RNNBinaryClassifierModule

def create_model(ema=False, n_hiddens=32):
    model = RNNBinaryClassifierModule(rnn_type='GRU', 
                                      n_inputs=D, 
                                      n_hiddens=n_hiddens, 
                                      n_layers=1,
                                      dropout_proba=0.0, 
                                      dropout_proba_non_recurrent=0.0, 
                                      bidirectional=False)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


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
    
    f.savefig('MixMatch-semi-supervised-best-model-training-plots.png')

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
    for f in training_files: 
        perf_df = pd.read_csv(f)
        aucroc_per_fit_list.append(perf_df.valid_auroc.values[-1])
        auprc_per_fit_list.append(perf_df.valid_auprc.values[-1])
    
    best_fit_ind = np.argmax(auprc_per_fit_list)
    best_model_file = training_files[best_fit_ind]    
    
    return best_model_file



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

    clf_models_dir = args.clf_models_dir
    ## get the test data
    x_train_np_filename = os.path.join(args.clf_train_test_split_dir, 'X_train.npy')
    x_valid_np_filename = os.path.join(args.clf_train_test_split_dir, 'X_valid.npy')
    x_test_np_filename = os.path.join(args.clf_train_test_split_dir, 'X_test.npy')
    y_train_np_filename = os.path.join(args.clf_train_test_split_dir, 'y_train.npy')
    y_valid_np_filename = os.path.join(args.clf_train_test_split_dir, 'y_valid.npy')
    y_test_np_filename = os.path.join(args.clf_train_test_split_dir, 'y_test.npy')    
    
    
    X_train_fs = np.load(x_train_np_filename)
    X_test = np.load(x_test_np_filename)
    y_train = np.load(y_train_np_filename)
    y_test = np.load(y_test_np_filename)
    X_valid = np.load(x_valid_np_filename)
    y_valid = np.load(y_valid_np_filename)
    
    
    features_dict = load_data_dict_json(os.path.join(args.preproc_data_dir, 'Spec_FeaturesPerTimestep.json')) 
    feature_cols = parse_feature_cols(features_dict)
    
    N,T,D = X_test.shape
    print('number of data points : %d\nnumber of time points : %s\nnumber of features : %s\n'%(N,T,D))
    
    X_train = X_train_fs.copy()
    for d in range(D):
        # impute missing values by mean filling using estimates from full training set
        fill_vals = np.nanmean(X_train_fs[:, :, d])
        
        X_train[:, :, d] = np.nan_to_num(X_train[:, :, d], nan=fill_vals)
        X_valid[:, :, d] = np.nan_to_num(X_valid[:, :, d], nan=fill_vals)
        X_test[:, :, d] = np.nan_to_num(X_test[:, :, d], nan=fill_vals)        
        
        
        # min max normalization
#         den = np.nanmax(X_train[:, :, d])-np.nanmin(X_train[:, :, d])
#         X_train[:, :, d] = (X_train[:, :, d] - np.nanmin(X_train[:, :, d]))/den
#         X_valid[:, :, d] = (X_valid[:, :, d] - np.nanmin(X_train[:, :, d]))/den
#         X_test[:, :, d] = (X_test[:, :, d] - np.nanmin(X_train[:, :, d]))/den
        
        # zscore normalization
        den = np.nanstd(X_train_fs[:, :, d])
        X_train[:, :, d] = (X_train[:, :, d] - np.nanmean(X_train_fs[:, :, d]))/den
        X_valid[:, :, d] = (X_valid[:, :, d] - np.nanmean(X_train_fs[:, :, d]))/den
        X_test[:, :, d] = (X_test[:, :, d] - np.nanmean(X_train_fs[:, :, d]))/den    

    
    test_data, test_label = torch.Tensor(X_test), torch.Tensor(y_test[:, np.newaxis])
    test_set = data.TensorDataset(test_data, test_label)  
    test_loader = data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=0)
    perf_df = pd.DataFrame()
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    for perc_labelled in ['1.2', '3.7', '11.1', '33.3', '100']:

        saved_model_files_aka = os.path.join(clf_models_dir, 
                                             "MixMatch*perc_labelled=%s*.csv"%(perc_labelled))
        best_model_file = get_best_model_file(saved_model_files_aka).replace('.csv', '.pt')
        print('Evaluating Mixmatch model with perc_labelled =%s \nfile=%s'%(perc_labelled, best_model_file))
        model = create_model()
        
        try:
            model.load_state_dict(torch.load(best_model_file))
        except:
            model = create_model(n_hiddens=128)
            model.load_state_dict(torch.load(best_model_file))
        model.eval()
        
        with torch.no_grad():      
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                outputs = model(inputs)
                batch_size = inputs.size(0)
                # Transform label to one-hot
                targets = torch.zeros(batch_size, 2).scatter_(1, targets.view(-1,1).long(), 1)            
                test_auroc = roc_auc_score(targets, outputs)
                test_auprc = average_precision_score(targets[:, 1], outputs[:, 1])
                
        
#         bootstrapping to get CI on metrics
        roc_auc_np = np.zeros(len(random_seed_list))
        avg_precision_np = np.zeros(len(random_seed_list))

        for k, seed in enumerate(random_seed_list):
            random.seed(int(seed))
            rnd_inds = random.sample(range(targets.shape[0]), int(0.8*targets.shape[0])) 
            curr_y_test = targets[rnd_inds]
#             curr_x_test = inputs[rnd_inds, :]
            curr_y_pred = np.argmax(outputs[rnd_inds], -1)
            curr_y_pred_proba = outputs[rnd_inds]

            roc_auc_np[k] = roc_auc_score(curr_y_test, curr_y_pred_proba)
            avg_precision_np[k] = average_precision_score(curr_y_test[:, 1], curr_y_pred_proba[:, 1])

        print('perc_labelled = %s, \nMedian ROC-AUC : %.3f'%(perc_labelled, np.percentile(roc_auc_np, 50)))
        print('Median average precision : %.3f'%np.percentile(avg_precision_np, 50))
        

        for prctile in prctile_vals:
            row_dict = dict()
            row_dict['model'] = 'MixMatch'
            row_dict['percentile'] = prctile
            row_dict['perc_labelled'] = perc_labelled
            row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
            row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)
            perf_df = perf_df.append(row_dict, ignore_index=True)      

    perf_csv = os.path.join(args.output_dir, 'MixMatch_performance.csv')
    print('Saving Mixmatch performance to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)
