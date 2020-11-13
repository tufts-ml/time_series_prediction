import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                             balanced_accuracy_score, confusion_matrix, 
                             roc_auc_score, make_scorer)
from yattag import Doc
import matplotlib.pyplot as plt
DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'rnn'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'PC-VAE'))

from dataset_loader import TidySequentialDataCSVLoader
from feature_transformation import (parse_id_cols, parse_feature_cols)
from utils import load_data_dict_json
from joblib import dump

import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import glob
from numpy.random import RandomState
from pcvae.datasets.toy import toy_line, custom_dataset
from pcvae.models.hmm import HMM
from pcvae.datasets.base import dataset, real_dataset, classification_dataset
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description='PyTorch RNN with variable-length numeric sequences wrapper')
    parser.add_argument('--outcome_col_name', type=str, required=True)
    parser.add_argument('--train_csv_files', type=str, required=True)
    parser.add_argument('--test_csv_files', type=str, required=True)
    parser.add_argument('--data_dict_files', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of sequences per minibatch')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--validation_size', type=float, default=0.15,
                        help='validation split size')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='directory where trained model and loss curves over epochs are saved')
    parser.add_argument('--output_filename_prefix', type=str, default=None, 
                        help='prefix for the training history jsons and trained classifier')
    args = parser.parse_args()

    rs = RandomState(args.seed)

    x_train_csv_filename, y_train_csv_filename = args.train_csv_files.split(',')
    x_test_csv_filename, y_test_csv_filename = args.test_csv_files.split(',')
    x_dict, y_dict = args.data_dict_files.split(',')
    x_data_dict = load_data_dict_json(x_dict)

    # get the id and feature columns
    id_cols = parse_id_cols(x_data_dict)
    feature_cols = parse_feature_cols(x_data_dict)
    # extract data
    train_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_train_csv_filename,
        y_csv_path=y_train_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=args.outcome_col_name,
        y_label_type='per_sequence'
    )

    test_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_test_csv_filename,
        y_csv_path=y_test_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=args.outcome_col_name,
        y_label_type='per_sequence'
    )

    X_train, y_train = train_vitals.get_batch_data(batch_id=0)
    X_test, y_test = test_vitals.get_batch_data(batch_id=0)
    _,T,F = X_train.shape
    
    print('number of time points : %s\nnumber of features : %s\n'%(T,F))
    
    # split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.validation_size, random_state=42)
    
    # set class weights as 1/(number of samples in class) for each class to handle class imbalance
#     class_weights = torch.tensor([1/(y_train==0).sum(),
#                                   1/(y_train==1).sum()]).double()
    
    # scale features
#     X_train = abs_scaler_3d(X_train)
#     X_test = abs_scaler_3d(X_test)
    
    # standardize data for PC-HMM
    key_list = ['train', 'valid', 'test']
    data_dict = dict.fromkeys(keys_list)
    data_dict['train'] = (X_train, y_train)
    data_dict['valid'] = (X_valid, y_valid)
    data_dict['test'] = (X_test, y_test)
    
    ## might need to expand X_train dims
    ## make y categorical
    ## check toy.py half moons class for hint
    
    from IPython import embed; embed()
    
    

def standard_scaler_3d(X):
    # input : X (N, T, F)
    # ouput : scaled_X (N, T, F)
    N, T, F = X.shape
    if T==1:
        scalers = {}
        for i in range(X.shape[1]):
            scalers[i] = StandardScaler()
            X[:, i, :] = scalers[i].fit_transform(X[:, i, :])
    else:
        # zscore across subjects and time points for each feature
        for i in range(F):
            mean_across_NT = X[:,:,i].mean()
            std_across_NT = X[:,:,i].std()            
            if std_across_NT<0.0001: # handling precision
                std_across_NT = 0.0001
            X[:,:,i] = (X[:,:,i]-mean_across_NT)/std_across_NT
    return X

def abs_scaler_3d(X):
    # input : X (N, T, F)
    # ouput : scaled_X (N, T, F)
    N, T, F = X.shape
    # zscore across subjects and time points for each feature
    for i in range(F):
        max_across_NT = X[:,:,i].max()
        min_across_NT = X[:,:,i].min()  
        den = max_across_NT - min_across_NT
        X[:,:,i] = (X[:,:,i]-min_across_NT)/den
    return X


# def standardize_data_for_pchmm(X_train, y_train, X_test, y_test):
    



if __name__ == '__main__':
    main()
    