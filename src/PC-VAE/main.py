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
# DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
# PROJECT_REPO_DIR = os.path.abspath(
#     os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))
PROJECT_REPO_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
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
from pcvae.datasets.base import dataset, real_dataset, classification_dataset, make_dataset
from sklearn.model_selection import train_test_split
from pcvae.util.optimizers import get_optimizer

def main():
    parser = argparse.ArgumentParser(description='pchmm fitting')
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
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.validation_size, random_state=213)
    
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)
    
    # standardize data for PC-HMM
    key_list = ['train', 'valid', 'test']
    data_dict = dict.fromkeys(key_list)
    data_dict['train'] = (X_train, y_train)
    data_dict['valid'] = (X_val, y_val)
    data_dict['test'] = (X_test, y_test)
    
    # train model
    n_states = 10
    spacing = 5
    optimizer = get_optimizer('adam', lr = args.lr)
    init_means = np.zeros((n_states, F))
    init_means[:, 0] = np.arange(0, n_states*spacing, spacing)
    init_means[0, :] = [0, 0.5]
    init_means[1, :] = [0, -0.5]
    init_means[2, :] = [15, 0.5]
    init_means[3, :] = [15, -0.5]
    init_means[4, :] = [7.5, 0]
#     init_covs = np.stack([np.zeros((F,F)) for i in range(n_states)])
    init_covs = np.stack([np.zeros(F) for i in range(n_states)])
    init_covs[0,:] = [-0.6, -0.9]
    init_covs[1,:] = [-0.6, -0.9]
    init_covs[2,:] = [-0.6, -0.9]
    init_covs[3,:] = [-0.6, -0.9]
    
    init_state_probas = 0.124*np.ones(n_states)
    init_state_probas[0] = 0.002
    init_state_probas[3] = 0.002

    model = HMM(
            states=n_states,                                     
            lam=10000,                                      
            prior_weight=0.01,
            observation_dist='MultivariateNormalDiag',
            observation_initializer=dict(loc=init_means, 
                scale_diag=init_covs),
            initial_state_alpha=1.,
            initial_state_initializer=np.log(init_state_probas),
            transition_alpha=1.,
            transition_initializer=None, optimizer=optimizer)
    
    data = custom_dataset(data_dict=data_dict)
    model.build(data) 
    
#     from IPython import embed; embed()
    model.fit(data, steps_per_epoch=50, epochs=100, batch_size=args.batch_size)
    
    # get the parameters of the fit distribution
    x,y = data.train().numpy()
    mu_all = model.hmm_model(x).observation_distribution.distribution.mean().numpy()
    cov_all = model.hmm_model(x).observation_distribution.distribution.covariance().numpy()
    eta_all = np.vstack(model._predictor.get_weights())
    mu_params_file = os.path.join(args.output_dir, args.output_filename_prefix+'-fit-mu.npy')
    cov_params_file = os.path.join(args.output_dir, args.output_filename_prefix+'-fit-cov.npy')
    eta_params_file = os.path.join(args.output_dir, args.output_filename_prefix+'-fit-eta.npy')
    np.save(mu_params_file, mu_all)
    np.save(cov_params_file, cov_all)
    np.save(eta_params_file, eta_all)
    
    # save the loss plots of the model
    save_file = os.path.join(args.output_dir, args.output_filename_prefix+'.csv')
    save_loss_plots(model, save_file, data_dict)
        
    

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

def abs_scaler_3d(X, min_per_feat=None, max_per_feat=None):
    # input : X (N, T, F)
    # ouput : scaled_X (N, T, F)
    N, T, F = X.shape
    # zscore across subjects and time points for each feature
    if min_per_feat is None:
        min_per_feat = np.zeros(F)
        max_per_feat = np.zeros(F)
        
        for i in range(F):
            min_per_feat[i] = X[:,:,i].min()
            max_per_feat[i] = X[:,:,i].max()
    
    for i in range(F):
        min_across_NT = min_per_feat[i]
        max_across_NT = max_per_feat[i]
        den = max_across_NT - min_across_NT
        X[:,:,i] = (X[:,:,i]-min_across_NT)/den
    return X, min_per_feat, max_per_feat


# def standardize_data_for_pchmm(X_train, y_train, X_test, y_test):
def convert_to_categorical(y):
    C = len(np.unique(y))
    N = len(y)
    y_cat = np.zeros((N, C))
    y_cat[:, 0] = (y==0)*1.0
    y_cat[:, 1] = (y==1)*1.0
    
    return y_cat

def save_loss_plots(model, save_file, data_dict):
    model_hist = model.history.history
    epochs = range(len(model_hist['loss']))
    hmm_model_loss = model_hist['hmm_model_loss']
    predictor_loss = model_hist['predictor_loss']
    model_hist['epochs'] = epochs
    model_hist['n_train'] = len(data_dict['train'][1])
    model_hist['n_valid'] = len(data_dict['valid'][1])
    model_hist['n_test'] = len(data_dict['test'][1])
    model_hist_df = pd.DataFrame(model_hist)
    
    # save to file
    model_hist_df.to_csv(save_file, index=False)
    print('Training plots saved to %s'%save_file)
    
    
if __name__ == '__main__':
    main()
    