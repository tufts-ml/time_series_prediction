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
    n_states = 12
    spacing = 5
    optimizer = get_optimizer('adam', lr = args.lr)
    init_means = np.zeros((n_states, F))
    init_means[:, 0] = np.arange(0, n_states*spacing, spacing)
    init_means[0, :] = [0, 1]
    init_means[10, :] = [0, -1]
    init_means[3, :] = [15, 1]
    init_means[11, :] = [15, -1]
#     init_means[4, :] = [7.5, 0]
#     init_means[11:, 0] = 50
#     init_covs = np.stack([np.zeros((F,F)) for i in range(n_states)])
    init_covs = np.stack([np.zeros(F) for i in range(n_states)])
    init_covs[0,:] = [-0.45, -0.75]
    init_covs[10,:] = [-0.45, -0.75]
    init_covs[3,:] = [-0.45, -0.75]
    init_covs[11,:] = [-0.45, -0.75]
    
    init_state_probas = 0.124*np.ones(n_states)
    init_state_probas[0] = 0.006
    init_state_probas[3] = 0.006
    init_state_probas[10] = 0.006
    init_state_probas[11] = 0.006

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
    
    
    # set the regression coefficients of the model
    eta_weights = np.zeros((n_states, 2))
    eta_weights[1:11, :] = [1, -1]
    eta_weights[0, :] = [-100, 100]
    eta_weights[-1, :] = [-100, 100]
    pos_ratio = y_train.sum()/len(y_train)
    eta_bias = np.array([1-pos_ratio, pos_ratio])
    eta_all = [eta_weights, eta_bias]
    
    model._predictor.set_weights(eta_all)
    model.fit(data, steps_per_epoch=50, epochs=200, batch_size=args.batch_size)
    
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
    
#     from IPython import embed; embed()
    # save the loss plots of the model
    save_file = os.path.join(args.output_dir, args.output_filename_prefix+'.csv')
    save_loss_plots(model, save_file, data_dict)
    


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
    
    '''
    Debugging code
    
    # get 10 positive and 10 negative sequences
    pos_inds = np.where(y[:,1]==1)[0]
    neg_inds = np.where(y[:,0]==1)[0]
    
    n=10
    x_pos = x[pos_inds[:n]]
    y_pos = y[pos_inds[:n]]
    x_neg = x[neg_inds[:n]]
    y_neg = y[neg_inds[:n]]
    
    # get their belief states of the positive and negative sequences
    z_pos = model.hmm_model.predict(x_pos)
    z_neg = model.hmm_model.predict(x_neg)
    
    # print the average belief states across time
    z_pos.mean(axis=1)
    z_neg.mean(axis=1)
    
    # print the predicted probabilities of class 0 and class 1
    y_pred_pos = model._predictor(z_pos).distribution.logits.numpy()
    y_pred_neg = model._predictor(z_neg).distribution.logits.numpy()
    
    
    ''' 
    