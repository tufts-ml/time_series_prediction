import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit, StratifiedShuffleSplit
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
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy.ma as ma
from sklearn.cluster import KMeans

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
    parser.add_argument('--lamb', type=int, default=100,
                        help='Langrange multiplier')
    parser.add_argument('--validation_size', type=float, default=0.15,
                        help='validation split size')
    parser.add_argument('--n_states', type=int, default=0.15,
                        help='number of HMM states')
    parser.add_argument('--perc_labelled', type=int, default=0.15,
                        help='percentage of labelled examples')
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
    
    del train_vitals, test_vitals
    N,T,F = X_train.shape
    
    print('number of data points : %d\nnumber of time points : %s\nnumber of features : %s\n'%(N,T,F))
      
    # split train into train and validation with label balancing
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.validation_size, 
#                                                       random_state=213, shuffle=False)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.validation_size, random_state=213)
    for train_idx, val_idx in sss.split(X_train, y_train):
        X_train, X_val = X_train[train_idx], X_train[val_idx]
        y_train, y_val = y_train[train_idx], y_train[val_idx]
    
    
    # mask labels in training set and validation set as per user provided %perc_labelled
    N_tr = len(X_train)
    N_va = len(X_val)
    N_te = len(X_test)
    
    state_id = 41
    rnd_state = np.random.RandomState(state_id)
    n_unlabelled_tr = int((1-(args.perc_labelled)/100)*N_tr)
    unlabelled_inds_tr = rnd_state.permutation(N_tr)[:n_unlabelled_tr]
    y_train = y_train.astype(np.float32)
    y_train[unlabelled_inds_tr] = np.nan  
    
    rnd_state = np.random.RandomState(state_id)
    n_unlabelled_va = int((1-(args.perc_labelled)/100)*N_va)
    unlabelled_inds_va = rnd_state.permutation(N_va)[:n_unlabelled_va]
    y_val = y_val.astype(np.float32)
    y_val[unlabelled_inds_va] = np.nan 
    
    rnd_state = np.random.RandomState(state_id)
    n_unlabelled_te = int((1-(args.perc_labelled)/100)*N_te)
    unlabelled_inds_te = rnd_state.permutation(N_te)[:n_unlabelled_te]
    y_test = y_test.astype(np.float32)
    y_test[unlabelled_inds_te] = np.nan 
    
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)
    
    # standardize data for PC-HMM
    key_list = ['train', 'valid', 'test']
    data_dict = dict.fromkeys(key_list)
    data_dict['train'] = (X_train, y_train)
    data_dict['valid'] = (X_val, y_val)
    data_dict['test'] = (X_test, y_test)
    
    # get the init means and covariances of the observation distribution by clustering
    print('Initializing cluster means with kmeans...')
    prng = np.random.RandomState(args.seed)
    n_init = 15000
    init_inds = prng.permutation(N)[:n_init]# select random samples from training set
    X_flat = X_train[:n_init].reshape(n_init*T, X_train.shape[-1])# flatten across time
    X_flat = np.where(np.isnan(X_flat), ma.array(X_flat, mask=np.isnan(X_flat)).mean(axis=0), X_flat)
        
    # get cluster means
    n_states = args.n_states
    kmeans = KMeans(n_clusters=n_states, n_init=10).fit(X_flat)
    init_means = kmeans.cluster_centers_
    
    # get cluster covariance in each cluster
    init_covs = np.stack([np.zeros(F) for i in range(n_states)])
#     for i in range(n_states):
#         init_covs[i,:]=np.var(X_flat[X_flat_preds==i], axis=0)
    
    # train model
    optimizer = get_optimizer('adam', lr = args.lr)

    # draw the initial probabilities from dirichlet distribution
    prng = np.random.RandomState(args.seed)
    init_state_probas = prng.dirichlet(5*np.ones(n_states))

    model = HMM(
            states=n_states,                                     
            lam=args.lamb,                                      
            prior_weight=0.01,
            observation_dist='NormalWithMissing',
            observation_initializer=dict(loc=init_means, 
                scale=init_covs),
            initial_state_alpha=1.,
            initial_state_initializer=np.log(init_state_probas),
            transition_alpha=1.,
            transition_initializer=None, optimizer=optimizer)
    
    
    data = custom_dataset(data_dict=data_dict)
    data.batch_size = args.batch_size
    model.build(data) 
    
    # set the regression coefficients of the model
    eta_weights = np.zeros((n_states, 2))  
        
    # set the initial etas as the weights from logistic regression classifier with average beliefs as features 
    x_train,y_train = data.train().numpy()
    
    pos_inds = np.where(y_train[:,1]==1)[0]
    neg_inds = np.where(y_train[:,0]==1)[0]
    
    x_pos = x_train[pos_inds]
    y_pos = y_train[pos_inds]
    x_neg = x_train[neg_inds]
    y_neg = y_train[neg_inds]
    
    # get their belief states of the positive and negative sequences
    z_pos = model.hmm_model.predict(x_pos)
    z_neg = model.hmm_model.predict(x_neg)
    
    # print the average belief states across time
    beliefs_pos = z_pos.mean(axis=1)
    beliefs_neg = z_neg.mean(axis=1)
    
    
    # perform logistic regression with belief states as features for these positive and negative samples
    print('Performing Logistic Regression on initial belief states to get the initial eta coefficients...')
    logistic = LogisticRegression(solver='lbfgs', random_state = 42)
    penalty = ['l2']
    C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3, 1e4, 1e5]
    hyperparameters = dict(C=C, penalty=penalty)
    classifier = GridSearchCV(logistic, hyperparameters, cv=5, verbose=10, scoring = 'roc_auc')
    
    X_tr = np.vstack([beliefs_pos, beliefs_neg])
    y_tr = np.vstack([y_pos, y_neg])
    cv = classifier.fit(X_tr, y_tr[:,1])
    
    # get the logistic regression coefficients. These are the optimal eta coefficients.
    lr_weights = cv.best_estimator_.coef_
    
    # set the K logistic regression weights as K x 2 eta coefficients
    opt_eta_weights = np.vstack([np.zeros_like(lr_weights), lr_weights]).T
    
    # set the intercept of the eta coefficients
    opt_eta_intercept = np.asarray([0, cv.best_estimator_.intercept_[0]])
    init_etas = [opt_eta_weights, opt_eta_intercept]
    model._predictor.set_weights(init_etas)
    
    
    model.fit(data, steps_per_epoch=10, epochs=50, batch_size=args.batch_size, lr=args.lr,
              initial_weights=model.model.get_weights())
    
    # evaluate on test set
#     x_train, y_train = data.train().numpy()
    z_train = model.hmm_model.predict(x_train)
    y_train_pred_proba = model._predictor.predict(z_train)
    labelled_inds_tr = ~np.isnan(y_train[:,0])
    train_roc_auc = roc_auc_score(y_train[labelled_inds_tr], y_train_pred_proba[labelled_inds_tr])
    print('ROC AUC on train : %.2f'%train_roc_auc)
    
    
    # evaluate on test set
    x_test, y_test = data.test().numpy()
    z_test = model.hmm_model.predict(x_test)
    y_test_pred_proba = model._predictor.predict(z_test)
    labelled_inds_te = ~np.isnan(y_test[:,0])
    test_roc_auc = roc_auc_score(y_test[labelled_inds_te], y_test_pred_proba[labelled_inds_te])
    print('ROC AUC on test : %.2f'%test_roc_auc)
    
    # get the parameters of the fit distribution
    mu_all = model.hmm_model(x_train).observation_distribution.distribution.mean().numpy()
    try:
        cov_all = model.hmm_model(x_train).observation_distribution.distribution.covariance().numpy()
    except:
        cov_all = model.hmm_model(x_train).observation_distribution.distribution.scale.numpy()
        
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
    
    # save the model
    model_filename = os.path.join(args.output_dir, args.output_filename_prefix+'-weights.h5')
    model.model.save_weights(model_filename, save_format='h5')
    
    
    
# if __name__ == '__main__':
#     main()
    
    '''
    Debugging code
    
    from sklearn.linear_model import LogisticRegression
    # get 10 positive and 10 negative sequences
    pos_inds = np.where(y[:,1]==1)[0]
    neg_inds = np.where(y[:,0]==1)[0]
    
    x_pos = x[pos_inds]
    y_pos = y[pos_inds]
    x_neg = x[neg_inds]
    y_neg = y[neg_inds]
    
    # get their belief states of the positive and negative sequences
    z_pos = model.hmm_model.predict(x_pos)
    z_neg = model.hmm_model.predict(x_neg)
    
    # print the average belief states across time
    beliefs_pos = z_pos.mean(axis=1)
    beliefs_neg = z_neg.mean(axis=1)
    
    # perform logistic regression with belief states as features for these positive and negative samples
    
    logistic = LogisticRegression(solver='lbfgs', random_state = 42)
    
    
    penalty = ['l2']
    C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3, 1e4]
    hyperparameters = dict(C=C, penalty=penalty)
    classifier = GridSearchCV(logistic, hyperparameters, cv=5, verbose=10, scoring = 'roc_auc')
    
    X_tr = np.vstack([beliefs_pos, beliefs_neg])
    y_tr = np.vstack([y_pos, y_neg])
    cv = classifier.fit(X_tr, y_tr[:,1])
    
    # get the logistic regression coefficients. These are the optimal eta coefficients.
    lr_weights = cv.best_estimator_.coef_
    
    # plot the log probas from of a few negative and positive samples using the logistic regression estimates 
    n=50
    x_pos = x[pos_inds[:n]]
    y_pos = y[pos_inds[:n]]
    x_neg = x[neg_inds[:n]]
    y_neg = y[neg_inds[:n]]

    # get their belief states of the positive and negative sequences
    z_pos = model.hmm_model.predict(x_pos)
    z_neg = model.hmm_model.predict(x_neg)

    # print the average belief states across time
    beliefs_pos = z_pos.mean(axis=1)
    beliefs_neg = z_neg.mean(axis=1)
    
    X_eval = np.vstack([beliefs_pos, beliefs_neg])
    predict_probas = cv.predict_log_proba(X_eval)
    predict_probas_pos = predict_probas[:n, 1]
    predict_probas_neg = predict_probas[n:, 1]
    
    f, axs = plt.subplots(1,1, figsize=(5, 5))
    axs.plot(np.zeros_like(predict_probas_neg), predict_probas_neg, '.')
    axs.plot(np.ones_like(predict_probas_pos), predict_probas_pos, '.')
    axs.set_ylabel('log p(y=1)')
    axs.set_xticks([0, 1])
    axs.set_xticklabels(['50 negative samples', '50 positive samples'])
    f.savefig('predicted_proba_viz_post_lr.png')
    
    # plot some negative sequences that have a high log p(y=1)
    outcast_inds = np.where(predict_probas_neg>-3)
    x_neg_outcasts = x_neg(outcast_inds)
    f, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(x_neg_outcasts[:, :, :, 0], x_neg_outcasts[:, :, :, 1], s=2, marker='x')
    ax.set_xlim([-5, 50])
    ax.set_ylim([-5, 5])
    fontsize=10
    ax.set_ylabel('Temperature_1 (deg C)', fontsize=fontsize)
    ax.set_xlabel('Temperature_0 (deg C)', fontsize = fontsize)
    ax.set_title('negative sequences with log p(y=1)>-3')
    f.savefig('x_outcasts.png')
    
    ###### Converting logistic regression weights to etas
    # set the lr weights as initial etas
    opt_eta_weights = np.vstack([np.zeros_like(lr_weights), lr_weights]).T
    init_etas[0] = opt_eta_weights
    model._predictor.set_weights(init_etas)
    
    # print the predicted probabilities of class 0 and class 1
    y_pred_pos = model._predictor(z_pos).distribution.logits.numpy()
    y_pred_neg = model._predictor(z_neg).distribution.logits.numpy()
    
    # the 2 lines above is the same as 
    y_pred_pos = np.matmul(beliefs_pos, opt_eta_weights)+opt_eta_intercept
    y_pred_neg = np.matmul(beliefs_neg, opt_eta_weights)+opt_eta_intercept
    
    
    # plot the log p(y=1) for the negative and positive samples
    y1_pos = expit(y_pred_pos[:,1])
    y1_neg = expit(y_pred_neg[:,1])
    f, axs = plt.subplots(1,1, figsize=(5, 5))
    axs.plot(np.zeros_like(y1_neg), y1_neg, '.')
    axs.plot(np.ones_like(y1_pos), y1_pos, '.')
    axs.set_ylabel('p(y=1)')
    axs.set_xticks([0, 1])
    axs.set_xticklabels(['50 negative samples', '50 positive samples'])
    f.savefig('predicted_proba_viz.png')
    ''' 
    

