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
import seaborn as sns
from itertools import combinations


def plotGauss2DContour(
        mu, Sigma,
        color='b',
        radiusLengths=[1.0, 3.0],
        markersize=3.0,
        ax_handle=None,
        label=''
        ):
    ''' Plot elliptical contours for provided mean mu, covariance Sigma.
    Uses only the first 2 dimensions.
    Post Condition
    --------------
    Plot created on current axes
    '''
    
    # Decompose cov matrix into eigenvalues "lambda[d]" and eigenvectors "U[:,d]"
    lambda_D, U_DD = np.linalg.eig(Sigma)
    
    # Verify orthonormal
    D = len(mu)
    assert np.allclose(np.eye(D), np.dot(U_DD, U_DD.T))
    # View eigenvector matrix as a rotation transformation
    rot_DD = U_DD

    # Prep for plotting elliptical contours
    # by creating grid of G different (x,y) points along perfect circle
    # Recall that a perfect circle is swept by considering all radians between [-pi, +pi]
    unit_circle_radian_step_size=0.03
    t_G = np.arange(-np.pi, np.pi, unit_circle_radian_step_size)
    x_G = np.sin(t_G)
    y_G = np.cos(t_G)
    Zcirc_DG = np.vstack([x_G, y_G])

    # Warp circle into ellipse defined by Sigma's eigenvectors
    # Rescale according to eigenvalues
    Zellipse_DG = np.sqrt(lambda_D)[:,np.newaxis] * Zcirc_DG
    # Rotate according to eigenvectors
    Zrotellipse_DG = np.dot(rot_DD, Zellipse_DG)

    radius_lengths=[0.3, 0.6, 0.9, 1.2, 1.5]

    # Plot contour lines across several radius lengths
    for r in radius_lengths:
        Z_DG = r * Zrotellipse_DG + mu[:, np.newaxis]
        ax_handle.plot(
            Z_DG[0], Z_DG[1], '.-',
            color=color,
            markersize=3.0,
            markerfacecolor=color,
            markeredgecolor=color, 
            label=label)
        
    return ax_handle


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

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    return ax
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

    
    ## get the test data
    x_train_np_filename = os.path.join(args.clf_train_test_split_dir, 'X_train.npy')
    x_valid_np_filename = os.path.join(args.clf_train_test_split_dir, 'X_valid.npy')
    x_test_np_filename = os.path.join(args.clf_train_test_split_dir, 'X_test.npy')
    y_train_np_filename = os.path.join(args.clf_train_test_split_dir, 'y_train.npy')
    y_valid_np_filename = os.path.join(args.clf_train_test_split_dir, 'y_valid.npy')
    y_test_np_filename = os.path.join(args.clf_train_test_split_dir, 'y_test.npy')    
    
    
    X_train = np.load(x_train_np_filename)
    X_test = np.load(x_test_np_filename)
    y_train = np.load(y_train_np_filename)
    y_test = np.load(y_test_np_filename)
    X_val = np.load(x_valid_np_filename)
    y_val = np.load(y_valid_np_filename)
    
    
    features_dict = load_data_dict_json(os.path.join(args.preproc_data_dir, 'Spec_FeaturesPerTimestep.json')) 
    feature_cols = parse_feature_cols(features_dict)
    
    N,T,F = X_test.shape
    print('number of data points : %d\nnumber of time points : %s\nnumber of features : %s\n'%(N,T,F))
    
    
    # mask labels in training set and validation set as per user provided %perc_labelled
#     N_tr = len(X_train)
    N_va = len(X_val)
    N_te = len(X_test)
    
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)
    
    # standardize data for PC-HMM
    key_list = ['train', 'valid', 'test']
    data_dict = dict.fromkeys(key_list)
    data_dict['train'] = (X_train, y_train)
    data_dict['valid'] = (X_val, y_val)
    data_dict['test'] = (X_test, y_test)
    
    clf_models_dir=args.clf_models_dir

    # predict on each tslice
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    perf_df = pd.DataFrame()


    # load the test data into dataset object for evaluation
    x_test = np.expand_dims(X_test, 1)
    key_list = ['train', 'valid', 'test']
    data_dict = dict.fromkeys(key_list)
    data_dict['train'] = (X_test[:2], y_test[:2])
    data_dict['valid'] = (X_test[:2], y_test[:2])
    data_dict['test'] = (X_test, y_test)
    data = custom_dataset(data_dict=data_dict)


    for perc_labelled in ['1.2', '3.7', '11.1', '33.3', '100']:#'3.7', '11.1', '33.3', '100'
        f, axs = plt.subplots(1, 1, figsize=(12, 8))
        sns.set_style("whitegrid") # or use "white" if we don't want grid lines
        sns.set_context("notebook", font_scale=1.4)
        for states in ['10', '30', '60', '90']:#'10', '30', '60', '90'

            saved_model_files_aka = os.path.join(clf_models_dir, 
                                                 "final_perf_*semi-supervised-pchmm*perc_labelled=%s-*n_states=%s-*lamb=*.csv"%(perc_labelled, states))
            best_model_file, n_states = get_best_model_file(saved_model_files_aka)
            best_model_weights = best_model_file.replace('.csv', '-weights.h5').replace('final_perf_', '')

            # load classifier
            model = HMM(states=n_states,
                        observation_dist='NormalWithMissing',
                        predictor_dist='Categorical')
            
            model.build(data)
            model.model.load_weights(best_model_weights)
            
            # see if you can visualize the means of the top predictor weight states
            mu_all = model.hmm_model(x_test[:10]).observation_distribution.distribution.mean().numpy()
            cov_all = model.hmm_model(x_test[:10]).observation_distribution.distribution.scale.numpy()
            '''
            if (perc_labelled=='100')&(states=='90'):
                sorted_inds = np.argsort(model._predictor.get_weights()[0][:, 1])[::-1]
#                 keep_features_cols = ['age', 'heart rate', 
#                                       'diastolic blood pressure', 'systolic blood pressure', 
#                                       'blood urea nitrogen', 'oxygen saturation', 
#                                       'white blood cell count', 'red blood cell count']

                keep_features_cols = ['age', 
                                      'blood urea nitrogen',
                                      'oxygen saturation']

                keep_features_inds = np.where(np.isin(feature_cols, keep_features_cols))
                n_influential = 3 # number of influential states to retain from predictor
                mu_KD_influential = np.squeeze(mu_all[sorted_inds[:n_influential], :][:, keep_features_inds])
                cov_KD_influential = np.squeeze(cov_all[sorted_inds[:n_influential], :][:, keep_features_inds])
                feature_cols_reindexed = np.array(feature_cols)[keep_features_inds[0]]
                
                cohort_colors = ['r', 'g', 'b', 'k']
                for combo in combinations(keep_features_cols, 2):
                    f, axs = plt.subplots(1, 1, figsize=(8, 8))
                    sns.set_style("white") # or use "white" if we don't want grid lines
                    sns.set_context("notebook", font_scale=1.3)
                    curr_feature_combo_inds = np.isin(keep_features_cols, combo)
                    curr_feature_combo = np.array(keep_features_cols)[curr_feature_combo_inds]
                    for kk in range(n_influential):
                        mu = mu_KD_influential[kk, curr_feature_combo_inds]
                        Sigma = np.diag(cov_KD_influential[kk, curr_feature_combo_inds])
                        axs = plotGauss2DContour(mu, Sigma, ax_handle=axs, 
                                                 label='Cohort %s'%kk, 
                                                 color=cohort_colors[kk])
                    
                    axs.legend()
                    axs = legend_without_duplicate_labels(axs)
                    axs.set_xlabel(curr_feature_combo[0])
                    axs.set_ylabel(curr_feature_combo[1])
                    f.savefig('interpretability_%s_%s.pdf'%(curr_feature_combo[0], curr_feature_combo[1]), 
                              bbox_inches='tight', 
                              pad_inches=0)
                    
#                 cohorts_from_pchmm_df = pd.DataFrame(mu_KD_influential, columns=feature_cols_reindexed)
#                 cohorts_from_pchmm_df['cohort'] = range(n_influential)
#                 sns.set_style("white") # or use "white" if we don't want grid lines
#                 sns.set_context("notebook", font_scale=1.3)
#                 sns.pairplot(cohorts_from_pchmm_df, hue='cohort', palette='bright', diag_kind=None, plot_kws={"s": 80})
#                 plt.savefig('interpretability.png')
            
            '''
            
            x_test, y_test = data.test().numpy()
            # get the beliefs of the test set
            z_test = model.hmm_model.predict(x_test)
            y_test_pred_proba = model._predictor.predict(z_test)
            

            precisions, recalls, thresholds_pr = precision_recall_curve(y_test[:, 1], y_test_pred_proba[:, 1])
            axs.plot(recalls, precisions, label='PCHMM-n_states=%s (AUPRC = %.2f)'%(states, 
                                                                                    average_precision_score(y_test[:, 1],
                                                                                                    y_test_pred_proba[:, 1])))
            
            
            print('Evaluating PCHMM with perc_labelled =%s and states =%s with model %s'%(perc_labelled, states, best_model_file))

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
                avg_precision_np[k] = average_precision_score(curr_y_test[:, 1], curr_y_pred_proba[:, 1])


            print('perc_labelled = %s, states = %s, \nROC-AUC : %.2f'%(perc_labelled, states, np.percentile(roc_auc_np, 50)))
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
#                     row_dict['imputation_strategy'] = imputation_strategy

                perf_df = perf_df.append(row_dict, ignore_index=True) 
        axs.set_xlabel('recall')
        axs.set_ylabel('precision')
        axs.legend()
        axs.set_title('Percent labeled sequences = %s'%perc_labelled)
        f.savefig(os.path.join('pr_curves', 'PR_curve_perc_labeled=%s.pdf'%perc_labelled),
                  bbox_inches='tight',
                  pad_inches=0)
    perf_csv = os.path.join(args.output_dir, 'semi_supervised_pchmm_performance.csv')
    print('Saving semi-supervised PCHMM performance to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)