import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
import torch
import skorch
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                             balanced_accuracy_score, confusion_matrix, 
                             roc_auc_score, make_scorer)
from yattag import Doc
import matplotlib.pyplot as plt

from dataset_loader import TidySequentialDataCSVLoader
from RNNBinaryClassifier import RNNBinaryClassifier
from feature_transformation import (parse_id_cols, parse_feature_cols)
from utils import load_data_dict_json
from joblib import dump

import warnings
warnings.filterwarnings("ignore")

from skorch.callbacks import (Callback, LoadInitState, 
                              TrainEndCheckpoint, Checkpoint, 
                              EpochScoring, EarlyStopping, LRScheduler, GradientNormClipping, TrainEndCheckpoint)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch.utils import noop
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
    parser.add_argument('--hidden_units', type=int, default=32,
                        help='Number of hidden units')
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help='Number of hidden layers')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--perc_labelled', type=int, default=100,
                        help='percentage of labelled examples')
    parser.add_argument('--validation_size', type=float, default=0.15,
                        help='validation split size')
    parser.add_argument('--is_data_simulated', type=bool, default=False,
                        help='boolean to check if data is simulated or from mimic')
    parser.add_argument('--simulated_data_dir', type=str, default='simulated_data/2-state/',
                        help='dir in which to simulated data is saved.Must be provide if is_data_simulated = True')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='directory where trained model and loss curves over epochs are saved')
    parser.add_argument('--output_filename_prefix', type=str, default=None, 
                        help='prefix for the training history jsons and trained classifier')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cpu'

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
    N,T,F = X_train.shape
    
    # keep only as many labelled examples as specified in %perc_labelled args
    state_id = 41
    rnd_state = np.random.RandomState(state_id)
    n_unlabelled = int((1-(args.perc_labelled)/100)*N)
    unlabelled_inds = rnd_state.permutation(N)[:n_unlabelled]
    X_train = np.delete(X_train, unlabelled_inds, axis=0)
    y_train = np.delete(y_train, unlabelled_inds, axis=0)    
    
    print('number of time points : %s\nnumber of features : %s\n'%(T,F))
    
    # set class weights as 1/(number of samples in class) for each class to handle class imbalance
    class_weights = torch.tensor([1/(y_train==0).sum(),
                                  1/(y_train==1).sum()]).double()
    
    print('Number of training sequences : %s'%len(y_train))
    print('Number of test sequences : %s'%X_test.shape[0])
    print('Ratio positive in train : %.2f'%((y_train==1).sum()/len(y_train)))
    print('Ratio positive in test : %.2f'%((y_test==1).sum()/len(y_test)))
    
    # callback to compute gradient norm
    compute_grad_norm = ComputeGradientNorm(norm_type=2)

    # LSTM
    if args.output_filename_prefix==None:
        output_filename_prefix = ('hiddens=%s-layers=%s-lr=%s-dropout=%s-weight_decay=%s'%(args.hidden_units, 
                                                                       args.hidden_layers, 
                                                                       args.lr, 
                                                                       args.dropout, args.weight_decay))
    else:
        output_filename_prefix = args.output_filename_prefix
        
        
    print('RNN parameters : '+ output_filename_prefix)
    
    rnn = RNNBinaryClassifier(
              max_epochs=50,
              batch_size=args.batch_size,
              device=device,
              lr=args.lr,
              callbacks=[
              EpochScoring('roc_auc', lower_is_better=False, on_train=True, name='aucroc_score_train'),
              EpochScoring('roc_auc', lower_is_better=False, on_train=False, name='aucroc_score_valid'),
#               EarlyStopping(monitor='aucroc_score_valid', patience=5, threshold=0.002, threshold_mode='rel',
#                                              lower_is_better=False),
#               LRScheduler(policy=ReduceLROnPlateau, mode='max', monitor='aucroc_score_valid', patience=10),
                  compute_grad_norm,
#               GradientNormClipping(gradient_clip_value=0.5, gradient_clip_norm_type=2),
              Checkpoint(monitor='aucroc_score_valid', f_history=os.path.join(args.output_dir, output_filename_prefix+'.json')),
              TrainEndCheckpoint(dirname=args.output_dir, fn_prefix=output_filename_prefix),
              ],
              criterion=torch.nn.CrossEntropyLoss,
              criterion__weight=class_weights,
              train_split=skorch.dataset.CVSplit(args.validation_size),
              module__rnn_type='GRU',
              module__n_layers=args.hidden_layers,
              module__n_hiddens=args.hidden_units,
              module__n_inputs=X_train.shape[-1],
              module__dropout_proba=args.dropout,
              optimizer=torch.optim.Adam,
              optimizer__weight_decay=args.weight_decay
                         ) 
    '''
    # fit on subset of data
    print('fitting on subset of data...')
    n = 500
    rnd_inds = np.random.permutation(N)[:n]
    X = X_train[rnd_inds]
    y = y_train[rnd_inds]
    clf = rnn.fit(X, y)
    '''
    
    clf = rnn.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_train)
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    auroc_train_final = roc_auc_score(y_train, y_pred_proba_pos)
    print('AUROC with LSTM (Train) : %.2f'%auroc_train_final)

    y_pred_proba = clf.predict_proba(X_test)
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    auroc_test_final = roc_auc_score(y_test, y_pred_proba_pos)
    print('AUROC with LSTM (Test) : %.2f'%auroc_test_final)
    
    # save the performance on test set
    test_perf_df = pd.DataFrame(columns={'test_auc'})
    test_perf_df = test_perf_df.append({'test_auc': auroc_test_final}, ignore_index=True)
    test_perf_csv = os.path.join(args.output_dir, output_filename_prefix+'.csv')
    test_perf_df.to_csv(test_perf_csv, index=False)
    

def convert_proba_to_binary(probabilites):
    return [0 if probs[0] > probs[1] else 1 for probs in probabilites]

def get_paramater_gradient_l2_norm(net,**kwargs):
    parameters = [i for  _,i in net.module_.named_parameters()]
    total_norm = 0
    for p in parameters:
        if p.requires_grad==True:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def get_paramater_gradient_inf_norm(net, **kwargs):
    parameters = [i for  _,i in net.module_.named_parameters()]
    total_norm = max(p.grad.data.abs().max() for p in parameters if p.grad==True)
    return total_norm


class ComputeGradientNorm(Callback):
    def __init__(self, norm_type=1, f_history=None):
        self.norm_type = norm_type
        self.batch_num = 1
        self.epoch_num = 1
        self.f_history = f_history

    def on_epoch_begin(self, net,  dataset_train=None, dataset_valid=None, **kwargs):
        self.batch_num = 1

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        self.epoch_num += 1

    def on_grad_computed(self, net, named_parameters, **kwargs):
        if self.norm_type == 1:
            gradient_norm = get_paramater_gradient_inf_norm(net)
            print('epoch: %d, batch: %d, gradient_norm: %.3f'%(self.epoch_num, self.batch_num, gradient_norm))
            if self.f_history is not None:
                self.write_to_file(gradient_norm)
            self.batch_num += 1
        else:
            gradient_norm = get_paramater_gradient_l2_norm(net)
            print('epoch: %d, batch: %d, gradient_norm: %.3f'%(self.epoch_num, self.batch_num, gradient_norm))
            if self.f_history is not None:
                self.write_to_file(gradient_norm)
            self.batch_num += 1

            
def get_sequence_lengths(X_NTF, pad_val):
    N = X_NTF.shape[0]
    seq_lens_N = np.zeros(N, dtype=np.int64)
    for n in range(N):
        bmask_T = np.all(X_NTF[n] == 0, axis=-1)
        seq_lens_N[n] = np.searchsorted(bmask_T, 1)
    return seq_lens_N
    

def get_per_feature_estimates_from_padded_sequences(X_NTF, seq_lens_N):
    '''
    normalizes the data using minmax scaling 
    
    input : X_NTF N examples, T timepoints, F features 
            seq_lens_N : lengths of sequences per example
    
    output : 1xF vector of means, mins and maxs per feature
    '''   
    
    N, T, F = X_NTF.shape
    per_feature_means = np.zeros(F)
    per_feature_min = np.zeros(F)
    per_feature_max = np.zeros(F)
    
    for f in range(F):
        per_feature_sum = 0
        for n in range(N):
            # add up the features at observed time points across all examples
            per_feature_sum += np.sum(X_NTF[n,:seq_lens_N[n],f])
            
            if n==0:
                curr_min = np.min(X_NTF[n, :seq_lens_N[n], f])
                curr_max = np.max(X_NTF[n, :seq_lens_N[n], f])
            else:
                curr_min = min(np.min(X_NTF[n, :seq_lens_N[n], f]), curr_min)
                curr_max = max(np.max(X_NTF[n, :seq_lens_N[n], f]), curr_max)
        
        # divide by total time points over all sequences
        per_feature_means[f] = per_feature_sum/(np.sum(seq_lens_N))
        per_feature_min[f] = curr_min
        per_feature_max[f] = curr_max
    
    return per_feature_means, per_feature_min, per_feature_max

def normalize_data(X_NTF, seq_lens_N, per_feature_mins, per_feature_maxs):
    '''
    normalizes the data using minmax scaling 
    
    input : X_NTF N examples, T timepoints, F features 
            per_feature_mins : 1xF mins per feature
            per_feature_maxs : 1xF maxs per feature
    
    output : X_NTF normalized over only the non padded segments in each sequence
    '''
    
    N, T, F = X_NTF.shape
    
    # Get the min and max per feature in the non-padded segments of the data 
    for f in range(F):
        den = per_feature_maxs[f] - per_feature_mins[f]
        if den>0:
            for n in range(N):
                X_NTF[n, :seq_lens_N[n], f] = (X_NTF[n, :seq_lens_N[n], f] - per_feature_mins[f])/den

    return X_NTF

if __name__ == '__main__':
    main()
    
