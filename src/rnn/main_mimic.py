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
    _,T,F = X_train.shape
    
    print('number of time points : %s\nnumber of features : %s\n'%(T,F))
    
    # set class weights as 1/(number of samples in class) for each class to handle class imbalance
    class_weights = torch.tensor([1/(y_train==0).sum(),
                                  1/(y_train==1).sum()]).double()

    # scale features
#     X_train = standard_scaler_3d(X_train)
#     X_test = standard_scaler_3d(X_test)

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
              max_epochs=30,
              batch_size=args.batch_size,
              device=device,
              lr=args.lr,
              callbacks=[
              EpochScoring('roc_auc', lower_is_better=False, on_train=True, name='aucroc_score_train'),
              EpochScoring('roc_auc', lower_is_better=False, on_train=False, name='aucroc_score_valid'),
              EarlyStopping(monitor='aucroc_score_valid', patience=15, threshold=0.002, threshold_mode='rel',
                                             lower_is_better=False),
              LRScheduler(policy=ReduceLROnPlateau, mode='max', monitor='aucroc_score_valid', patience=10),
#                   compute_grad_norm,
              GradientNormClipping(gradient_clip_value=0.3, gradient_clip_norm_type=2),
              Checkpoint(monitor='aucroc_score_valid', f_history=os.path.join(args.output_dir, output_filename_prefix+'.json')),
              TrainEndCheckpoint(dirname=args.output_dir, fn_prefix=output_filename_prefix),
              ],
              criterion=torch.nn.CrossEntropyLoss,
              criterion__weight=class_weights,
              train_split=skorch.dataset.CVSplit(args.validation_size),
              module__rnn_type='LSTM',
              module__n_layers=args.hidden_layers,
              module__n_hiddens=args.hidden_units,
              module__n_inputs=X_train.shape[-1],
              module__dropout_proba=args.dropout,
              optimizer=torch.optim.Adam,
              optimizer__weight_decay=args.weight_decay
                         ) 
    
    clf = rnn.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_train)
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    auroc_train_final = roc_auc_score(y_train, y_pred_proba_pos)
    print('AUROC with LSTM (Train) : %.2f'%auroc_train_final)

    y_pred_proba = clf.predict_proba(X_test)
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    auroc_test_final = roc_auc_score(y_test, y_pred_proba_pos)
    print('AUROC with LSTM (Test) : %.2f'%auroc_test_final)



def convert_proba_to_binary(probabilites):
    return [0 if probs[0] > probs[1] else 1 for probs in probabilites]

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


if __name__ == '__main__':
    main()
    
