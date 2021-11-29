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
                             roc_auc_score, make_scorer, precision_score, recall_score, average_precision_score)
from yattag import Doc
import matplotlib.pyplot as plt

from dataset_loader import TidySequentialDataCSVLoader
from RNNPerTStepBinaryClassifier import RNNPerTStepBinaryClassifier
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
from skorch.dataset import Dataset
from skorch.helper import predefined_split

# define AUPRC callback

def calc_auprc(net, X, y):
    y_pred_probas_NT2 = net.predict_proba(X)
    keep_inds = torch.logical_not(torch.all(torch.isnan(torch.FloatTensor(X.X)), dim=-1))
    return average_precision_score(y[keep_inds], y_pred_probas_NT2[keep_inds][:,1].detach().numpy())

def calc_auroc(net, X, y):
    y_pred_probas_NT2 = net.predict_proba(X)
    keep_inds = torch.logical_not(torch.all(torch.isnan(torch.FloatTensor(X.X)), dim=-1))
    return roc_auc_score(y[keep_inds], y_pred_probas_NT2[keep_inds][:,1].detach().numpy())

def main():
    parser = argparse.ArgumentParser(description='PyTorch RNN with variable-length numeric sequences wrapper')
    parser.add_argument('--outcome_col_name', type=str, required=True)
    parser.add_argument('--train_csv_files', type=str, required=True)
    parser.add_argument('--valid_csv_files', type=str, required=True)
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
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='directory where trained model and loss curves over epochs are saved')
    parser.add_argument('--output_filename_prefix', type=str, default=None, 
                        help='prefix for the training history jsons and trained classifier')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cpu'

    x_train_csv_filename, y_train_csv_filename = args.train_csv_files.split(',')
    x_valid_csv_filename, y_valid_csv_filename = args.valid_csv_files.split(',')
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
        y_label_type='per_tstep'
    )

    valid_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_valid_csv_filename,
        y_csv_path=y_valid_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=args.outcome_col_name,
        y_label_type='per_tstep'
    )
    
    test_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_test_csv_filename,
        y_csv_path=y_test_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=args.outcome_col_name,
        y_label_type='per_tstep'
    )

    X_train, y_train = train_vitals.get_batch_data(batch_id=0)
    X_valid, y_valid = valid_vitals.get_batch_data(batch_id=0)
    X_test, y_test = test_vitals.get_batch_data(batch_id=0)
    N,T,F = X_train.shape
    
#     from IPython import embed; embed()
#     X_train = (X_train - np.min(X_train))/(np.max(X_train)-np.min(X_train))
#     X_valid = (X_valid - np.min(X_train))/(np.max(X_train)-np.min(X_train))
#     X_test = (X_test - np.min(X_train))/(np.max(X_train)-np.min(X_train))
    
    valid_ds = Dataset(X_valid, y_valid) 
    
    print('number of time points : %s\nnumber of features : %s\n'%(T,F))
    
    # set class weights as 1/(number of samples in class) for each class to handle class imbalance
    class_weights = torch.tensor([1/(y_train==0).sum(),
                                  1/(y_train==1).sum()]).float()
    
    print('Number of training sequences : %s'%N)
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
    
    loss_early_stopping_cp =  EarlyStopping(monitor='valid_loss', patience=15, threshold=0.002, threshold_mode='rel',
                                            lower_is_better=True)
    
    rnn = RNNPerTStepBinaryClassifier(
              max_epochs=100,
              batch_size=args.batch_size,
              device=device,
              lr=args.lr,
              callbacks=[
                  EpochScoring(calc_auprc, lower_is_better=False, on_train=True, name='auprc_train'),
                  EpochScoring(calc_auprc, lower_is_better=False, on_train=False, name='auprc_valid'),
                  EpochScoring(calc_auroc, lower_is_better=False, on_train=True, name='auroc_train'),
                  EpochScoring(calc_auroc, lower_is_better=False, on_train=False, name='auroc_valid'),
#               EpochScoring(calc_precision, lower_is_better=False, on_train=True, name='precision_train'),
#               EpochScoring(calc_precision, lower_is_better=False, on_train=False, name='precision_valid'),
#               EpochScoring(calc_recall, lower_is_better=False, on_train=True, name='recall_train'),
#               EpochScoring(calc_recall, lower_is_better=False, on_train=False, name='recall_valid'),
#               EpochScoring('roc_auc', lower_is_better=False, on_train=True, name='aucroc_score_train'),
#               EpochScoring('roc_auc', lower_is_better=False, on_train=False, name='aucroc_score_valid'),
#                   EarlyStopping(monitor='auprc_valid', patience=5, threshold=0.002, threshold_mode='rel',
#                                                  lower_is_better=False),
#               LRScheduler(policy=ReduceLROnPlateau, mode='max', monitor='aucroc_score_valid', patience=10),
#                   compute_grad_norm,
#               GradientNormClipping(gradient_clip_value=0.5, gradient_clip_norm_type=2),
              loss_early_stopping_cp,
              Checkpoint(monitor='auprc_valid', f_history=os.path.join(args.output_dir, output_filename_prefix+'.json')),
              TrainEndCheckpoint(dirname=args.output_dir, fn_prefix=output_filename_prefix),
              ],
#               criterion=torch.nn.CrossEntropyLoss,
#               criterion__weight=class_weights,
              train_split=predefined_split(valid_ds),
              module__rnn_type='GRU',
              module__n_layers=args.hidden_layers,
              module__n_hiddens=args.hidden_units,
              module__n_inputs=X_train.shape[-1],
              module__dropout_proba=args.dropout,
              optimizer=torch.optim.Adam,
              optimizer__weight_decay=args.weight_decay
                         ) 
    
#     N=len(X_train)
#     X_train = X_train[:N]
#     y_train = y_train[:N]
    

    clf = rnn.fit(X_train, y_train)
    
    # get threshold with max recall at fixed precision
    fixed_precision=0.2
    
    # get predict probas for y=1 on validation set
    keep_inds_va = torch.logical_not(torch.all(torch.isnan(torch.FloatTensor(X_valid)), dim=-1))
    y_valid_pred_probas = clf.predict_proba(X_valid)[keep_inds_va][:,1].detach().numpy()
    
    unique_probas = np.unique(y_valid_pred_probas)
    thr_grid_G = np.linspace(np.percentile(unique_probas,1), max(unique_probas), 100)
        
    precision_scores_G, recall_scores_G = [np.zeros(thr_grid_G.size), np.zeros(thr_grid_G.size)]
#     y_valid_pred_probas = clf.predict_proba(torch.FloatTensor(X_valid))
    for gg, thr in enumerate(thr_grid_G): 
#             logistic_clf.module_.linear_transform_layer.bias.data = torch.tensor(thr_grid[gg]).double()
        curr_thr_y_preds = y_valid_pred_probas>=thr_grid_G[gg] 
        precision_scores_G[gg] = precision_score(y_valid[keep_inds_va], curr_thr_y_preds)
        recall_scores_G[gg] = recall_score(y_valid[keep_inds_va], curr_thr_y_preds) 
    
    
    keep_inds = precision_scores_G>=fixed_precision

    if keep_inds.sum()>0:
        print('Choosing threshold with precision >= %.3f'%fixed_precision)
    else:
        fixed_precision_old = fixed_precision
        fixed_precision = np.percentile(precision_scores_G, 99)
        keep_inds = precision_scores_G>=fixed_precision
        print('Could not find threshold with precision >= %.3f \n Choosing threshold to maximize recall at precision %.3f'%(fixed_precision_old, fixed_precision))
    
    thr_grid_G = thr_grid_G[keep_inds]
    precision_scores_G = precision_scores_G[keep_inds]
    recall_scores_G = recall_scores_G[keep_inds]
    thr_perf_df = pd.DataFrame(np.vstack([thr_grid_G[np.newaxis,:],
                                          precision_scores_G[np.newaxis,:],
                                          recall_scores_G[np.newaxis, :]]).T, 
                               columns=['thr', 'precision_score', 'recall_score'])

    print(thr_perf_df)
    best_ind = np.argmax(recall_scores_G)
    best_thr = thr_grid_G[best_ind]
    print('chosen threshold : %.3f'%best_thr)
    
    splits = ['train', 'valid', 'test']
#     data_splits = ((x_tr, y_tr), (x_va, y_va), (X_test, y_test))
    auroc_per_split,  auprc_per_split, precisions_per_split, recalls_per_split = [np.zeros(len(splits)),
                                                                np.zeros(len(splits)),
                                                                np.zeros(len(splits)),
                                                                np.zeros(len(splits))]
    
    
    for ii, (X, y) in enumerate([(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]):
        keep_inds = torch.logical_not(torch.all(torch.isnan(torch.FloatTensor(X)), dim=-1))
        y_pred_proba_pos = clf.predict_proba(X)[keep_inds][:,1].detach().numpy()
#         y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
        auroc_per_split[ii] = roc_auc_score(y[keep_inds], y_pred_proba_pos)
#         y_pred_proba_pos = np.asarray(y_pred_proba_pos)
        auprc_per_split[ii] = average_precision_score(y[keep_inds], y_pred_proba_pos)
        y_pred = y_pred_proba_pos>=best_thr
        precisions_per_split[ii] = precision_score(y[keep_inds], y_pred)
        recalls_per_split[ii] = recall_score(y[keep_inds], y_pred)
    
    auroc_train, auroc_valid, auroc_test = auroc_per_split
    auprc_train, auprc_valid, auprc_test = auprc_per_split
    precision_train, precision_valid, precision_test = precisions_per_split
    recall_train, recall_valid, recall_test = recalls_per_split
    
    # save performance
    perf_dict = {'auroc_train':auroc_train,
                'auroc_valid':auroc_valid,
                'auroc_test':auroc_test,
                'auprc_train':auprc_train,
                'auprc_valid':auprc_valid,
                'auprc_test':auprc_test,
                'precision_train':precision_train,
                'precision_valid':precision_valid,
                'precision_test':precision_test,
                'recall_train':recall_train,
                'recall_valid':recall_valid,
                'recall_test':recall_test,
                'threshold':best_thr}
    
    perf_df = pd.DataFrame([perf_dict])
    perf_csv = os.path.join(args.output_dir, output_filename_prefix+'.csv')
    print('Final performance on train, valid and test :\n')
    print(perf_df)
    
    print('Final performance saved to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)
    


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
    


if __name__ == '__main__':
    main()
