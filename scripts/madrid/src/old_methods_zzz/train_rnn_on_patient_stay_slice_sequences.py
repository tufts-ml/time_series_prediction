import os
import argparse
import pandas as pd
import numpy as np
import sys
DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'rnn'))
from dataset_loader import TidySequentialDataCSVLoader
from utils import load_data_dict_json
import json
from feature_transformation import (parse_id_cols, parse_output_cols, parse_feature_cols, parse_time_col, get_fenceposts)

import torch
import skorch
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                             balanced_accuracy_score, confusion_matrix, 
                             roc_auc_score, make_scorer)

from RNNBinaryClassifier import RNNBinaryClassifier
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


def get_tslice_x_y(x, y, tstops_df, id_cols, time_col):
    '''Filters the full sequence by tslice'''
    x_curr_tslice = pd.merge(x, tstops_df, on=id_cols, how='inner')
    curr_slice_tinds = x_curr_tslice[time_col]<=x_curr_tslice['tstop']
    x_curr_tslice = x_curr_tslice.loc[curr_slice_tinds, :].copy()
    x_curr_tslice.drop(columns=['tstop'], inplace=True)
    curr_ids = x_curr_tslice[id_cols].drop_duplicates(subset=id_cols)
    y_curr_tslice = pd.merge(y, curr_ids, on=id_cols, how='inner')
    
    return x_curr_tslice, y_curr_tslice


def get_sequence_lengths(X_NTF, pad_val):
    N = X_NTF.shape[0]
    seq_lens_N = np.zeros(N, dtype=np.int64)
    for n in range(N):
        bmask_T = np.all(X_NTF[n] == 0, axis=-1)
        seq_lens_N[n] = np.searchsorted(bmask_T, 1)
    return seq_lens_N

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test_split_dir', type=str)
    parser.add_argument('--outcome_col_name', type=str)
    parser.add_argument('--train_tslices', type=str)
    parser.add_argument('--tstops_dir', type=str)
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
    parser.add_argument('--pretrained_model_dir', type=str, default=None,
                        help='load pretrained model from this dir if not None. If None, then start from scratch')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='directory where trained model and loss curves over epochs are saved')
    parser.add_argument('--output_filename_prefix', type=str, default=None, 
                        help='prefix for the training history jsons and trained classifier')   
    args = parser.parse_args()
    
    # Load x-train, ytrain and x-test, ytest
    print('Loading full sequence train-test data...')
    x_train = pd.read_csv(os.path.join(args.train_test_split_dir, 'x_train.csv.gz'))
    x_test = pd.read_csv(os.path.join(args.train_test_split_dir, 'x_test.csv.gz'))
    y_train = pd.read_csv(os.path.join(args.train_test_split_dir, 'y_train.csv.gz'))
    y_test = pd.read_csv(os.path.join(args.train_test_split_dir, 'y_test.csv.gz'))
    x_data_dict = load_data_dict_json(os.path.join(args.train_test_split_dir, 'x_dict.json'))
    y_data_dict = load_data_dict_json(os.path.join(args.train_test_split_dir, 'y_dict.json'))
    
    
    max_T_train = 0
    max_T_test = 0
    id_cols = parse_id_cols(x_data_dict)
    time_col = parse_time_col(x_data_dict)
    feature_cols = parse_feature_cols(x_data_dict)
    
    x_train[feature_cols] = x_train[feature_cols].astype(np.float32)
    x_test[feature_cols] = x_test[feature_cols].astype(np.float32)
    
    # Get 2 different train and test dataframes divided by slice
    train_tensors_per_tslice_list = []
    test_tensors_per_tslice_list = []
    for ind, tslice in enumerate(args.train_tslices.split(' ')):
        # Get the tstops_df for each patient-stay-slice
        tstops_df = pd.read_csv(os.path.join(args.tstops_dir, 'TSLICE={tslice}', 
                                             'tstops_filtered_{tslice}_hours.csv.gz').format(tslice=tslice))
        x_train_curr_tslice, y_train_curr_tslice = get_tslice_x_y(x_train, y_train, tstops_df, id_cols, time_col)
        x_test_curr_tslice, y_test_curr_tslice = get_tslice_x_y(x_test, y_test, tstops_df, id_cols, time_col)
        
        # limit sequence length
        reduced_T = 200
        
        print('Getting train and test sets for all patient stay slices...')
        # Pass each of the 3 dataframes through dataset_loader and 3 different tensors
        train_vitals = TidySequentialDataCSVLoader(
            x_csv_path=x_train_curr_tslice,
            y_csv_path=y_train_curr_tslice,
            x_col_names=feature_cols,
            idx_col_names=id_cols,
            y_col_name='clinical_deterioration_outcome',
            y_label_type='per_sequence',
            batch_size=45000,
            max_seq_len=reduced_T
        )

        test_vitals = TidySequentialDataCSVLoader(
            x_csv_path=x_test_curr_tslice,
            y_csv_path=y_test_curr_tslice,
            x_col_names=feature_cols,
            idx_col_names=id_cols,
            y_col_name='clinical_deterioration_outcome',
            y_label_type='per_sequence', 
            batch_size=10,
            max_seq_len=reduced_T
        )
        del x_train_curr_tslice, x_test_curr_tslice, y_train_curr_tslice, y_test_curr_tslice
        
        X_train_tensor_curr_tslice, y_train_tensor_curr_tslice = train_vitals.get_batch_data(batch_id=0)
        X_test_tensor_curr_tslice, y_test_tensor_curr_tslice = test_vitals.get_batch_data(batch_id=0)
        
        del train_vitals, test_vitals
        
        curr_T_train = X_train_tensor_curr_tslice.shape[1]
        curr_T_test = X_test_tensor_curr_tslice.shape[1]
        
        
        if curr_T_train > reduced_T:
            print('Unable to handle %s time points due to memory issues.. limiting to %s time points'%(curr_T_train, reduced_T))
            X_train_tensor_curr_tslice = X_train_tensor_curr_tslice[:,:reduced_T,:]
            curr_T_train = reduced_T

        print('Merging into a single large tensor')
        # make all the patient-stay-slice sequences of equal timesteps before merging
        if curr_T_train>max_T_train:
            max_T_train = curr_T_train

        if curr_T_test>max_T_test:
            max_T_test = curr_T_test

        X_train_tensor_curr_tslice = np.pad(X_train_tensor_curr_tslice, 
                                            ((0,0), (0, max_T_train-curr_T_train), (0,0)), 'constant')
        X_test_tensor_curr_tslice = np.pad(X_test_tensor_curr_tslice, 
                                            ((0,0), (0, max_T_test-curr_T_test), (0,0)), 'constant')
        
        # Merge the 3 tensors into a single large tensor
        if ind==0:
            X_train_tensor = X_train_tensor_curr_tslice
            y_train_tensor = y_train_tensor_curr_tslice
            X_test_tensor = X_test_tensor_curr_tslice
            y_test_tensor = y_test_tensor_curr_tslice
        else:
            del x_train, x_test
            
            X_train_tensor = np.vstack((X_train_tensor, X_train_tensor_curr_tslice))
            y_train_tensor = np.hstack((y_train_tensor, y_train_tensor_curr_tslice))
            X_test_tensor = np.vstack((X_test_tensor, X_test_tensor_curr_tslice))
            y_test_tensor = np.hstack((y_test_tensor, y_test_tensor_curr_tslice))   

        del X_train_tensor_curr_tslice, X_test_tensor_curr_tslice
    
    
    X_train_tensor[np.isnan(X_train_tensor)]=0

    ## Start RNN training
    print('Training RNN...')
    torch.manual_seed(args.seed)
    device = 'cpu'
    N,T,F = X_train_tensor.shape
    print('number of datapoints : %s\nnumber of time points : %s\nnumber of features : %s\npos sample ratio: %.4f'%(N, T,F,y_train_tensor.sum()/y_train_tensor.shape[0]))

    
    # get all the sequence lengths
    seq_lens_N = get_sequence_lengths(X_train_tensor, pad_val = 0)

    
    # set class weights as 1/(number of samples in class) for each class to handle class imbalance
    class_weights = torch.tensor([1/(y_train_tensor==0).sum(),
                                  1/(y_train_tensor==1).sum()]).float()

    # RNN
    if args.output_filename_prefix==None:
        output_filename_prefix = ('hiddens=%s-layers=%s-lr=%s-dropout=%s-weight_decay=%s-seed=%s'%(args.hidden_units, 
                                                                       args.hidden_layers, 
                                                                       args.lr, 
                                                                       args.dropout, args.weight_decay, args.seed))
    else:
        output_filename_prefix = args.output_filename_prefix
    
    compute_grad_norm = ComputeGradientNorm(norm_type=2) 
    print('RNN parameters : '+ output_filename_prefix)
        
#     if args.pretrained_model_dir is None:
    rnn = RNNBinaryClassifier(
              max_epochs=100,
              batch_size=args.batch_size,
              device=device,
              lr=args.lr,
              callbacks=[
              EpochScoring('roc_auc', lower_is_better=False, on_train=True, name='aucroc_score_train'),
              EpochScoring('roc_auc', lower_is_better=False, on_train=False, name='aucroc_score_valid'),
              EarlyStopping(monitor='aucroc_score_valid', patience=25, threshold=0.00001, threshold_mode='abs', lower_is_better=False),
#              LRScheduler(policy=ReduceLROnPlateau, mode='min', monitor='valid_loss', patience=25, factor=0.1,
#                   min_lr=0.0001, verbose=True),
              compute_grad_norm,
#               GradientNormClipping(gradient_clip_value=0.00001, gradient_clip_norm_type=2),
              Checkpoint(monitor='aucroc_score_valid', f_history=os.path.join(args.output_dir, output_filename_prefix+'.json')),
              TrainEndCheckpoint(dirname=args.output_dir, fn_prefix=output_filename_prefix),
              ],
              criterion=torch.nn.CrossEntropyLoss,
              criterion__weight=class_weights,
              train_split=skorch.dataset.CVSplit(args.validation_size, random_state=42),
              module__rnn_type='LSTM',
              module__n_layers=args.hidden_layers,
              module__n_hiddens=args.hidden_units,
              module__n_inputs=X_train_tensor.shape[-1],
              module__dropout_proba=args.dropout,
              optimizer=torch.optim.Adam,
              optimizer__weight_decay=args.weight_decay
                         ) 
    
    if args.pretrained_model_dir is not None:
        print('loading pre-trained model : %s' %(args.pretrained_model_dir + output_filename_prefix))
        
        # continue training at half the learning rate
        rnn.initialize()
        rnn.load_params(f_params=os.path.join(args.pretrained_model_dir,
                                      output_filename_prefix+'params.pt'),
                f_optimizer=os.path.join(args.pretrained_model_dir,
                                         output_filename_prefix+'optimizer.pt'),
                f_history=os.path.join(args.pretrained_model_dir,
                                       output_filename_prefix+'history.json'))
#         rnn.lr = rnn.lr*0.5
        clf = rnn.partial_fit(X_train_tensor, y_train_tensor)
    else:
        print('training model from scratch...')
        clf = rnn.fit(X_train_tensor, y_train_tensor)
 
    y_pred_proba = clf.predict_proba(X_train_tensor)
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    auroc_train_final = roc_auc_score(y_train_tensor, y_pred_proba_pos)
    print('AUROC with LSTM (Train) : %.2f'%auroc_train_final)

#     y_pred_proba = clf.predict_proba(X_test_tensor)
#     y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
#     auroc_test_final = roc_auc_score(y_test_tensor, y_pred_proba_pos)
#     print('AUROC with LSTM (Test) : %.2f'%auroc_test_final)
