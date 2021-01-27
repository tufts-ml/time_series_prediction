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
PROJECT_REPO_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'rnn'))


from dataset_loader import TidySequentialDataCSVLoader
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
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from CNNBinaryClassifier import CNNBinaryClassifier
from CNNBinaryClassifierModule import compute_linear_layer_input_dims

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
    parser.add_argument('--n_filters', type=int, default=32,
                        help='Number of filters')
    parser.add_argument('--kernel_size', type=int, default=1,
                        help='size of eack kernel')
    parser.add_argument('--n_conv_layers', type=int, default=1,
                        help='number of convolutional layers') 
    parser.add_argument('--stride', type=int, default=1,
                        help='stride')  
    parser.add_argument('--pool_size', type=int, default=4,
                        help='max pool size')  
    parser.add_argument('--dense_units', type=int, default=128,
                        help='number of units in fully connected layer')     
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
    
    print('number of time points : %s\nnumber of features : %s\n'%(T,F))
    
    # add class weights    
    
    # set class weights as 1/(number of samples in class) for each class to handle class imbalance
    class_weights = torch.tensor([1/(y_train==0).sum(),
                                  1/(y_train==1).sum()]).double()
    
    print('Number of training sequences : %s'%N)
    print('Number of test sequences : %s'%X_test.shape[0])
    print('Ratio positive in train : %.2f'%((y_train==1).sum()/len(y_train)))
    print('Ratio positive in test : %.2f'%((y_test==1).sum()/len(y_test)))
    
    
    torch.manual_seed(0)
    
    # define conv params
#     channels = [F, 10, 10]
#     kernel_sizes = [3, 3]
#     strides = [2, 2]
#     paddings = [1, 1]
#     pools = [3, 3]
    channels = [F]
    kernel_sizes = []
    strides=[]
    paddings = []
    pools = []
    for i in range(args.n_conv_layers):
        channels.append(args.n_filters)
        kernel_sizes.append(args.kernel_size)
        strides.append(args.stride)
        pools.append(args.pool_size)
        paddings.append(1)
        
    
    # reshape train test to (N, F, T) for pytorch cnn
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))
    
    # calculate inputs to the linear layer
    linear_in = compute_linear_layer_input_dims(torch.from_numpy(X_train[:1, :, :]), channels, kernel_sizes, strides, paddings, pools)
        
    cnn = CNNBinaryClassifier(
          max_epochs=1000,
          batch_size=args.batch_size,
          device=device,
          lr=args.lr,
          callbacks=[
          EpochScoring('roc_auc', lower_is_better=False, on_train=True, name='aucroc_score_train'),
          EpochScoring('roc_auc', lower_is_better=False, on_train=False, name='aucroc_score_valid'),
#               EarlyStopping(monitor='aucroc_score_valid', patience=5, threshold=0.002, threshold_mode='rel',
#                                              lower_is_better=False),
#               LRScheduler(policy=ReduceLROnPlateau, mode='max', monitor='aucroc_score_valid', patience=10),
#               compute_grad_norm,
#               GradientNormClipping(gradient_clip_value=0.5, gradient_clip_norm_type=2),
          Checkpoint(monitor='aucroc_score_valid', f_history=os.path.join(args.output_dir, args.output_filename_prefix+'.json')),
          TrainEndCheckpoint(dirname=args.output_dir, fn_prefix=args.output_filename_prefix),
          ],
          criterion=torch.nn.CrossEntropyLoss,
          criterion__weight=class_weights,
          train_split=skorch.dataset.CVSplit(args.validation_size),
          module__channels=channels,
          module__kernel_sizes=kernel_sizes,
          module__strides=strides,
          module__paddings=paddings,
          module__pools=pools,
          module__linear_in_units=linear_in,
          module__linear_out_units=args.dense_units,
          module__dropout_proba=args.dropout,
          optimizer=torch.optim.Adam,
          optimizer__weight_decay=args.weight_decay,
          optimizer__lr = args.lr
                     ) 
    
    clf = cnn.fit(X_train, y_train)
    y_score_test = clf.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_score_test)
    print('AUC on test set : %.4f'%test_auc)
    
#     # save the model history
#     training_hist_df = pd.DataFrame(model.history.history) 
#     training_hist_df.loc[:, 'test_auc'] = test_auc
#     training_hist_csv = os.path.join(args.output_dir, args.output_filename_prefix+'.csv')
#     training_hist_df.to_csv(training_hist_csv, index=False)
    
    
#     # save the model
#     model_file = os.path.join(args.output_dir, args.output_filename_prefix+'.model')
#     model.save(model_file)
    
    
    
            
def get_sequence_lengths(X_NTF, pad_val):
    N = X_NTF.shape[0]
    seq_lens_N = np.zeros(N, dtype=np.int64)
    for n in range(N):
        bmask_T = np.all(X_NTF[n] == 0, axis=-1)
        seq_lens_N[n] = np.searchsorted(bmask_T, 1)
    return seq_lens_N
    


if __name__ == '__main__':
    main()
    

