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
from feature_transformation import (parse_id_cols, parse_output_cols, parse_feature_cols, parse_time_cols)

import torch
import skorch
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                             balanced_accuracy_score, confusion_matrix, 
                             roc_auc_score, make_scorer)
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
    parser.add_argument('--pretrained_model_dir', type=str, default=None,
                        help='load pretrained model from this dir if not None. If None, then start from scratch')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='directory where trained model and loss curves over epochs are saved')
    parser.add_argument('--output_filename_prefix', type=str, default=None, 
                        help='prefix for the training history jsons and trained classifier')   
    args = parser.parse_args()
    
    # Load x-train, ytrain and x-test, ytest
    print('Loading full sequence train-test data...')
    x_train = pd.read_csv(os.path.join(args.train_test_split_dir, 'x_train.csv'))
    x_test = pd.read_csv(os.path.join(args.train_test_split_dir, 'x_test.csv'))
    y_train = pd.read_csv(os.path.join(args.train_test_split_dir, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(args.train_test_split_dir, 'y_test.csv'))
    x_data_dict = load_data_dict_json(os.path.join(args.train_test_split_dir, 'x_dict.json'))
    y_data_dict = load_data_dict_json(os.path.join(args.train_test_split_dir, 'y_dict.json'))
    
    
    max_T_train = 0
    max_T_test = 0
    id_cols = parse_id_cols(x_data_dict)
    time_col = 'hours_in'
    feature_cols = parse_feature_cols(x_data_dict)
    
    # Get 2 different train and test dataframes divided by slice
    train_tensors_per_tslice_list = []
    test_tensors_per_tslice_list = []
    for ind, tslice in enumerate(args.train_tslices.split(' ')):
        # Get the tstops_df for each patient-stay-slice
        tstops_df = pd.read_csv(os.path.join(args.tstops_dir, 'TSLICE={tslice}', 
                                             'tstops_filtered_{tslice}_hours.csv').format(tslice=tslice))
        x_train_curr_tslice, y_train_curr_tslice = get_tslice_x_y(x_train, y_train, tstops_df, id_cols, time_col)
        x_test_curr_tslice, y_test_curr_tslice = get_tslice_x_y(x_test, y_test, tstops_df, id_cols, time_col)
        
        print('Getting train and test sets for all patient stay slices...')
        # Pass each of the 3 dataframes through dataset_loader and 3 different tensors
        train_vitals = TidySequentialDataCSVLoader(
            x_csv_path=x_train_curr_tslice,
            y_csv_path=y_train_curr_tslice,
            x_col_names=feature_cols,
            idx_col_names=id_cols,
            y_col_name='mort_hosp',
            y_label_type='per_sequence',
            batch_size=25000
        )

        test_vitals = TidySequentialDataCSVLoader(
            x_csv_path=x_test_curr_tslice,
            y_csv_path=y_test_curr_tslice,
            x_col_names=feature_cols,
            idx_col_names=id_cols,
            y_col_name='mort_hosp',
            y_label_type='per_sequence', 
            batch_size=10
        )
        del x_train_curr_tslice, x_test_curr_tslice, y_train_curr_tslice, y_test_curr_tslice
        
        X_train_tensor_curr_tslice, y_train_tensor_curr_tslice = train_vitals.get_batch_data(batch_id=0)
        X_test_tensor_curr_tslice, y_test_tensor_curr_tslice = test_vitals.get_batch_data(batch_id=0)
        
        curr_T_train = X_train_tensor_curr_tslice.shape[1]
        curr_T_test = X_test_tensor_curr_tslice.shape[1]
        
        reduced_T = 320
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
            X_train_tensor = np.vstack((X_train_tensor, X_train_tensor_curr_tslice))
            y_train_tensor = np.hstack((y_train_tensor, y_train_tensor_curr_tslice))
            X_test_tensor = np.vstack((X_test_tensor, X_test_tensor_curr_tslice))
            y_test_tensor = np.hstack((y_test_tensor, y_test_tensor_curr_tslice))   

        del X_train_tensor_curr_tslice, X_test_tensor_curr_tslice
    
    del x_train, x_test
    
    ## Start RNN training
    print('Training CNN...')
    N,T,F = X_train_tensor.shape
    print('number of time points : %s\nnumber of features : %s\npos sample ratio: %.4f'%(T,F,y_train_tensor.sum()/y_train_tensor.shape[0]))
    # add class weights
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
#     class_weights = dict(zip(range(len(class_weights)), class_weights))
    
    # convert y_train to categorical
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.validation_size, random_state=213)
    
    print('number of time points : %s\nnumber of features : %s\n'%(T,F))
    set_random_seed(args.seed)
    model = keras.Sequential()
    for i in range(args.n_conv_layers):
        model.add(keras.layers.Conv1D(filters=args.n_filters, kernel_size=args.kernel_size, activation='relu', strides=args.stride))
    model.add(keras.layers.Dropout(args.dropout)) 
    model.add(keras.layers.MaxPooling1D(pool_size=args.pool_size))
    model.add(keras.layers.Flatten()) 
    model.add(keras.layers.Dense(args.dense_units, activation='relu')) 
    model.add(keras.layers.Dense(2, activation='softmax')) 
    
    # set optimizer
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', keras.metrics.AUC()])
    
    # set early stopping
    early_stopping = EarlyStopping(monitor='val_auc', patience=20, mode='max', verbose=1)
    
    
    model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val),
             callbacks=[early_stopping], 
             class_weight=class_weights, batch_size=args.batch_size) 
    
    
    val_perf = model.evaluate(X_val, y_val)
    val_auc = val_perf[-1]
    print('AUC on val set : %.2f'%val_auc)
    test_perf = model.evaluate(X_test, y_test) 
    test_auc = test_perf[-1]
    print('AUC on test set : %.2f'%test_auc)
    
    # save the model history
    training_hist_df = pd.DataFrame(model.history.history) 
    training_hist_df.loc[:, 'test_auc'] = test_auc
    training_hist_csv = os.path.join(args.output_dir, args.output_filename_prefix+'.csv')
    training_hist_df.to_csv(training_hist_csv, index=False)
    
    # save the model
    model_file = os.path.join(args.output_dir, args.output_filename_prefix+'.model')
    model.save(model_file)