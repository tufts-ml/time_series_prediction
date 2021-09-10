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
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from tensorflow import set_random_seed
from sklearn.metrics import roc_auc_score


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
    
    
    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
             callbacks=[early_stopping], 
             class_weight=class_weights, batch_size=args.batch_size) 
    

    y_score_val = model.predict_proba(X_val)
    val_auc = roc_auc_score(y_val, y_score_val)
    print('AUC on val set : %.4f'%val_auc)
    
    y_score_test = model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_score_test)
    print('AUC on val set : %.4f'%test_auc)
    
    # save the model history
    training_hist_df = pd.DataFrame(model.history.history) 
    training_hist_df.loc[:, 'test_auc'] = test_auc
    training_hist_csv = os.path.join(args.output_dir, args.output_filename_prefix+'.csv')
    training_hist_df.to_csv(training_hist_csv, index=False)
    
    
    # save the model
    model_file = os.path.join(args.output_dir, args.output_filename_prefix+'.model')
    model.save(model_file)    
            
def get_sequence_lengths(X_NTF, pad_val):
    N = X_NTF.shape[0]
    seq_lens_N = np.zeros(N, dtype=np.int64)
    for n in range(N):
        bmask_T = np.all(X_NTF[n] == 0, axis=-1)
        seq_lens_N[n] = np.searchsorted(bmask_T, 1)
    return seq_lens_N
    


if __name__ == '__main__':
    main()