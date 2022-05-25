import sys
import os
import json
PROJECT_SRC_DIR = '/cluster/tufts/hugheslab/prath01/projects/time_series_prediction/src/'
sys.path.append(PROJECT_SRC_DIR)
sys.path.append(os.path.join(PROJECT_SRC_DIR, "rnn"))
sys.path.append(os.path.join(PROJECT_SRC_DIR, "GRU_D", "GRU-D"))
sys.path.append(os.path.abspath("../src"))
from dataset_loader import TidySequentialDataCSVLoader
from feature_transformation import parse_id_cols, parse_output_cols, parse_feature_cols, parse_id_cols, parse_time_cols
from numpy.random import seed
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import load_data_dict_json
from sklearn.metrics import (roc_auc_score, average_precision_score)
sys.path.append('./GRU-D/')
from models import create_grud_model
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
    

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRU-D fitting')
    parser.add_argument('--outcome_col_name', type=str, required=True)
    parser.add_argument('--train_np_files', type=str, required=True)
    parser.add_argument('--test_np_files', type=str, required=True)
    parser.add_argument('--valid_np_files', type=str, default=None)
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
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout for GRUD')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='directory where trained model and loss curves over epochs are saved')
    parser.add_argument('--output_filename_prefix', type=str, default=None, 
                        help='prefix for the training history jsons and trained classifier')
    parser.add_argument('--l2_penalty', type=float, default=1e-4,
                        help='weight decay l2')
    parser.add_argument('--perc_labelled', type=str, default='100',
                        help='weight decay l2')
    
    
    args = parser.parse_args()

    x_train_np_filename, y_train_np_filename = args.train_np_files.split(',')
    x_test_np_filename, y_test_np_filename = args.test_np_files.split(',')
    x_valid_np_filename, y_valid_np_filename = args.valid_np_files.split(',')
    
    
    X_train = np.load(x_train_np_filename)
    X_test = np.load(x_test_np_filename)
    y_train = np.load(y_train_np_filename)
    y_test = np.load(y_test_np_filename)
    X_valid = np.load(x_valid_np_filename)
    y_valid = np.load(y_valid_np_filename)
    
    # get only the labelled data
    labelled_inds_tr = ~np.isnan(y_train)
    labelled_inds_va = ~np.isnan(y_valid)
    labelled_inds_te = ~np.isnan(y_test)
    
    X_train_labelled = X_train[labelled_inds_tr].copy()
    if args.perc_labelled=='100':# hack for MixMatch with fully observed
        X_train_unlabelled = X_train_labelled.copy()
    else:
        X_train_unlabelled = X_train[~labelled_inds_tr]
    y_train_labelled = y_train[labelled_inds_tr]
    
    X_valid_labelled = X_valid[labelled_inds_va]
#     X_valid_unlabelled = X_valid[~labelled_inds_va]
    y_valid_labelled = y_valid[labelled_inds_va]  
    
    
    X_test_labelled = X_test[labelled_inds_te]
#     X_test_unlabelled = X_test[~labelled_inds_te]
    y_test_labelled = y_test[labelled_inds_te]    
    
    
    train_x_mask_NTD = 1-np.isnan(X_train_labelled).astype(int)
    valid_x_mask_NTD = 1-np.isnan(X_valid_labelled).astype(int)
    test_x_mask_NTD = 1-np.isnan(X_test_labelled).astype(int)
    
    _, T, D = X_test_labelled.shape
    for d in range(D):
        
        # impute missing values by mean filling using estimates from full training set
        fill_vals = np.nanmean(X_train[:, :, d])
        
        X_train_labelled[:, :, d] = np.nan_to_num(X_train_labelled[:, :, d], nan=fill_vals)
        X_valid_labelled[:, :, d] = np.nan_to_num(X_valid_labelled[:, :, d], nan=fill_vals)
        X_test_labelled[:, :, d] = np.nan_to_num(X_test_labelled[:, :, d], nan=fill_vals)
        X_train_unlabelled[:, :, d] = np.nan_to_num(X_train_unlabelled[:, :, d], nan=fill_vals)
        
        
#         min max normalization
#         den = np.nanmax(X_train[:, :, d])-np.nanmin(X_train[:, :, d])
#         X_train_labelled[:, :, d] = (X_train_labelled[:, :, d] - np.nanmin(X_train[:, :, d]))/den
#         X_valid_labelled[:, :, d] = (X_valid_labelled[:, :, d] - np.nanmin(X_train[:, :, d]))/den
#         X_test_labelled[:, :, d] = (X_test_labelled[:, :, d] - np.nanmin(X_train[:, :, d]))/den
#         X_train_unlabelled[:, :, d] = (X_train_unlabelled[:, :, d] - np.nanmin(X_train[:, :, d]))/den
        
        
        # zscore normalization
        den = np.nanstd(X_train[:, :, d])
    
        X_train_labelled[:, :, d] = (X_train_labelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den
        X_valid_labelled[:, :, d] = (X_valid_labelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den
        X_test_labelled[:, :, d] = (X_test_labelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den
        X_train_unlabelled[:, :, d] = (X_train_unlabelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den

    # Mask = 1 if present
    train_x_NTD = X_train_labelled.copy()
    valid_x_NTD = X_valid_labelled.copy()
    test_x_NTD = X_test_labelled.copy()
    train_y = y_train_labelled.copy()
    valid_y = y_valid_labelled.copy()
    test_y = y_test_labelled.copy()
    

    # Center data at 0 and only use estimates from training set
    _, T, D = train_x_NTD.shape
    
    
    just_T_hrs = np.arange(0, T)
    # create a timestep tensor that ix (observations x timesteps x 1)
    train_timestep = np.expand_dims(just_T_hrs * np.ones(train_x_NTD.shape[:-1]), -1)
    valid_timestep = np.expand_dims(just_T_hrs * np.ones(valid_x_NTD.shape[:-1]), -1)
    test_timestep = np.expand_dims(just_T_hrs * np.ones(test_x_NTD.shape[:-1]), -1)

    input_dim = D
    output_dim = 1
    output_activation = 'sigmoid'
    predefined_model = 'GRUD'
    use_bidirectional_rnn=False
    outcome="mort_hosp"
    recurrent_dim = [256]#256#32
    hidden_dim = [128]#128#8
    l2_penalty = args.l2_penalty
    dropout=args.dropout
    
    # set random seed
    seed(args.seed)
    
    model = create_grud_model(input_dim=input_dim,
                              output_dim=output_dim,
                              output_activation=output_activation,
                              recurrent_dim=recurrent_dim,
                              hidden_dim=hidden_dim,
                              predefined_model=predefined_model,
                              use_bidirectional_rnn=use_bidirectional_rnn,
                              dropout=dropout, 
                              l2_penalty=l2_penalty)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
             metrics = [tf.keras.metrics.AUC(name='AUC'), 
                        tf.keras.metrics.AUC(curve='PR', name='AUPRC')])

    model.fit(x=[train_x_NTD, train_x_mask_NTD, train_timestep], y=train_y,
         epochs=500,
         batch_size=args.batch_size,
         validation_data=([valid_x_NTD, valid_x_mask_NTD, valid_timestep],
                         valid_y),
              callbacks=[
                         tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=5),
#                          tf.keras.callbacks.LearningRateScheduler(scheduler)
                        ])

    model_hist_df = pd.DataFrame(model.history.history)
    
    
    y_train_pred_proba = model.predict([train_x_NTD, train_x_mask_NTD, train_timestep])
    train_roc_auc = roc_auc_score(train_y, y_train_pred_proba)
    train_auprc = average_precision_score(train_y, y_train_pred_proba)
    print('ROC AUC on train : %.5f'%train_roc_auc)
    print('AUPRC on train : %.5f'%train_auprc)
    
    y_valid_pred_proba = model.predict([valid_x_NTD, valid_x_mask_NTD, valid_timestep])
    valid_roc_auc = roc_auc_score(valid_y, y_valid_pred_proba)
    valid_auprc = average_precision_score(valid_y, y_valid_pred_proba)
    print('ROC AUC on valid : %.5f'%valid_roc_auc)
    print('AUPRC on valid : %.5f'%valid_auprc)
    
    y_test_pred_proba = model.predict([test_x_NTD, test_x_mask_NTD, test_timestep])
    test_roc_auc = roc_auc_score(test_y, y_test_pred_proba)
    test_auprc = average_precision_score(test_y, y_test_pred_proba)
    
    print('ROC AUC on test : %.5f'%test_roc_auc)        
    print('AUPRC on test : %.5f'%valid_auprc)
    
    # save the model
    model_filename = os.path.join(args.output_dir, args.output_filename_prefix+'-weights.h5')
    model.save_weights(model_filename, save_format='h5')
    
    # save some additional performance 
    perf_save_file = os.path.join(args.output_dir, 'final_perf_'+args.output_filename_prefix+'.csv')
    model_perf_df = pd.DataFrame([{'train_AUC' : train_roc_auc, 
                                   'valid_AUC' : valid_roc_auc, 
                                   'test_AUC' : test_roc_auc,
                                   'train_AUPRC' : train_auprc, 
                                   'valid_AUPRC' : valid_auprc, 
                                   'test_AUPRC' : test_auprc}])
    
    
    model_perf_df.to_csv(perf_save_file, index=False)