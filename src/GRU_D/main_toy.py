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
from sklearn.metrics import (roc_auc_score, make_scorer)
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
    parser.add_argument('--train_csv_files', type=str, required=True)
    parser.add_argument('--valid_csv_files', type=str, default=None)
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
    
    
    args = parser.parse_args()

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
    

    valid_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_valid_csv_filename,
        y_csv_path=y_valid_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=args.outcome_col_name,
        y_label_type='per_sequence')
    
    
    train_x_NTD, train_y = train_vitals.get_batch_data(batch_id=0)
    valid_x_NTD, valid_y = valid_vitals.get_batch_data(batch_id=0)
    test_x_NTD, test_y = test_vitals.get_batch_data(batch_id=0)
    
    

    # Mask = 1 if present
    train_x_mask_NTD = 1-np.isnan(train_x_NTD).astype(int)
    valid_x_mask_NTD = 1-np.isnan(valid_x_NTD).astype(int)
    test_x_mask_NTD = 1-np.isnan(test_x_NTD).astype(int)

    # Center data at 0 and only use estimates from training set
    _, T, D = train_x_NTD.shape


    # I get good performance only when I normalize to 0-1. 
#     for d in range(D):
#         den = np.nanstd(train_x_NTD[:, :, d])

#         train_x_NTD[:, :, d] = (train_x_NTD[:, :, d] - np.nanmean(train_x_NTD[:, :, d]))

#         valid_x_NTD[:, :, d] = (valid_x_NTD[:, :, d] - np.nanmean(train_x_NTD[:, :, d]))

#         test_x_NTD[:, :, d] = (test_x_NTD[:, :, d] - np.nanmean(train_x_NTD[:, :, d]))

    # Make missing data not nan
    train_x_NTD[np.isnan(train_x_NTD)] = 0
    valid_x_NTD[np.isnan(valid_x_NTD)] = 0
    test_x_NTD[np.isnan(test_x_NTD)] = 0
    
    
    just_8_hrs = np.arange(0, 8)
    # create a timestep tensor that ix (observations x timesteps x 1)
    train_timestep = np.expand_dims(just_8_hrs * np.ones(train_x_NTD.shape[:-1]), -1)
    valid_timestep = np.expand_dims(just_8_hrs * np.ones(valid_x_NTD.shape[:-1]), -1)
    test_timestep = np.expand_dims(just_8_hrs * np.ones(test_x_NTD.shape[:-1]), -1)

    input_dim = D
    output_dim = 1
    output_activation = 'sigmoid'
    predefined_model = 'GRUD'
    use_bidirectional_rnn=False
    outcome="did_overheat_binary_label"
    recurrent_dim = [32]#256
    hidden_dim = [8]#128
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
    
    model.summary()
    model.fit(x=[train_x_NTD, train_x_mask_NTD, train_timestep], y=train_y,
         epochs=500,
         batch_size=train_x_NTD.shape[0],
         # No validation happening, just putting test data here to see test metrics
         validation_data=([valid_x_NTD, valid_x_mask_NTD, valid_timestep],
                         valid_y),
              callbacks=[
                         tf.keras.callbacks.EarlyStopping(monitor='val_AUC', mode='max', verbose=1, patience=50),
#                          tf.keras.callbacks.LearningRateScheduler(scheduler)
                        ])
    
    model_hist_df = pd.DataFrame(model.history.history)
    
    
    y_train_pred_proba = model.predict([train_x_NTD, train_x_mask_NTD, train_timestep])
    train_roc_auc = roc_auc_score(train_y, y_train_pred_proba)
    print('ROC AUC on train : %.5f'%train_roc_auc)
    
    y_valid_pred_proba = model.predict([valid_x_NTD, valid_x_mask_NTD, valid_timestep])
    valid_roc_auc = roc_auc_score(valid_y, y_valid_pred_proba)
    print('ROC AUC on valid : %.5f'%valid_roc_auc)
    
    y_test_pred_proba = model.predict([test_x_NTD, test_x_mask_NTD, test_timestep])
    test_roc_auc = roc_auc_score(test_y, y_test_pred_proba)
    print('ROC AUC on test : %.5f'%test_roc_auc)        
    
    # save the model
    model_filename = os.path.join(args.output_dir, args.output_filename_prefix+'-weights.h5')
    model.save_weights(model_filename, save_format='h5')
    
    # save some additional performance 
    perf_save_file = os.path.join(args.output_dir, 'final_perf_'+args.output_filename_prefix+'.csv')
    model_perf_df = pd.DataFrame([{'train_AUC' : train_roc_auc, 
                                   'valid_AUC' : valid_roc_auc, 
                                   'test_AUC' : test_roc_auc}])
    
    
    model_perf_df.to_csv(perf_save_file, index=False)