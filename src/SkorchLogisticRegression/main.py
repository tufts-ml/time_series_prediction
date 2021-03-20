import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
import time
import torch
import skorch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                            balanced_accuracy_score, confusion_matrix, 
                            roc_auc_score, make_scorer, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from SkorchLogisticRegression import SkorchLogisticRegression
from skorch.callbacks import (Callback, LoadInitState, 
                              TrainEndCheckpoint, Checkpoint, 
                              EpochScoring, EarlyStopping, LRScheduler, GradientNormClipping, TrainEndCheckpoint)
from yattag import Doc
import matplotlib.pyplot as plt
import sys
cwd = sys.path[0]
sys.path.append(os.path.dirname(cwd))
from feature_transformation import parse_feature_cols, parse_output_cols
from utils import load_data_dict_json
from pickle import dump
from torch.optim.lr_scheduler import ReduceLROnPlateau


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sklearn LogisticRegression')

    parser.add_argument('--train_csv_files', type=str, required=True,
                        help='csv files for training')
    parser.add_argument('--test_csv_files', type=str, required=True,
                        help='csv files for testing')
    parser.add_argument('--outcome_col_name', type=str, required=True,
                        help='outcome column name')  
    parser.add_argument('--output_dir', type=str, required=True,
                        help='save dir of trained classifier and associated performance metrics') 
    parser.add_argument('--output_filename_prefix', type=str, required=True,
                        help='save dir of trained classifier and associated performance metrics') 
    parser.add_argument('--data_dict_files', type=str, required=True,
                        help='dict files for features and outcomes') 
    parser.add_argument('--validation_size', type=float, help='Validation size') 
    parser.add_argument('--key_cols_to_group_when_splitting', type=str,
                        help='columns for splitter') 
    parser.add_argument('--scoring', type=str,
                        help='loss scoring. Choose amongst binary_cross_entropy or surrogate_loss_tight')    
    parser.add_argument('--lr', type=float, help='learning rate')    
    parser.add_argument('--weight_decay', type=float, help='penalty for weights')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--merge_x_y', default=True,
                                type=lambda x: (str(x).lower() == 'true'), required=False)

    args = parser.parse_args()

    # read the data dictionaries
    print('Reading train-test data...')
    
    # read the data dict JSONs and parse the feature and outcome columns
    x_data_dict_file, y_data_dict_file = args.data_dict_files.split(',')
    x_data_dict = load_data_dict_json(x_data_dict_file)
    y_data_dict = load_data_dict_json(y_data_dict_file)
    
    feature_cols = parse_feature_cols(x_data_dict)
    
    
    df_by_split = dict()
    for split_name, csv_files in [
            ('train', args.train_csv_files.split(',')),
            ('test', args.test_csv_files.split(','))]:
        cur_df = None
        for csv_file in csv_files:

            # TODO use json data dict to load specific columns as desired types
            more_df =  pd.read_csv(csv_file)
            if cur_df is None:
                cur_df = more_df
            else:
                if args.merge_x_y:
                    cur_df = cur_df.merge(more_df, on=key_cols)
                else:
                    cur_df = pd.concat([cur_df, more_df], axis=1)
                    cur_df = cur_df.loc[:,~cur_df.columns.duplicated()]
        
        df_by_split[split_name] = cur_df
 
    outcome_col_name = args.outcome_col_name
    
    # Prepare data for classification
    x_train = df_by_split['train'][feature_cols].values.astype(np.float32)
    y_train = np.ravel(df_by_split['train'][outcome_col_name])
    
    x_test = df_by_split['test'][feature_cols].values.astype(np.float32)
    y_test = np.ravel(df_by_split['test'][outcome_col_name])        
    
    # get the feature columns
    class_weights_ = torch.tensor(np.asarray([1/((y_train==0).sum()), 1/((y_train==1).sum())]))
    
    # get the splitter to work
    
    # normalize data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_transformed = scaler.transform(x_train) 
    x_test_transformed = scaler.transform(x_test) 
    
    # set random seed
    torch.manual_seed(args.seed)
    
    # set max_epochs 
    max_epochs=150
    
    # define callbacks 
    epoch_scoring_precision_train = EpochScoring('precision', lower_is_better=False, on_train=True,
                                                                    name='precision_train')
    
    epoch_scoring_precision_valid = EpochScoring('precision', lower_is_better=False, on_train=False,
                                                                        name='precision_valid')
    
    epoch_scoring_recall_train = EpochScoring('recall', lower_is_better=False, on_train=True,
                                                                    name='recall_train')
    
    epoch_scoring_recall_valid = EpochScoring('recall', lower_is_better=False, on_train=False,
                                                  name='recall_valid')
    
    early_stopping_cp = EarlyStopping(monitor='precision_valid',
                                          patience=15, threshold=1e-10, threshold_mode='rel', 
                                          lower_is_better=False)
    
    cp = Checkpoint(monitor='precision_valid',
                    f_history=os.path.join(args.output_dir,
                                           args.output_filename_prefix+'.json'))
    
    train_end_cp = TrainEndCheckpoint(dirname=args.output_dir,
                                          fn_prefix=args.output_filename_prefix)
    
    lr_scheduler = LRScheduler(policy=ReduceLROnPlateau, mode='max', factor=0.1, patience=3, min_lr=1e-04, verbose=True)
    
    
    n_cv_folds = int(np.floor(1/args.validation_size))
    
    ## start training
    if args.scoring == 'cross_entropy_loss':
        logistic = SkorchLogisticRegression(n_features=len(feature_cols),
                                            l2_penalty_weights=args.weight_decay,
#                                             l2_penalty_bias=args.weight_decay,
                                            lr=args.lr,
                                            batch_size=args.batch_size, 
                                            max_epochs=max_epochs,
                                            train_split=skorch.dataset.CVSplit(n_cv_folds),
                                            loss_name=args.scoring,
                                            callbacks=[epoch_scoring_precision_train,
                                                       epoch_scoring_precision_valid,
                                                       epoch_scoring_recall_train,
                                                       epoch_scoring_recall_valid,
                                                       early_stopping_cp,
                                                       cp, train_end_cp],
                                           optimizer=torch.optim.Adam)
        print('Training Skorch Logistic Regression minimizing cross entropy loss...')
        logistic_clf = logistic.fit(x_train_transformed, y_train)
        
    elif args.scoring == 'surrogate_loss_tight':
        print('Warm starting by minimizing cross entropy loss first...')
        # warm start classifier and train with miminizing cross entropy loss first
        logistic_init = SkorchLogisticRegression(n_features=len(feature_cols),
                                            l2_penalty_weights=args.weight_decay,
#                                             l2_penalty_bias=args.weight_decay,
                                            lr=args.lr,
                                            batch_size=args.batch_size, 
                                            max_epochs=max_epochs,
                                            train_split=skorch.dataset.CVSplit(n_cv_folds),
                                            loss_name='cross_entropy_loss',
                                            callbacks=[epoch_scoring_precision_train,
                                                       epoch_scoring_precision_valid,
                                                       epoch_scoring_recall_train,
                                                       epoch_scoring_recall_valid,
                                                       early_stopping_cp,
                                                       cp, train_end_cp],
                                           optimizer=torch.optim.Adam)
        
        logistic_init_clf = logistic_init.fit(x_train_transformed, y_train)
        
        # transfer the weights and to a new instance of SkorchLogisticRegression and train minimizing surrogate loss
        print('Transferring weights and training to minimize surrogate loss...')
        
        logistic = SkorchLogisticRegression(n_features=len(feature_cols),
                                            l2_penalty_weights=args.weight_decay,
#                                             l2_penalty_bias=args.weight_decay,
                                            lr=args.lr,
                                            batch_size=args.batch_size, 
                                            max_epochs=max_epochs,
                                            train_split=skorch.dataset.CVSplit(n_cv_folds),
                                            loss_name=args.scoring,
                                            min_precision=0.9,
                                            constraint_lambda=100,
                                            callbacks=[epoch_scoring_precision_train,
                                                       epoch_scoring_precision_valid,
                                                       epoch_scoring_recall_train,
                                                       epoch_scoring_recall_valid,
#                                                        lr_scheduler,
                                                       early_stopping_cp,
                                                       cp, train_end_cp],
                                           optimizer=torch.optim.Adam)
        
        logistic.initialize()
        # load the past history
        logistic.load_params(checkpoint=cp)
        
        # load the weights
        logistic.module_.linear_transform_layer.weight.data = logistic_init_clf.module_.linear_transform_layer.weight.data
        logistic.module_.linear_transform_layer.bias.data = logistic_init_clf.module_.linear_transform_layer.bias.data


        logistic_clf = logistic.partial_fit(x_train_transformed, y_train)
    
    # save the scaler
    pickle_file = os.path.join(args.output_dir, 'scaler.pkl')
    dump(scaler, open(pickle_file, 'wb'))
    
    # print precision on train and validation
    y_train_pred = logistic_clf.predict(x_train_transformed)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
#     print('Train precision when minimizing %s loss :%.3f'%(args.scoring, precision_train))
    #
    y_test_pred = logistic_clf.predict(x_test_transformed)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
#     print('Test precision when minimizing %s loss :%.3f'%(args.scoring, precision_test))
    
    
    # compute the TP's, FP's, TN's and FN's
    y_train = y_train[:, np.newaxis] 
    y_test = y_test[:, np.newaxis]
    
    TP_train = np.sum(np.logical_and(y_train == 1, y_train_pred == 1))
    FP_train = np.sum(np.logical_and(y_train == 0, y_train_pred == 1))
    TN_train = np.sum(np.logical_and(y_train == 0, y_train_pred == 0))
    FN_train = np.sum(np.logical_and(y_train == 1, y_train_pred == 0))
    
    TP_test = np.sum(np.logical_and(y_test == 1, y_test_pred == 1))
    FP_test = np.sum(np.logical_and(y_test == 0, y_test_pred == 1))
    TN_test = np.sum(np.logical_and(y_test == 0, y_test_pred == 0))
    FN_test = np.sum(np.logical_and(y_test == 1, y_test_pred == 0))    
    
    
    f_out_name = os.path.join(args.output_dir, args.output_filename_prefix+'.txt')
    f_out = open(f_out_name, 'w')
    
    print_st_tr = "Training performance minimizing %s loss | Precision %.3f | Recall %.3f | TP %5d | FP %5d | TN %5d | FN %5d"%(args.scoring, precision_train, recall_train, TP_train, FP_train, TN_train, FN_train)
    print_st_te = "Test performance minimizing %s loss | Precision %.3f | Recall %.3f | TP %5d | FP %5d | TN %5d | FN %5d"%(args.scoring, precision_test, recall_test,TP_test, FP_test, TN_test, FN_test) 
        
    
    print(print_st_tr)
    print(print_st_te)
    f_out.write(print_st_tr + ' \n ' + print_st_te)
    f_out.close()
