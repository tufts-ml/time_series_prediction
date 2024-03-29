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
                            roc_auc_score, average_precision_score, make_scorer, precision_score, recall_score)

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
from feature_transformation import parse_feature_cols, parse_output_cols, parse_id_cols
from utils import load_data_dict_json
from pickle import dump
from torch.optim.lr_scheduler import ReduceLROnPlateau
from split_dataset import Splitter
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from sklearn.linear_model import LogisticRegression
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

def read_csv_with_float32_dtypes(filename, nrows=None):
    # Sample 100 rows of data to determine dtypes.
    df_test = pd.read_csv(filename, nrows=100)

    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    
    if nrows is not None:
        df = pd.read_csv(filename, dtype=float32_cols, nrows=nrows)
    else:
        df = pd.read_csv(filename, dtype=float32_cols)
    
    return df


# define callbacks
def calc_surrogate_loss_skorch_callback(net, X, y):
    if isinstance(X, torch.utils.data.dataset.Subset):
        indices = X.indices
        X_NF = torch.DoubleTensor(X.dataset.X[indices])
    elif isinstance(X, skorch.dataset.Dataset):
        X_NF = torch.DoubleTensor(X.X)
    y_est_logits_ = net.module_.linear_transform_layer.forward(X_NF)[:,0]
    if net.loss_name=='surrogate_loss_loose':
        surr_loss = net.calc_surrogate_loss(y_true=y, y_est_logits_=y_est_logits_, X=X_NF, alpha=net.min_precision,
                                              lamb=net.constraint_lambda, bounds='loose')
    else:
        surr_loss = net.calc_surrogate_loss(y_true=y, y_est_logits_=y_est_logits_, X=X_NF, alpha=net.min_precision,
                                              lamb=net.constraint_lambda, bounds='tight')      
    return float(surr_loss.detach().numpy())

def calc_bce_loss_skorch_callback(net, X, y):
    if isinstance(X, torch.utils.data.dataset.Subset):
        indices = X.indices
        X_NF = torch.DoubleTensor(X.dataset.X[indices])
    elif isinstance(X, skorch.dataset.Dataset):
        X_NF = torch.DoubleTensor(X.X)
    y_est_logits_ = net.module_.linear_transform_layer.forward(X_NF)[:,0]
    bce_loss = net.calc_bce_loss(y_true=y, y_est_logits_=y_est_logits_, X=X_NF)
    
    return float(bce_loss.detach().numpy())

def calc_tp_lower_bound_skorch_callback(net, X, y):
    if isinstance(X, torch.utils.data.dataset.Subset):
        indices = X.indices
        X_NF = torch.DoubleTensor(X.dataset.X[indices])
    elif isinstance(X, skorch.dataset.Dataset):
        X_NF = torch.DoubleTensor(X.X)
    y_ = torch.DoubleTensor(y)
    y_est_logits_ = net.module_.linear_transform_layer.forward(X_NF)[:,0]
    _, tp_lower_bound = net.calc_fp_tp_bounds_better(y_, y_est_logits_)
    return float(tp_lower_bound.detach().numpy())

def calc_fp_upper_bound_skorch_callback(net, X, y):
    if isinstance(X, torch.utils.data.dataset.Subset):
        indices = X.indices
        X_NF = torch.DoubleTensor(X.dataset.X[indices])
    elif isinstance(X, skorch.dataset.Dataset):
        X_NF = torch.DoubleTensor(X.X)
    y_ = torch.DoubleTensor(y)
    y_est_logits_ = net.module_.linear_transform_layer.forward(X_NF)[:,0]
    fp_upper_bound, _ = net.calc_fp_tp_bounds_better(y_, y_est_logits_)
    return float(fp_upper_bound.detach().numpy())

def calc_tp(net, X, y):
    if isinstance(X, torch.utils.data.dataset.Subset):
        indices = X.indices
        X_NF = torch.DoubleTensor(X.dataset.X[indices])
    elif isinstance(X, skorch.dataset.Dataset):
        X_NF = torch.DoubleTensor(X.X)
    y_ = torch.DoubleTensor(y)
    y_est_logits_ = net.module_.linear_transform_layer.forward(X_NF)[:,0]
    _, tp = net.calc_fp_tp_bounds_ideal(y_, y_est_logits_)
    return float(tp.detach().numpy())

def calc_fp(net, X, y):
    if isinstance(X, torch.utils.data.dataset.Subset):
        indices = X.indices
        X_NF = torch.DoubleTensor(X.dataset.X[indices])
    elif isinstance(X, skorch.dataset.Dataset):
        X_NF = torch.DoubleTensor(X.X)
    y_ = torch.DoubleTensor(y)
    y_est_logits_ = net.module_.linear_transform_layer.forward(X_NF)[:,0]
    fp, _ = net.calc_fp_tp_bounds_ideal(y_, y_est_logits_)
    return float(fp.detach().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sklearn LogisticRegression')

    parser.add_argument('--train_csv_files', type=str, required=True,
                        help='csv files for training')
    parser.add_argument('--test_csv_files', type=str, required=True,
                        help='csv files for testing')
    parser.add_argument('--valid_csv_files', type=str, default=None, required=False,
                        help='csv files for testing')

    parser.add_argument('--outcome_col_name', type=str, required=True,
                        help='outcome column name')  
    parser.add_argument('--output_dir', type=str, required=True,
                        help='save dir of trained classifier and associated performance metrics') 
    parser.add_argument('--output_filename_prefix', type=str, required=True,
                        help='save dir of trained classifier and associated performance metrics') 
    parser.add_argument('--data_dict_files', type=str, required=True,
                        help='dict files for features and outcomes') 
    parser.add_argument('--validation_size', type=float, default=0.2, help='Validation size') 

    parser.add_argument('--key_cols_to_group_when_splitting', type=str,
                        help='columns for splitter') 
    parser.add_argument('--scoring', type=str,
                        help='loss scoring. Choose amongst binary_cross_entropy or surrogate_loss_tight')    
    parser.add_argument('--lr', type=float, help='learning rate')    
    parser.add_argument('--weight_decay', type=float, help='penalty for weights')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--n_splits', type=int, default=2)
    parser.add_argument('--lamb', type=float, default=1000)
    parser.add_argument('--warm_start', type=str, default='false')

    parser.add_argument('--merge_x_y', default=True,
                                type=lambda x: (str(x).lower() == 'true'), required=False)
    parser.add_argument('--initialization_gain', type=float, default=1.0)
    parser.add_argument('--incremental_min_precision', type=str, default='false')
    parser.add_argument('--tighten_bounds', type=str, default='false')

    args = parser.parse_args()
    
    torch.device('cuda')
    

    if args.incremental_min_precision=="true":
        args.incremental_min_precision=True
    else:
        args.incremental_min_precision=False
        
        
    if args.tighten_bounds=="true":
        args.tighten_bounds=True
    else:
        args.tighten_bounds=False

    # read the data dictionaries
    print('Reading train-test data...')
    
    # read the data dict JSONs and parse the feature and outcome columns
    x_data_dict_file, y_data_dict_file = args.data_dict_files.split(',')
    x_data_dict = load_data_dict_json(x_data_dict_file)
    y_data_dict = load_data_dict_json(y_data_dict_file)
    
    feature_cols = parse_feature_cols(x_data_dict)
    key_cols = parse_id_cols(x_data_dict)
    
    df_by_split = dict()
    for split_name, csv_files in [
            ('train', args.train_csv_files.split(',')),
            ('test', args.test_csv_files.split(','))]:
        cur_df = None
        for csv_file in csv_files:

            # TODO use json data dict to load specific columns as desired types
            more_df =  read_csv_with_float32_dtypes(csv_file)
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
    try:
        x_train = df_by_split['train'][feature_cols].values.astype(np.float32)
    except KeyError:
        feature_cols = [col.replace('_to_', '-') for col in feature_cols]
        x_train = df_by_split['train'][feature_cols].values.astype(np.float32)
    y_train = np.ravel(df_by_split['train'][outcome_col_name])
    
    x_test = df_by_split['test'][feature_cols].values.astype(np.float32)
    y_test = np.ravel(df_by_split['test'][outcome_col_name])        
    

    if args.valid_csv_files is None:
        # get the validation set
        splitter = Splitter(
            size=args.validation_size, random_state=41,
            n_splits=args.n_splits, cols_to_group=args.key_cols_to_group_when_splitting)
        # Assign training instances to splits by provided keys
        key_train = splitter.make_groups_from_df(df_by_split['train'][key_cols])


        # get the train and validation splits
        for ss, (tr_inds, va_inds) in enumerate(splitter.split(x_train, y_train, groups=key_train)):
            x_tr = x_train[tr_inds].copy()
            y_tr = y_train[tr_inds].copy()
            x_valid = x_train[va_inds]
            y_valid = y_train[va_inds]

        x_train = x_tr
        y_train = y_tr
        del(x_tr, y_tr, df_by_split)
    
    else:
        x_valid_csv, y_valid_csv = args.valid_csv_files.split(',')
        x_valid_df = read_csv_with_float32_dtypes(x_valid_csv)
        y_valid_df = read_csv_with_float32_dtypes(y_valid_csv)
        
        x_valid = x_valid_df[feature_cols].values.astype(np.float32)
        y_valid = np.ravel(y_valid_df[outcome_col_name])
        
        del(x_valid_df, y_valid_df, df_by_split)
        
    split_dict = {'N_train' : len(x_train),
                 'N_valid' : len(x_valid),
                 'N_test' : len(x_test),
                 'pos_frac_train' : y_train.sum()/len(y_train),
                 'pos_frac_valid' : y_valid.sum()/len(y_valid),
                 'pos_frac_test' : y_test.sum()/len(y_test),
                 'N_features' : x_tr.shape[1]
                 }
    
    print(split_dict)
    
    # get the feature columns
    class_weights_ = torch.tensor(np.asarray([1/((y_train==0).sum()), 1/((y_train==1).sum())]))
    
    
    # normalize data
    scaler = StandardScaler()

    scaler.fit(x_train)
    x_train_transformed = scaler.transform(x_train) 
    x_valid_transformed = scaler.transform(x_valid)
    x_test_transformed = scaler.transform(x_test) 
    
    del(x_train, x_valid, x_test)
    
    # store the fixed validation set as a skorch dataset
    valid_ds = Dataset(x_valid_transformed, y_valid) 
    
    # set random seed
    torch.manual_seed(args.seed)
    
    # set max_epochs 
    max_epochs=100

    
    # define callbacks 
    epoch_scoring_precision_train = EpochScoring('precision', lower_is_better=False, on_train=True,
                                                                    name='precision_train')
    
    epoch_scoring_precision_valid = EpochScoring('precision', lower_is_better=False, on_train=False,
                                                                        name='precision_valid')
    
    epoch_scoring_recall_train = EpochScoring('recall', lower_is_better=False, on_train=True,
                                                                    name='recall_train')
    
    epoch_scoring_recall_valid = EpochScoring('recall', lower_is_better=False, on_train=False,
                                                  name='recall_valid')
    
    epoch_scoring_auprc_train = EpochScoring('average_precision', lower_is_better=False, on_train=True,
                                                                    name='auprc_train')
    
    epoch_scoring_auprc_valid = EpochScoring('average_precision', lower_is_better=False, on_train=False,
                                                  name='auprc_valid')
    
    precision_early_stopping_cp = EarlyStopping(monitor='precision_train',
                                          patience=15, threshold=1e-10, threshold_mode='rel', 
                                          lower_is_better=False)
    
    loss_early_stopping_cp = EarlyStopping(monitor='valid_loss',
                                          patience=5, threshold=1e-10, threshold_mode='rel', 
                                          lower_is_better=True)
    
    auprc_early_stopping_cp = EarlyStopping(monitor='auprc_valid',
                                          patience=15, threshold=1e-10, threshold_mode='rel', 
                                          lower_is_better=False)
    

    epoch_scoring_surr_loss_train = EpochScoring(calc_surrogate_loss_skorch_callback, lower_is_better=True, on_train=True,
                                         name='surr_loss_train')

    epoch_scoring_surr_loss_valid = EpochScoring(calc_surrogate_loss_skorch_callback, lower_is_better=True, on_train=False,
                                                 name='surr_loss_valid')

    epoch_scoring_bce_loss_train = EpochScoring(calc_bce_loss_skorch_callback, lower_is_better=True, on_train=True,
                                                name='bce_loss_train')

    epoch_scoring_bce_loss_valid = EpochScoring(calc_bce_loss_skorch_callback, lower_is_better=True, on_train=False,
                                                name='bce_loss_valid')

    tpl_bound_train = EpochScoring(calc_tp_lower_bound_skorch_callback, lower_is_better=True, on_train=True,
                                   name='tpl_bound_train')

    tpl_bound_valid = EpochScoring(calc_tp_lower_bound_skorch_callback, lower_is_better=True, on_train=False,
                                   name='tpl_bound_valid')

    fpu_bound_train = EpochScoring(calc_fp_upper_bound_skorch_callback, lower_is_better=True, on_train=True,
                                   name='fpu_bound_train')

    fpu_bound_valid = EpochScoring(calc_fp_upper_bound_skorch_callback, lower_is_better=True, on_train=False,
                                   name='fpu_bound_valid')

    calc_tp_train = EpochScoring(calc_tp, lower_is_better=False, on_train=True, name='tp_train')
    
    calc_tp_valid = EpochScoring(calc_tp, lower_is_better=False, on_train=False, name='tp_valid')

    calc_fp_train = EpochScoring(calc_fp, lower_is_better=False, on_train=True, name='fp_train')
    
    calc_fp_valid = EpochScoring(calc_fp, lower_is_better=False, on_train=False, name='fp_valid')

#     cp = Checkpoint(monitor='precision_valid',
#                     f_history=os.path.join(args.output_dir,
#                                            args.output_filename_prefix+'.json'))



    cp = Checkpoint(monitor='surr_loss_valid_best',
                    f_history=os.path.join(args.output_dir,
                                           args.output_filename_prefix+'_surr_loss_valid_best.json'),
                    f_params=os.path.join(args.output_dir,
                                          args.output_filename_prefix+'_surr_loss_valid_best.pt')
                   )
    
    loss_best_cp = Checkpoint(monitor='valid_loss_best',
                f_history=os.path.join(args.output_dir,
                                       args.output_filename_prefix+'_loss_valid_best.json'),
                f_params=os.path.join(args.output_dir,
                                      args.output_filename_prefix+'_loss_valid_best.pt')
               )
    
    train_end_cp = TrainEndCheckpoint(dirname=args.output_dir,
                                          fn_prefix=args.output_filename_prefix)
    
    lr_scheduler = LRScheduler(policy=ReduceLROnPlateau, mode='max', factor=0.1, patience=3, min_lr=1e-04, verbose=True)
    
    
    
    

    fixed_precision = 0.7

    thr_list = [0.5]
    ## start training
    if args.scoring == 'cross_entropy_loss':
        logistic = SkorchLogisticRegression(n_features=len(feature_cols),
                                            l2_penalty_weights=args.weight_decay,
#                                             l2_penalty_bias=args.weight_decay,
                                            lr=args.lr,
                                            batch_size=args.batch_size, 
                                            max_epochs=max_epochs,
                                            train_split=predefined_split(valid_ds),
                                            loss_name=args.scoring,
                                            callbacks=[epoch_scoring_precision_train,
                                                       epoch_scoring_precision_valid,
                                                       epoch_scoring_recall_train,
                                                       epoch_scoring_recall_valid,
                                                       epoch_scoring_surr_loss_train,
                                                       epoch_scoring_surr_loss_valid,
                                                       epoch_scoring_bce_loss_train,
                                                       epoch_scoring_bce_loss_valid,
                                                       epoch_scoring_auprc_train,
                                                       epoch_scoring_auprc_valid,
                                                       tpl_bound_train,
                                                       tpl_bound_valid,
                                                       fpu_bound_train,
                                                       fpu_bound_valid,
                                                       calc_tp_train,
                                                       calc_tp_valid,
                                                       calc_fp_train,
                                                       calc_fp_valid,
                                                       auprc_early_stopping_cp,
                                                       loss_best_cp, 
                                                       train_end_cp],
                                           optimizer=torch.optim.Adam)
        print('Training Skorch Logistic Regression minimizing cross entropy loss...')

        logistic_clf = logistic.fit(x_train_transformed, y_train)
        
        print('Loading the parameters from checkpoint with best loss on validation set')
        logistic.load_params(checkpoint=loss_best_cp)
        
        # search multiple decision thresolds and pick the threshold that performs best on validation set
        print('Searching thresholds that maximize recall at fixed precision %.4f'%fixed_precision)
        
        y_valid_proba_vals = logistic_clf.predict_proba(x_valid_transformed)
        unique_probas = np.unique(y_valid_proba_vals)
        thr_grid = np.linspace(np.percentile(unique_probas,1), np.percentile(unique_probas, 99), 5000)

        
        precision_scores_G, recall_scores_G = [np.zeros(thr_grid.size), np.zeros(thr_grid.size)]
        y_train_pred_probas = logistic_clf.predict_proba(x_train_transformed)[:,1]
        for gg, thr in enumerate(thr_grid): 
#             logistic_clf.module_.linear_transform_layer.bias.data = torch.tensor(thr_grid[gg]).double()
            curr_thr_y_preds = y_valid_proba_vals[:,1]>=thr_grid[gg] 
            precision_scores_G[gg] = precision_score(y_valid, curr_thr_y_preds)
            recall_scores_G[gg] = recall_score(y_valid, curr_thr_y_preds)

        keep_inds = precision_scores_G>=fixed_precision
        if keep_inds.sum()>0:
            precision_scores_G = precision_scores_G[keep_inds]
            recall_scores_G = recall_scores_G[keep_inds]
            thr_grid = thr_grid[keep_inds]
            best_ind = np.argmax(recall_scores_G)
            best_thr = thr_grid[best_ind]
            thr_list.append(best_thr)

            thr_perf_df = pd.DataFrame(np.vstack([
                        thr_grid[np.newaxis,:],
                        precision_scores_G[np.newaxis,:],
                        recall_scores_G[np.newaxis,:]]).T,
                    columns=['thr', 'precision_score', 'recall_score'])
            print(thr_perf_df)
            print('Chosen threshold : '+str(best_thr))
        else:
            print('Could not find thresholds achieving fixed precision of %.2f. Evaluating with threshold 0.5'%fixed_precision)
        
    
    elif args.scoring == 'surrogate_loss_sigmoid_with_wbce_init':
        class_0_pen = 1.0 
        class_1_pen = 0.25
        
        lr_clf = LogisticRegression(solver='lbfgs', 
                                    max_iter=10000, 
                                    tol=1e-6, 
                                    class_weight={0 : class_0_pen, 1 : class_1_pen}, 
                                    verbose=1)
        
        # initialize with weight cross entropy
        print('Initializing with weighted cross entropy..')
        lr_clf.fit(x_train_transformed, y_train)
        
        # transfer weights
        print('Transferring weights and training with surrogate sigmoid loss..')
        logistic_clf = SkorchLogisticRegression(n_features=len(feature_cols), 
                                            l2_penalty_weights=args.weight_decay, 
                                            lr=1e-5, 
                                            batch_size=args.batch_size,  
                                            max_epochs=6,# temporary 
                                            train_split=predefined_split(valid_ds), 
                                            loss_name='surrogate_loss_tight', 
                                            min_precision=fixed_precision, 
                                            constraint_lambda=0.2, 
                                            callbacks=[epoch_scoring_precision_train, 
                                                       epoch_scoring_precision_valid, 
                                                       epoch_scoring_recall_train, 
                                                       epoch_scoring_recall_valid, 
                                                       epoch_scoring_surr_loss_train, 
                                                       epoch_scoring_surr_loss_valid, 
                                                       epoch_scoring_bce_loss_train, 
                                                       epoch_scoring_bce_loss_valid, 
                                                       tpl_bound_train, 
                                                       tpl_bound_valid, 
                                                       fpu_bound_train, 
                                                       fpu_bound_valid, 
                                                       calc_tp_train, 
                                                       calc_tp_valid, 
                                                       calc_fp_train, 
                                                       calc_fp_valid,
                                                       cp, train_end_cp], 
                                            optimizer=torch.optim.Adam) 
        
        logistic_clf.initialize()
        
        D = len(feature_cols) 
        
        # get sklearn weights and bias stacked into np array
        sklearn_lr_balanced_weights_ = np.hstack((lr_clf.coef_, lr_clf.intercept_[:,None]))[0]  
        
        
        
        logistic_clf.module_.linear_transform_layer.weight.data[0] = torch.tensor(sklearn_lr_balanced_weights_[:D],
                                                                              dtype=torch.float64)
        logistic_clf.module_.linear_transform_layer.bias.data = torch.tensor(sklearn_lr_balanced_weights_[-1], 
                                                                         dtype=torch.float64) 
        
        logistic_clf.partial_fit(x_train_transformed, y_train) 
        
    elif args.scoring == 'surrogate_loss_hinge_with_wbce_init':
        class_0_pen = 1.0 
        class_1_pen = 0.5
        
        lr_clf = LogisticRegression(solver='lbfgs', 
                                    max_iter=10000, 
                                    tol=1e-6, 
                                    class_weight={0 : class_0_pen, 1 : class_1_pen}, 
                                    verbose=1)
        
        # initialize with weight cross entropy
        print('Initializing with weighted cross entropy..')
        lr_clf.fit(x_train_transformed, y_train)
        
        # transfer weights
        print('Transferring weights and training with surrogate hinge loss..')
        logistic_clf = SkorchLogisticRegression(n_features=len(feature_cols), 
                                            l2_penalty_weights=args.weight_decay, 
                                            lr=1e-5, 
                                            batch_size=args.batch_size,  
                                            max_epochs=6,# temporary 
                                            train_split=predefined_split(valid_ds), 
                                            loss_name='surrogate_loss_loose', 
                                            min_precision=fixed_precision, 
                                            constraint_lambda=0.2, 
                                            callbacks=[epoch_scoring_precision_train, 
                                                       epoch_scoring_precision_valid, 
                                                       epoch_scoring_recall_train, 
                                                       epoch_scoring_recall_valid, 
                                                       epoch_scoring_surr_loss_train, 
                                                       epoch_scoring_surr_loss_valid, 
                                                       epoch_scoring_bce_loss_train, 
                                                       epoch_scoring_bce_loss_valid, 
                                                       tpl_bound_train, 
                                                       tpl_bound_valid, 
                                                       fpu_bound_train, 
                                                       fpu_bound_valid, 
                                                       calc_tp_train, 
                                                       calc_tp_valid, 
                                                       calc_fp_train, 
                                                       calc_fp_valid,
                                                       cp, train_end_cp], 
                                            optimizer=torch.optim.Adam) 
        
        logistic_clf.initialize()
        
        D = len(feature_cols) 
        
        # get sklearn weights and bias stacked into np array
        sklearn_lr_balanced_weights_ = np.hstack((lr_clf.coef_, lr_clf.intercept_[:,None]))[0]  
        
        
        
        logistic_clf.module_.linear_transform_layer.weight.data[0] = torch.tensor(sklearn_lr_balanced_weights_[:D],
                                                                              dtype=torch.float64)
        logistic_clf.module_.linear_transform_layer.bias.data = torch.tensor(sklearn_lr_balanced_weights_[-1], 
                                                                         dtype=torch.float64) 
        
        logistic_clf.partial_fit(x_train_transformed, y_train) 
    
    

    elif (args.scoring == 'surrogate_loss_tight')|(args.scoring == 'surrogate_loss_loose'):
        if args.warm_start == 'true':
            print('Warm starting by minimizing cross entropy loss first...')
            # warm start classifier and train with miminizing cross entropy loss first
            
            logistic_init = SkorchLogisticRegression(n_features=len(feature_cols),
                                                l2_penalty_weights=args.weight_decay,
    #                                             l2_penalty_bias=args.weight_decay,
                                                lr=0.01,
                                                batch_size=args.batch_size, 
                                                max_epochs=20,
                                                train_split=predefined_split(valid_ds),
                                                loss_name='cross_entropy_loss',
                                                callbacks=[epoch_scoring_precision_train,
                                                           epoch_scoring_precision_valid,
                                                           epoch_scoring_recall_train,
                                                           epoch_scoring_recall_valid,
                                                           epoch_scoring_surr_loss_train,
                                                           epoch_scoring_surr_loss_valid,
                                                           epoch_scoring_bce_loss_train,
                                                           epoch_scoring_bce_loss_valid,
                                                           epoch_scoring_auprc_train,
                                                           epoch_scoring_auprc_valid,
                                                           tpl_bound_train,
                                                           tpl_bound_valid,
                                                           fpu_bound_train,
                                                           fpu_bound_valid,
                                                           calc_tp_train,
                                                           calc_tp_valid,
                                                           calc_fp_train,
                                                           calc_fp_valid,
#                                                            loss_early_stopping_cp,
                                                           loss_best_cp, 
                                                           train_end_cp],
                                               optimizer=torch.optim.Adam,
#                                                optimizer__lr=0.01
                                                    )
            
            
            logistic_init_clf = logistic_init.fit(x_train_transformed, y_train)

            # transfer the weights and to a new instance of SkorchLogisticRegression and train minimizing surrogate loss
            print('Transferring weights and training to minimize surrogate loss...')

            D = len(feature_cols)
            logistic = SkorchLogisticRegression(n_features=len(feature_cols),
                                                l2_penalty_weights=args.weight_decay,
    #                                             l2_penalty_bias=args.weight_decay,
                                                lr=args.lr,
                                                batch_size=args.batch_size, 
                                                max_epochs=max_epochs,
                                                train_split=predefined_split(valid_ds),
                                                loss_name=args.scoring,
                                                min_precision=fixed_precision,
                                                constraint_lambda=args.lamb,
                                                callbacks=[epoch_scoring_precision_train,
                                                           epoch_scoring_precision_valid,
                                                           epoch_scoring_recall_train,
                                                           epoch_scoring_recall_valid,
                                                           epoch_scoring_surr_loss_train,
                                                           epoch_scoring_surr_loss_valid,
                                                           epoch_scoring_bce_loss_train,
                                                           epoch_scoring_bce_loss_valid,
                                                           tpl_bound_train,
                                                           tpl_bound_valid,
                                                           fpu_bound_train,
                                                           fpu_bound_valid,
                                                           calc_tp_train,
                                                           calc_tp_valid,
                                                           calc_fp_train,
                                                           calc_fp_valid,
    #                                                        lr_scheduler,
#                                                            early_stopping_cp,
                                                           cp, train_end_cp],
                                               optimizer=torch.optim.Adam,
#                                                optimizer__lr=args.lr
                                               )

            logistic.initialize()
                        
            # load the past history
            logistic.load_params(checkpoint=loss_best_cp)
            logistic.module_.linear_transform_layer.weight.data = logistic_init_clf.module_.linear_transform_layer.weight.data
            logistic.module_.linear_transform_layer.bias.data = logistic_init_clf.module_.linear_transform_layer.bias.data
            logistic_clf = logistic.partial_fit(x_train_transformed, y_train)
            
            #
            print('Loading the parameters from checkpoint with best surrogate loss on validation set')
            logistic_clf.load_params(checkpoint=cp)
            
        
        else:
            print('Training with surrogate loss with random initialization...')
            # Train with random initialization
            logistic = SkorchLogisticRegression(n_features=len(feature_cols),
                                                l2_penalty_weights=args.weight_decay,
                                                lr=args.lr,
                                                batch_size=args.batch_size, 
                                                max_epochs=max_epochs,
                                                train_split=predefined_split(valid_ds),
                                                loss_name=args.scoring,
                                                min_precision=fixed_precision,
                                                constraint_lambda=args.lamb,
                                                incremental_min_precision=args.incremental_min_precision,
                                                initialization_gain=args.initialization_gain,
                                                tighten_bounds=args.tighten_bounds,
                                                callbacks=[epoch_scoring_precision_train,
                                                           epoch_scoring_precision_valid,
                                                           epoch_scoring_recall_train,
                                                           epoch_scoring_recall_valid,
                                                           epoch_scoring_surr_loss_train,
                                                           epoch_scoring_surr_loss_valid,
                                                           epoch_scoring_bce_loss_train,
                                                           epoch_scoring_bce_loss_valid,
                                                           tpl_bound_train,
                                                           tpl_bound_valid,
                                                           fpu_bound_train,
                                                           fpu_bound_valid,
                                                           calc_tp_train,
                                                           calc_tp_valid,
                                                           calc_fp_train,
                                                           calc_fp_valid,
#                                                            early_stopping_cp,
                                                           cp, train_end_cp],
                                               optimizer=torch.optim.Adam)
            
            logistic_clf = logistic.fit(x_train_transformed, y_train)
            print('Loading the parameters from checkpoint with best surrogate loss on validation set')
            logistic_clf.load_params(checkpoint=cp)

        
    # save the scaler
    pickle_file = os.path.join(args.output_dir, 'scaler.pkl')
    dump(scaler, open(pickle_file, 'wb'))
    
    
    # print precision on train and validation
    f_out_name = os.path.join(args.output_dir, args.output_filename_prefix+'.txt')
    f_out = open(f_out_name, 'w')
    for thr in thr_list : 
        y_train_pred_probas = logistic_clf.predict_proba(x_train_transformed)[:,1]
        y_train_pred = y_train_pred_probas>=thr
        precision_train = precision_score(y_train, y_train_pred)
        recall_train = recall_score(y_train, y_train_pred)
        auprc_train = average_precision_score(y_train, y_train_pred_probas)

        
        y_valid_pred_probas = logistic_clf.predict_proba(x_valid_transformed)[:,1]
        y_valid_pred = y_valid_pred_probas>=thr
        precision_valid = precision_score(y_valid, y_valid_pred)
        recall_valid = recall_score(y_valid, y_valid_pred)   
        auprc_valid = average_precision_score(y_valid, y_valid_pred_probas)

        
        y_test_pred_probas = logistic_clf.predict_proba(x_test_transformed)[:,1]
        y_test_pred = y_test_pred_probas>=thr
        precision_test = precision_score(y_test, y_test_pred)
        recall_test = recall_score(y_test, y_test_pred)
        auprc_test = average_precision_score(y_test, y_test_pred_probas)


        # compute the TP's, FP's, TN's and FN's
#         y_train = y_train[:, np.newaxis] 
#         y_valid = y_valid[:, np.newaxis] 
#         y_test = y_test[:, np.newaxis]
        

        TP_train = np.sum(np.logical_and(y_train == 1, y_train_pred == 1))
        FP_train = np.sum(np.logical_and(y_train == 0, y_train_pred == 1))
        TN_train = np.sum(np.logical_and(y_train == 0, y_train_pred == 0))
        FN_train = np.sum(np.logical_and(y_train == 1, y_train_pred == 0))
        
        TP_valid = np.sum(np.logical_and(y_valid == 1, y_valid_pred == 1))
        FP_valid = np.sum(np.logical_and(y_valid == 0, y_valid_pred == 1))
        TN_valid = np.sum(np.logical_and(y_valid == 0, y_valid_pred == 0))
        FN_valid = np.sum(np.logical_and(y_valid == 1, y_valid_pred == 0))          
        
        TP_test = np.sum(np.logical_and(y_test == 1, y_test_pred == 1))
        FP_test = np.sum(np.logical_and(y_test == 0, y_test_pred == 1))
        TN_test = np.sum(np.logical_and(y_test == 0, y_test_pred == 0))
        FN_test = np.sum(np.logical_and(y_test == 1, y_test_pred == 0))    

        print_st_tr = "Training performance minimizing %s loss at threshold %.5f : | Precision %.3f | Recall %.3f | TP %5d | FP %5d | TN %5d | FN %5d"%(args.scoring, thr, precision_train, recall_train, TP_train, FP_train, TN_train, FN_train)
        print_st_va = "Validation performance minimizing %s loss at threshold %.5f : | Precision %.3f | Recall %.3f | TP %5d | FP %5d | TN %5d | FN %5d"%(args.scoring, thr, precision_valid, recall_valid, TP_valid, FP_valid, TN_valid, FN_valid)
        print_st_te = "Test performance minimizing %s loss at threshold %.5f :  | Precision %.3f | Recall %.3f | TP %5d | FP %5d | TN %5d | FN %5d"%(args.scoring, thr, precision_test, recall_test,TP_test, FP_test, TN_test, FN_test) 


        print(print_st_tr)
        print(print_st_va)
        print(print_st_te)
        f_out.write(print_st_tr + '\n' + print_st_va + '\n' + print_st_te)
    f_out.close()
    
    
    perf_dict = {'precision_train':precision_train,
                'recall_train':recall_train,
                'TP_train':TP_train,
                'FP_train':FP_train,
                'TN_train':TN_train,
                'FN_train':FN_train,
                'N_train':len(x_train),
                'auprc_train':auprc_train,
                'precision_valid':precision_valid,
                'recall_valid':recall_valid,
                'TP_valid':TP_valid,
                'FP_valid':FP_valid,
                'TN_valid':TN_valid,
                'FN_valid':FN_valid,
                'N_valid':len(x_valid),
                'auprc_valid':auprc_valid,
                'precision_test':precision_test,
                'recall_test':recall_test,
                'TP_test':TP_test,
                'FP_test':FP_test,
                'TN_test':TN_test,
                'FN_test':FN_test,
                'threshold':threshold,
                'N_test':len(x_test),
                'auprc_test':auprc_test}

    perf_df = pd.DataFrame([perf_dict])
    perf_csv = os.path.join(args.output_dir, args.output_filename_prefix+'_perf.csv')
    print('Saving performance on train and test set to %s'%(perf_csv))
    perf_df.to_csv(perf_csv, index=False)
    
    
    # save to onnx
    
    # initialize a dummy sklearn logistic regression clf 
    sklearn_lr_clf = LogisticRegression() 
    
    # do some dummy fitting to initialize the classifier to access the coefficients
    sklearn_lr_clf.fit(x_train_transformed[:500], y_train[:500])
    
    # save the skorch model weights to a sklearn Logistic regression object
    sklearn_lr_clf.coef_ = logistic_clf.module_.linear_transform_layer.weight.detach().numpy()
    sklearn_lr_clf.intercept_ = logistic_clf.module_.linear_transform_layer.bias.detach().numpy()
    
    
    prediction_pipeline = Pipeline(steps=[('standardize', scaler), ('classifier', sklearn_lr_clf)])
    
    # save as onnx
    initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
    model_onnx = convert_sklearn(prediction_pipeline, initial_types=initial_type)
    
    onx_file = os.path.join(args.output_dir, args.output_filename_prefix+'.onnx')
    with open(onx_file, "wb") as f:
        f.write(model_onnx.SerializeToString())
    