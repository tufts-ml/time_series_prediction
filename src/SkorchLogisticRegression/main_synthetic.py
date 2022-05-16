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
from feature_transformation import parse_feature_cols, parse_output_cols, parse_id_cols
from utils import load_data_dict_json
from pickle import dump
from torch.optim.lr_scheduler import ReduceLROnPlateau
from split_dataset import Splitter
from skorch.dataset import Dataset
from skorch.helper import predefined_split

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

def calc_g_theta_skorch_callback(net, X, y):
    if isinstance(X, torch.utils.data.dataset.Subset):
        indices = X.indices
        X_NF = torch.DoubleTensor(X.dataset.X[indices])
    elif isinstance(X, skorch.dataset.Dataset):
        X_NF = torch.DoubleTensor(X.X)
    y_ = torch.DoubleTensor(y)
    y_est_logits_ = net.module_.linear_transform_layer.forward(X_NF)[:,0]
    fp_upper_bound, tp_lower_bound = net.calc_fp_tp_bounds_better(y_, y_est_logits_)
    
    alpha=net.min_precision
    frac_alpha = alpha/(1-alpha)
    g_theta = -tp_lower_bound + frac_alpha*fp_upper_bound
    
    return float(g_theta.detach().numpy())

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
    parser.add_argument('--valid_csv_files', type=str, required=True,
                        help='csv files for validation')
    parser.add_argument('--test_csv_files', type=str, required=True,
                        help='csv files for test')
    parser.add_argument('--outcome_col_name', type=str, required=True,
                        help='outcome column name')  
    parser.add_argument('--output_dir', type=str, required=True,
                        help='save dir of trained classifier and associated performance metrics') 
    parser.add_argument('--output_filename_prefix', type=str, required=True,
                        help='save dir of trained classifier and associated performance metrics') 
    parser.add_argument('--scoring', type=str,
                        help='loss scoring. Choose amongst binary_cross_entropy or surrogate_loss_tight')    
    parser.add_argument('--lr', type=float, help='learning rate')    
    parser.add_argument('--weight_decay', type=float, help='penalty for weights')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--n_splits', type=int, default=2)
    parser.add_argument('--lamb', type=float, default=1000)
    parser.add_argument('--warm_start', type=str, default='true')
    parser.add_argument('--incremental_min_precision', type=str, default='true')
    parser.add_argument('--initialization_gain', type=float, default=1.0)
    args = parser.parse_args()

    if args.incremental_min_precision=="true":
        args.incremental_min_precision=True
    else:
        args.incremental_min_precision=False
    
    # read the data dictionaries
    print('Reading train-test data...')
    
     
    x_train_csv, y_train_csv = args.train_csv_files.split(',')
    x_valid_csv, y_valid_csv = args.valid_csv_files.split(',')
    x_test_csv, y_test_csv = args.test_csv_files.split(',')
    outcome_col_name = args.outcome_col_name
    
    # Prepare data for classification
    x_train = pd.read_csv(x_train_csv).values.astype(np.float32)[:, :2]
    y_train = pd.read_csv(y_train_csv)[outcome_col_name].values
    
    x_valid = pd.read_csv(x_valid_csv).values.astype(np.float32)[:, :2]
    y_valid = pd.read_csv(y_valid_csv)[outcome_col_name].values
    
    x_test = pd.read_csv(x_test_csv).values.astype(np.float32)[:, :2]
    y_test = pd.read_csv(y_test_csv)[outcome_col_name].values    
    
    # normalize data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_transformed = scaler.transform(x_train) 
    x_valid_transformed = scaler.transform(x_valid)
    x_test_transformed = scaler.transform(x_test)
    
    
    del(x_train, x_valid, x_test)
    
#     from IPython import embed; embed()
    # store fixed validaiton set into dataset object object
    valid_ds = Dataset(x_valid_transformed, y_valid) 
    
    # set random seed
    torch.manual_seed(args.seed)
    
    # set max_epochs 
    max_epochs=200
    
    # define callbacks 
    epoch_scoring_precision_train = EpochScoring('precision', lower_is_better=False, on_train=True,
                                                                    name='precision_train')
    
    epoch_scoring_precision_valid = EpochScoring('precision', lower_is_better=False, on_train=False,
                                                                        name='precision_valid')
    
    epoch_scoring_recall_train = EpochScoring('recall', lower_is_better=False, on_train=True,
                                                                    name='recall_train')
    
    epoch_scoring_recall_valid = EpochScoring('recall', lower_is_better=False, on_train=False,
                                                  name='recall_valid')
    
    early_stopping_cp = EarlyStopping(monitor='precision_train',
                                          patience=15, threshold=1e-10, threshold_mode='rel', 
                                          lower_is_better=False)
    
    loss_early_stopping_cp = EarlyStopping(monitor='train_loss',
                                          patience=15, threshold=1e-10, threshold_mode='rel', 
                                          lower_is_better=True)
    
    epoch_scoring_surr_loss_train = EpochScoring(calc_surrogate_loss_skorch_callback, lower_is_better=True, on_train=True,
                                         name='surr_loss_train')

    epoch_scoring_surr_loss_valid = EpochScoring(calc_surrogate_loss_skorch_callback, lower_is_better=True, on_train=False,
                                                 name='surr_loss_valid')

    epoch_scoring_bce_loss_train = EpochScoring(calc_bce_loss_skorch_callback, lower_is_better=True, on_train=True,
                                                name='bce_loss_train')

    epoch_scoring_bce_loss_valid = EpochScoring(calc_bce_loss_skorch_callback, lower_is_better=True, on_train=False,
                                                name='bce_loss_valid')
    
    epoch_scoring_g_theta_train = EpochScoring(calc_g_theta_skorch_callback, lower_is_better=True, on_train=True,
                                                name='g_theta_train')
    
    epoch_scoring_g_theta_valid = EpochScoring(calc_g_theta_skorch_callback, lower_is_better=True, on_train=False,
                                                name='g_theta_valid')

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

    cp = Checkpoint(monitor='precision_train',
                    f_history=os.path.join(args.output_dir,
                                           args.output_filename_prefix+'.json'))
    
    train_end_cp = TrainEndCheckpoint(dirname=args.output_dir,
                                          fn_prefix=args.output_filename_prefix)
    
    lr_scheduler = LRScheduler(policy=ReduceLROnPlateau, mode='max', factor=0.1, patience=3, min_lr=1e-04, verbose=True)
    
    
    callbacks_dict = {'surrogate_loss_tight' : [epoch_scoring_precision_train,
                                                epoch_scoring_precision_valid,
                                                epoch_scoring_recall_train,
                                                epoch_scoring_recall_valid,
                                                epoch_scoring_surr_loss_train,
                                                epoch_scoring_surr_loss_valid,
                                                epoch_scoring_bce_loss_train,
                                                epoch_scoring_bce_loss_valid,
                                                epoch_scoring_g_theta_train,
                                                epoch_scoring_g_theta_valid,
                                                tpl_bound_train,
                                                fpu_bound_train,
                                                calc_tp_train,
                                                calc_fp_train,
                                                early_stopping_cp,
                                                cp, train_end_cp],
                     'surrogate_loss_loose' : [epoch_scoring_precision_train,
                                                epoch_scoring_precision_valid,
                                                epoch_scoring_recall_train,
                                                epoch_scoring_recall_valid,
                                                epoch_scoring_surr_loss_train,
                                                epoch_scoring_surr_loss_valid,
                                                epoch_scoring_bce_loss_train,
                                                epoch_scoring_bce_loss_valid,
                                                epoch_scoring_g_theta_train,
                                                epoch_scoring_g_theta_valid,
                                                tpl_bound_train,
                                                fpu_bound_train,
                                                calc_tp_train,
                                                calc_fp_train,
                                                early_stopping_cp,
                                                cp, train_end_cp],
                      'custom' : [epoch_scoring_precision_train,
                                                epoch_scoring_precision_valid,
                                                epoch_scoring_recall_train,
                                                epoch_scoring_recall_valid,
                                                epoch_scoring_surr_loss_train,
                                                epoch_scoring_surr_loss_valid,
                                                epoch_scoring_bce_loss_train,
                                                epoch_scoring_bce_loss_valid,
                                                epoch_scoring_g_theta_train,
                                                epoch_scoring_g_theta_valid,
                                                tpl_bound_train,
                                                fpu_bound_train,
                                                calc_tp_train,
                                                calc_fp_train,
                                                cp, train_end_cp]
                     }
    
    
    fixed_precision = 0.9
    thr_list = [0.5]
    ## start training
    if args.scoring == 'cross_entropy_loss':
        logistic = SkorchLogisticRegression(n_features=x_train_transformed.shape[1],
                                            l2_penalty_weights=args.weight_decay,
#                                             l2_penalty_bias=args.weight_decay,
                                            lr=args.lr,
                                            batch_size=args.batch_size, 
                                            max_epochs=max_epochs,
                                            train_split=predefined_split(valid_ds),#skorch.dataset.CVSplit(cv=100),
                                            loss_name=args.scoring,
                                            callbacks=[epoch_scoring_precision_train,
                                                       epoch_scoring_precision_valid,
                                                       epoch_scoring_recall_train,
                                                       epoch_scoring_recall_valid,
                                                       epoch_scoring_surr_loss_train,
                                                       epoch_scoring_surr_loss_valid,
                                                       epoch_scoring_bce_loss_train,
                                                       epoch_scoring_bce_loss_valid,
                                                       tpl_bound_train,
#                                                        tpl_bound_valid,
                                                       fpu_bound_train,
#                                                        fpu_bound_valid,
                                                       calc_tp_train,
#                                                        calc_tp_valid,
                                                       calc_fp_train,
#                                                        calc_fp_valid,
                                                       loss_early_stopping_cp,
                                                       cp, train_end_cp],
                                           optimizer=torch.optim.Adam)
        print('Training Skorch Logistic Regression minimizing cross entropy loss...')
        logistic_clf = logistic.fit(x_train_transformed, y_train)
        
        # search multiple decision thresolds and pick the threshold that performs best on validation set
        print('Searching thresholds that maximize recall at fixed precision %.4f'%fixed_precision)
        y_train_proba_vals = logistic_clf.predict_proba(x_train_transformed)
        y_valid_proba_vals = logistic_clf.predict_proba(x_valid_transformed)
        unique_probas = np.unique(y_train_proba_vals)
        thr_grid = np.linspace(np.percentile(unique_probas,1), np.percentile(unique_probas, 99), 100)
        
        # compute precision recall across folds
        precision_score_G = np.zeros(thr_grid.size)
        recall_score_G = np.zeros(thr_grid.size)
        
        for gg, thr in enumerate(thr_grid): 
#             logistic_clf.module_.linear_transform_layer.bias.data = torch.tensor(thr_grid[gg]).double()
            curr_thr_y_preds = y_valid_proba_vals[:,1]>=thr_grid[gg] 
            precision_score_G[gg] = precision_score(y_valid, curr_thr_y_preds)
            recall_score_G[gg] = recall_score(y_valid, curr_thr_y_preds) 
            
        
        keep_inds = precision_score_G>=fixed_precision
        if keep_inds.sum()>0:
            precision_score_G = precision_score_G[keep_inds]
            recall_score_G = recall_score_G[keep_inds]
            thr_grid = thr_grid[keep_inds]
            best_ind = np.argmax(recall_score_G)
            best_thr = thr_grid[best_ind]
#             precision_train = precision_score_G[best_ind]
#             recall_train = recall_score_G[best_ind]
            thr_list.append(best_thr)

            thr_perf_df = pd.DataFrame(np.vstack([
                        thr_grid[np.newaxis,:],
                        precision_score_G[np.newaxis,:],
                        recall_score_G[np.newaxis,:]]).T,
                    columns=['thr', 'precision_score', 'recall_score'])
            print(thr_perf_df)
            print('Chosen threshold : '+str(best_thr))
        else:
            print('Could not find thresholds achieving fixed precision of %.2f. Evaluating with threshold 0.5'%fixed_precision)
            
            
    elif args.scoring=='custom':
        print('Training with surrogate loss with custom initialization...')
        # Train with random initialization
        D = x_train_transformed.shape[1]
        logistic_init = SkorchLogisticRegression(n_features=D,
                                            l2_penalty_weights=args.weight_decay,
                                            lr=args.lr,
                                            batch_size=args.batch_size, 
                                            max_epochs=max_epochs,
                                            train_split=predefined_split(valid_ds),#skorch.dataset.CVSplit(cv=splitter),
                                            loss_name='surrogate_loss_tight',
                                            min_precision=fixed_precision,
                                            constraint_lambda=args.lamb,
                                            incremental_min_precision=args.incremental_min_precision,
                                            initialization_gain=args.initialization_gain,
                                            callbacks=callbacks_dict[args.scoring],
                                           optimizer=torch.optim.Adam)

        
#         logistic_init.initialize()
#         weights_arr = np.zeros((1, D))
#         weights_arr[:, :2] = np.array([0.02799723, -0.58969025])
#         logistic_init.module_.linear_transform_layer.weight.data = torch.tensor(weights_arr)
#         logistic_init.module_.linear_transform_layer.bias.data[0] = -1.405
        from IPython import embed; embed()
        logistic_clf = logistic_init.partial_fit(x_train_transformed, y_train)   
        
        
    elif (args.scoring == 'surrogate_loss_tight')|(args.scoring == 'surrogate_loss_loose'):
        if args.warm_start == 'true':
            print('Warm starting by minimizing cross entropy loss first...')
            # warm start classifier and train with miminizing cross entropy loss first
            logistic_init = SkorchLogisticRegression(n_features=x_train_transformed.shape[1],
                                                l2_penalty_weights=args.weight_decay,
    #                                             l2_penalty_bias=args.weight_decay,
                                                lr=args.lr,
                                                batch_size=args.batch_size, 
                                                max_epochs=max_epochs,
                                                train_split=predefined_split(valid_ds),#skorch.dataset.CVSplit(cv=splitter),
                                                loss_name='cross_entropy_loss',
                                                callbacks=[epoch_scoring_precision_train,
                                                           epoch_scoring_precision_valid,
                                                           epoch_scoring_recall_train,
                                                           epoch_scoring_recall_valid,
                                                           epoch_scoring_surr_loss_train,
                                                           epoch_scoring_surr_loss_valid,
                                                           epoch_scoring_bce_loss_train,
                                                           epoch_scoring_bce_loss_valid,
                                                           tpl_bound_train,
#                                                            tpl_bound_valid,
                                                           fpu_bound_train,
#                                                            fpu_bound_valid,
                                                           calc_tp_train,
#                                                            calc_tp_valid,
                                                           calc_fp_train,
#                                                            calc_fp_valid,
                                                           early_stopping_cp,
                                                           cp, train_end_cp],
                                               optimizer=torch.optim.Adam)

            logistic_init_clf = logistic_init.fit(x_train_transformed, y_train)

            # transfer the weights and to a new instance of SkorchLogisticRegression and train minimizing surrogate loss
            print('Transferring weights and training to minimize surrogate loss...')


            D = x_train_transformed.shape[1]
            logistic = SkorchLogisticRegression(n_features=x_train_transformed.shape[1],
                                                l2_penalty_weights=args.weight_decay,
    #                                             l2_penalty_bias=args.weight_decay,
                                                lr=args.lr,
                                                batch_size=args.batch_size, 
                                                max_epochs=max_epochs,
                                                train_split=predefined_split(valid_ds),#skorch.dataset.CVSplit(cv=splitter),
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
#                                                            tpl_bound_valid,
                                                           fpu_bound_train,
#                                                            fpu_bound_valid,
                                                           calc_tp_train,
#                                                            calc_tp_valid,
                                                           calc_fp_train,
#                                                            calc_fp_valid,
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
        
        else:
            print('Training with surrogate loss with random initialization...')
            # Train with random initialization
            D = x_train_transformed.shape[1]
            logistic_init = SkorchLogisticRegression(n_features=D,
                                                l2_penalty_weights=args.weight_decay,
                                                lr=args.lr,
                                                batch_size=args.batch_size, 
                                                max_epochs=max_epochs,
                                                train_split=predefined_split(valid_ds),#skorch.dataset.CVSplit(cv=splitter),
                                                loss_name=args.scoring,
                                                min_precision=fixed_precision,
                                                constraint_lambda=args.lamb,
                                                incremental_min_precision=args.incremental_min_precision,
                                                initialization_gain=args.initialization_gain,
                                                callbacks=callbacks_dict[args.scoring],
                                               optimizer=torch.optim.Adam)
            
            
            
            
            logistic_clf = logistic_init.fit(x_train_transformed, y_train)
            '''
            print('Transferring weights to match fixed precision...')
            logistic_final = SkorchLogisticRegression(n_features=D,
                                                l2_penalty_weights=args.weight_decay,
                                                lr=args.lr,
                                                batch_size=args.batch_size, 
                                                max_epochs=max_epochs,
                                                train_split=None,#skorch.dataset.CVSplit(cv=splitter),
                                                loss_name=args.scoring,
                                                min_precision=fixed_precision,
                                                constraint_lambda=args.lamb,
                                                callbacks=[epoch_scoring_precision_train,
                                                           epoch_scoring_recall_train,
                                                           epoch_scoring_surr_loss_train,
                                                           epoch_scoring_bce_loss_train,
                                                           tpl_bound_train,
                                                           fpu_bound_train,
                                                           calc_tp_train,
                                                           calc_fp_train,
                                                           early_stopping_cp,
                                                           cp, train_end_cp],
                                               optimizer=torch.optim.Adam)           
            
            
            logistic_final.initialize()
            # load the past history
            logistic_final.load_params(checkpoint=cp)

            # load the weights
            logistic_final.module_.linear_transform_layer.weight.data = logistic_init_clf.module_.linear_transform_layer.weight.data
            logistic_final.module_.linear_transform_layer.bias.data = logistic_init_clf.module_.linear_transform_layer.bias.data
            logistic_final_clf = logistic_final.partial_fit(x_train_transformed, y_train)
            '''
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
        
        y_valid_pred_probas = logistic_clf.predict_proba(x_valid_transformed)[:,1]
        y_valid_pred = y_valid_pred_probas>=thr
        precision_valid = precision_score(y_valid, y_valid_pred)
        recall_valid = recall_score(y_valid, y_valid_pred)       
        
        y_test_pred_probas = logistic_clf.predict_proba(x_test_transformed)[:,1]
        y_test_pred = y_test_pred_probas>=thr
        precision_test = precision_score(y_test, y_test_pred)
        recall_test = recall_score(y_test, y_test_pred)

#         # compute the TP's, FP's, TN's and FN's
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
    
    
    perf_dict = {'N_train':len(x_train_transformed),
                'precision_train':precision_train,
                'recall_train':recall_train,
                'TP_train':TP_train,
                'FP_train':FP_train,
                'TN_train':TN_train,
                'FN_train':FN_train,
                'precision_valid':precision_valid,
                'recall_valid':recall_valid,
                'TP_valid':TP_valid,
                'FP_valid':FP_valid,
                'TN_valid':TN_valid,
                'FN_valid':FN_valid,
                'N_valid':len(x_valid_transformed),
                'precision_test':precision_test,
                'recall_test':recall_test,
                'TP_test':TP_test,
                'FP_test':FP_test,
                'TN_test':TN_test,
                'FN_test':FN_test,
                'N_test':len(x_test_transformed)}
    perf_df = pd.DataFrame([perf_dict])
    perf_csv = os.path.join(args.output_dir, args.output_filename_prefix+'_perf.csv')
    print('Saving performance on train and test set to %s'%(perf_csv))
    perf_df.to_csv(perf_csv, index=False)
