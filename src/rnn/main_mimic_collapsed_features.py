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

from dataset_loader import TidySequentialDataCSVLoader
from RNNBinaryClassifier import RNNBinaryClassifier

from joblib import dump
import itertools
import warnings
warnings.filterwarnings("ignore")

from skorch.callbacks import (Callback, LoadInitState, 
                              TrainEndCheckpoint, Checkpoint, 
                              EpochScoring, EarlyStopping, LRScheduler)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from SkorchLogisticRegression import SkorchLogisticRegression
from skorch.utils import noop
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau

           
def main():
    parser = argparse.ArgumentParser(description='PyTorch RNN with variable-length numeric sequences wrapper')
    
    parser.add_argument('--train_vitals_csv', type=str,
                        help='Location of vitals data for training')
    parser.add_argument('--test_vitals_csv', type=str,
                        help='Location of vitals data for testing')
    parser.add_argument('--metadata_csv', type=str,
                        help='Location of metadata for testing and training')
    parser.add_argument('--data_dict', type=str)
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--report_dir', type=str, default='html',
                        help='dir in which to save results report')
    parser.add_argument('--simulated_data_dir', type=str, default='simulated_data/2-state/',
                        help='dir in which to simulated data is saved')    
    parser.add_argument('--is_data_simulated', type=bool, default=False,
                        help='boolean to check if data is simulated or from mimic')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cpu'
    
    # extract data
    if not(args.is_data_simulated):
        train_vitals = TidySequentialDataCSVLoader(
            per_tstep_csv_path=args.train_vitals_csv,
            per_seq_csv_path=args.metadata_csv,
            idx_col_names=['subject_id', 'episode_id'],
            x_col_names='__all__',
            y_col_name='inhospital_mortality',
            y_label_type='per_tstep')

        test_vitals = TidySequentialDataCSVLoader(
            per_tstep_csv_path=args.test_vitals_csv,
            per_seq_csv_path=args.metadata_csv,
            idx_col_names=['subject_id', 'episode_id'],
            x_col_names='__all__',
            y_col_name='inhospital_mortality',
            y_label_type='per_tstep')
        
        X_train_with_time_appended, y_train = train_vitals.get_batch_data(batch_id=0)
        X_test_with_time_appended, y_test = test_vitals.get_batch_data(batch_id=0)
        _,T,F = X_train_with_time_appended.shape
        
        if T>1:
            X_train = X_train_with_time_appended[:,:,1:]# removing hours column
            X_test = X_test_with_time_appended[:,:,1:]# removing hours column
        else:# account for collapsed features across time
            X_train = X_train_with_time_appended
            X_test = X_test_with_time_appended
    
    
    # set class weights as 1/(number of samples in class) for each class to handle class imbalance
    class_weights = torch.tensor([1/(y_train==0).sum(),
                                  1/(y_train==1).sum()]).double()
    
    
    # define a auc scorer function and pass it as callback of skorch to track training and validation AUROC
    roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                                 needs_threshold=True)
    
    # scale features
    X_train = standard_scaler_3d(X_train)
    X_test = standard_scaler_3d(X_test)  
    
    # Define parameter grid
    params = {'lr':[0.0001, 0.0005, 0.001, 0.005, 0.01], 'optimizer__weight_decay':[0.0001, 0.001, 0.01, 0.1, 1, 10]
            } 
#---------------------------------------------------------------------#
# LSTM with gridsearchcv
#---------------------------------------------------------------------#
    print('-------------------------------------------------------------------')
    print('Running LSTM converted to logistic regression on collapsed Features')
    model_name='logreg_hist'
    save_cv_results = SaveCVResults(dirname=args.report_dir, f_history=model_name+'.json')
    rnn = RNNBinaryClassifier( 
             max_epochs=args.epochs, 
             batch_size=-1, 
             device=device, 
             callbacks=[ 
             skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=True, name='aucroc_score_train'), 
             skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=False, name='aucroc_score_valid'), 
             skorch.callbacks.EarlyStopping(monitor='aucroc_score_valid', patience=5, threshold=1e-10, threshold_mode='rel', 
                                            lower_is_better=False),
             save_cv_results,
             ],
             criterion=torch.nn.NLLLoss, 
             criterion__weight=class_weights, 
             train_split=skorch.dataset.CVSplit(0.2), 
             module__rnn_type='LSTM', 
             module__n_layers=1,
             module__n_hiddens=X_train.shape[-1],
             module__n_inputs=X_train.shape[-1], 
             module__convert_to_log_reg=True,
             optimizer=torch.optim.Adam) 
    
    
    gs = GridSearchCV(rnn, params, scoring=roc_auc_scorer, refit=True, cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=14232)) 
    lr_cv = gs.fit(X_train, y_train)
    y_pred_proba = lr_cv.best_estimator_.predict_proba(X_train)
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    auroc_train_final = roc_auc_score(y_train, y_pred_proba_pos)
    print('AUROC with logistic regression (Train) : %.3f'%auroc_train_final)
    
    y_pred_proba = lr_cv.best_estimator_.predict_proba(X_test)
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    auroc_test_final = roc_auc_score(y_test, y_pred_proba_pos)
    print('AUROC with logistic regression (Test) : %.3f'%auroc_test_final)
    
    # get the loss plots for logistic regression
    plot_training_history(model_name='logreg_hist', model_alias = 'Logistic Regression',report_dir=args.report_dir, params=params, auroc_train_final = auroc_train_final, auroc_test_final=auroc_test_final)    
    
    # LSTM
    print('-------------------------------------------------------------------')
    print('Running LSTM on Collapsed Features')
    model_name='lstm_hist'
    save_cv_results = SaveCVResults(dirname=args.report_dir, f_history=model_name+'.json')
    rnn = RNNBinaryClassifier(  
              max_epochs=args.epochs,  
              batch_size=-1,  
              device=device,  
              callbacks=[
              save_cv_results,
              EpochScoring('roc_auc', lower_is_better=False, on_train=True, name='aucroc_score_train'),  
              EpochScoring('roc_auc', lower_is_better=False, on_train=False, name='aucroc_score_valid'),  
              EarlyStopping(monitor='aucroc_score_valid', patience=5, threshold=0.002, threshold_mode='rel',  
                                             lower_is_better=False)
              ],  
              criterion=torch.nn.CrossEntropyLoss, 
              criterion__weight=class_weights,  
              train_split=skorch.dataset.CVSplit(0.2), 
              module__rnn_type='LSTM',  
              module__n_layers=1, 
              module__n_hiddens=X_train.shape[-1], 
              module__n_inputs=X_train.shape[-1],  
              module__convert_to_log_reg=False, 
              optimizer=torch.optim.Adam)                    
    gs = GridSearchCV(rnn, params, scoring='roc_auc', cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=14232),
                    ) 
    rnn_cv = gs.fit(X_train, y_train)
    y_pred_proba = rnn_cv.predict_proba(X_train)
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    auroc_train_final = roc_auc_score(y_train, y_pred_proba_pos)
    print('AUROC with LSTM (Train) : %.2f'%auroc_train_final)
    
    y_pred_proba = rnn_cv.predict_proba(X_test)
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    auroc_test_final = roc_auc_score(y_test, y_pred_proba_pos)
    print('AUROC with LSTM (Test) : %.2f'%auroc_test_final)
    
    
    # get the loss plots for LSTM
    plot_training_history(model_name='lstm_hist', model_alias = 'LSTM', report_dir=args.report_dir,
                         params=params, auroc_train_final = auroc_train_final, auroc_test_final=auroc_test_final)
    
    
    

def standard_scaler_3d(X):
    # input : X (N, T, F)
    # ouput : scaled_X (N, T, F)
    N, T, F = X.shape
    if T==1:
        scalers = {}
        for i in range(X.shape[1]):
            scalers[i] = StandardScaler()
            X[:, i, :] = scalers[i].fit_transform(X[:, i, :]) 
    else:
        # zscore across subjects and time points for each feature
        for i in range(F):
            mean_across_NT = X[:,:,i].mean()
            std_across_NT = X[:,:,i].std()
            
            X[:,:,i] = (X[:,:,i]-mean_across_NT)/std_across_NT
    return X
    
        
        
def convert_proba_to_binary(probabilites):
    return [0 if probs[0] > probs[1] else 1 for probs in probabilites]


# def get_loss_plots_from_training_history(train_history):
#     epochs = train_history[:,'epoch']
#     train_loss = train_history[:,'train_loss']
#     valid_loss = train_history[:,'valid_loss']
#     aucroc_score_train = train_history[:,'aucroc_score_train']
#     aucroc_score_valid = train_history[:,'aucroc_score_valid']
    
#     return epochs, train_loss, valid_loss, aucroc_score_train, aucroc_score_valid

def plot_training_history(model_name='logreg_hist', model_alias = 'Logistic Regression', report_dir=None, params=None, auroc_train_final = None, auroc_test_final=None):
    # get the loss plots for logistic regression
    model_hist_files = glob.glob(os.path.join(report_dir, '*'+model_name+'*'))
    param_lens = [len(v) for v in params.values()]
    n_rows = param_lens[0]
    n_cols = np.prod(param_lens[1:])
    
    # get all the permutations of parameters in grid
    params_list = [[list(params.keys())[j]+'='+str(i) for i in list(params.values())[j]] for j in range(len(params.keys()))] 
    params_grid = list(itertools.product(*params_list))
    
    #plot 
    fig, axs = plt.subplots(n_rows,n_cols, figsize=(15,15))
    fontsize=5
    i=0
    j=0
    end_reached=False
    for fname in model_hist_files:
        for params_comb in params_grid:
            if all(param in fname for param in params_comb):
                with open(fname) as f: 
                    model_hist = json.load(f) 
                epochs = [hist['epoch'] for hist in model_hist] 
                train_loss = [hist['train_loss'] for hist in model_hist] 
                valid_loss = [hist['valid_loss'] for hist in model_hist] 
                auroc_train = [hist['aucroc_score_train'] for hist in model_hist]
                auroc_valid = [hist['aucroc_score_valid'] for hist in model_hist]
                axs[i,j].plot(epochs, train_loss, 'r-.', label = 'Train Loss')
                axs[i,j].plot(epochs, valid_loss, 'b-.', label = 'Validation Loss')
                axs[i,j].plot(epochs, auroc_train, 'g-.', label = 'AUCROC score (Train)')
                axs[i,j].plot(epochs, auroc_valid, 'm-.', label = 'AUCROC score (Valid)')
#                     if i==len(params['lr'])-1:
                axs[i,j].set_xlabel('Epochs', fontsize=fontsize+2)
                axs[i,j].set_ylabel('Metric', fontsize=fontsize+2)
                axs[i,j].set_ylim([0.4, 0.9])
#                     if i==0:
                axs[i,j].legend(fontsize=fontsize, loc='upper left')
                axs[i,j].set_title(params_comb, fontsize=fontsize+2)
                if (i == n_rows-1) and (j == n_cols-1):
                    end_reached=True
                if (i < n_rows) and (j < n_cols-1):
                    if (i+1)%n_rows == 0:
                        i=0
                        j=j+1 
                    else:
                        i=i+1
                elif (i < n_rows) and (j == n_cols-1):
                    if (i+1)%n_rows == 0:
                        i=0
                    else:
                        i=i+1
            if end_reached:
                break
    plt.suptitle(model_alias + ' collapsed features (AUROC Train : %.3f, AUROC Test : %.3f)'
                 %(auroc_train_final, auroc_test_final), fontsize=fontsize+2) 
#     plt.subplots_adjust(hspace=1)
    fig.savefig(model_name+'_collapsed_features.png')
    plt.close()
                
                
    
    
                         
#     fig, axs = plt.subplots(n_rows,n_cols)
#     fontsize=5
#     for k, param in enumerate(params.keys()):
#         for i,lrate in enumerate(params[param]):
#             for fname in model_hist_files: 
#                 param_str = param+'='+str(lrate)
#                 if  param_str in fname: 
#                     if k==0:
#                         j=0
#                     else:
#                         j+=1
#                     lr_fname = fname 
#                     with open(lr_fname) as f: 
#                         model_hist = json.load(f) 
#                     epochs = [hist['epoch'] for hist in model_hist] 
#                     train_loss = [hist['train_loss'] for hist in model_hist] 
#                     valid_loss = [hist['valid_loss'] for hist in model_hist] 
#                     auroc_train = [hist['aucroc_score_train'] for hist in model_hist]
#                     auroc_valid = [hist['aucroc_score_valid'] for hist in model_hist]
#                     axs[i,j].plot(epochs, train_loss, 'r-.', label = 'Train Loss')
#                     axs[i,j].plot(epochs, valid_loss, 'b-.', label = 'Validation Loss')
#                     axs[i,j].plot(epochs, auroc_train, 'g-.', label = 'AUCROC score (Train)')
#                     axs[i,j].plot(epochs, auroc_valid, 'm-.', label = 'AUCROC score (Valid)')
# #                     if i==len(params['lr'])-1:
#                     axs[i,j].set_xlabel('Epochs', fontsize=fontsize+2)
#                     axs[i,j].set_ylabel('Metric', fontsize=fontsize+2)
#                     axs[i,j].set_ylim([0.4, 0.9])
# #                     if i==0:
#                     axs[i,j].legend(fontsize=fontsize, loc='upper left')
#                     axs[i,j].set_title(param_str, fontsize=fontsize+2)
#     plt.suptitle(model_alias + ' collapsed features (AUROC Train : %.3f, AUROC Test : %.3f)'
#                  %(auroc_train_final, auroc_test_final), fontsize=fontsize+2)
# #     plt.subplots_adjust(hspace=1)
#     fig.savefig(model_name+'_collapsed_features.png')
#     plt.close()

# t



def get_paramater_gradient_l2_norm(net,**kwargs):
    parameters = [i for  _,i in net.module_.named_parameters()]
    total_norm = 0 
#     from IPython import embed; embed()
    for p in parameters: 
        if p.requires_grad==True:
            param_norm = p.grad.data.norm(2) 
            total_norm += param_norm.item() ** 2 
    total_norm = total_norm ** (1. / 2)
    return total_norm

def get_paramater_l2_norm(net,**kwargs):
    parameters = [i for  _,i in net.module_.named_parameters()]
    total_norm = 0 
    for p in parameters: 
        param_norm = p.norm(2) 
        total_norm += param_norm.item() ** 2 
    total_norm = total_norm ** (1. / 2)
    return total_norm

def get_paramater_gradient_inf_norm(net, **kwargs):
    parameters = [i for  _,i in net.module_.named_parameters()]
    total_norm = max(p.grad.data.abs().max() for p in parameters if p.grad==True)
    return total_norm


class LoadLRCheckpoint(LoadInitState):
    def __init__(self, lr_params):
        self.lr_params = lr_params
    
    def on_train_begin(self, net,
                       X=None, y=None, **kwargs):
        
        # set the parameters of the rnn as the lstm weights and biases
        net.module_=self.lr_params.module_
        # free all the weights and biases
        for _,j in net.module_.named_parameters(): 
            j.requires_grad=True 

class SaveCVResults(TrainEndCheckpoint):
    def __init__(
         self,
         f_params='params.pt',
         f_optimizer='optimizer.pt',
         f_history='history.json',
         f_pickle=None,
         fn_prefix='train_end_',
         dirname='',
         sink=noop
     ):
            self.f_params = f_params
            self.f_optimizer = f_optimizer
            self.f_history = f_history
            self.f_pickle = f_pickle
            self.fn_prefix = fn_prefix
            self.dirname = dirname
            self.sink=sink
    
#     def initialize(self):
#         self.checkpoint_ = Checkpoint(
#             monitor=None,
#             f_params=self.f_params,
#             f_optimizer=self.f_optimizer,
#             f_history=self.f_history,
#             f_pickle=self.f_pickle,
#             fn_prefix=self.fn_prefix,
#             dirname=self.dirname,
#             event_name=None,
#             sink=self.sink)
#         self.checkpoint_.initialize()
    
    
    def on_train_end(self, net, **kwargs):
        f_name = 'lr='+str(net.lr)+'-optimizer__weight_decay='+str(net.optimizer__weight_decay)+\
                                    '-module__n_hiddens='+str(net.module__n_hiddens)+ \
                                    '-module__n_layers='+str(net.module__n_layers)+ \
                                    '-batch_size='+str(net.batch_size)
        print('done with %s'%f_name)
        net.save_params(f_history=os.path.join(self.dirname, f_name+'-'+
                                               self.f_history)) 
    
#     def __getattr__(self, attr):
#         return getattr(self.checkpoint_, attr)

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
#     def on_batch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs ):
#         weights_norm = get_paramater_l2_norm(net)
#         print('epoch: %d, batch: %d, weights_norm : %.3f'%(self.epoch_num, self.batch_num, weights_norm))
    
    def on_grad_computed(self, net, named_parameters, **kwargs):
        if self.norm_type == 1:
            gradient_norm = get_paramater_gradient_inf_norm(net)
            print('epoch: %d, batch: %d, gradient_norm: %.3f'%(self.epoch_num, self.batch_num, gradient_norm))
            self.write_to_file(gradient_norm)
            self.batch_num += 1
        else:
            gradient_norm = get_paramater_gradient_l2_norm(net)
            print('epoch: %d, batch: %d, gradient_norm: %.3f'%(self.epoch_num, self.batch_num, gradient_norm))
            self.write_to_file(gradient_norm)
            self.batch_num += 1
        

    def write_to_file(self, gradient_norm):
        row_df = pd.DataFrame([[
            self.epoch_num,
            self.batch_num, 
            gradient_norm]],
            columns=['epoch', 
                     'batch', 
                     'gradient_norm'])
        csv_str = row_df.to_csv(
            None,
            float_format='%.3f',
            index=False,
            header=True if (self.epoch_num == 1 and self.batch_num == 1) else False,
            )
        
        
        if self.epoch_num == 1 and self.batch_num == 1:
            # At start, write to a clean file with mode 'w'
            with open(self.f_history, 'w') as f:
                f.write(csv_str)
        else:
            # Append to existing file with mode 'a'
            with open(self.f_history, 'a') as f:
                f.write(csv_str)        

                
# class LSTMtoLogReg(Callback):
#     def __init__(self):
#         self.batch_num = 1
#         self.epoch_num = 1
        
#     def on_epoch_begin(self, net,  dataset_train=None, dataset_valid=None, **kwargs):
#         self.batch_num = 1
            
#     def on_grad_computed(self, net, named_parameters, **kwargs):

#     def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
#         self.epoch_num += 1        
        # set the W_ii, W_if, W_ig to 0 and only make W_io trainable
        # TODO : remove gradient computation on these weights and biases which would require unvectorizing the rnn.weight_ih_l0 tensor and setting gradients to zero. Might need restructuring of forward method.
#         wmask = torch.ones(net.module_.rnn.weight_ih_l0.shape)
#         wmask[:3*net.module__n_hiddens,:]=0 
#         net.module_.rnn.weight_ih_l0.data = net.module_.rnn.weight_ih_l0*wmask
        
#         bmask = torch.ones(net.module_.rnn.bias_ih_l0.shape)   
#         bmask[:3*net.module__n_hiddens]=0 
#         net.module_.rnn.bias_ih_l0.data = net.module_.rnn.bias_ih_l0*bmask
        
if __name__ == '__main__':
    main()
    

##  SKORCH WITH GRIDSEARCHCV CODE (NOT USED)
# rnn = RNNBinaryClassifier( 
#          max_epochs=100, 
#          batch_size=args.batch_size, 
#          device=device, 
#          callbacks=[ 
#          skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=True, name='aucroc_score_train'), 
#          skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=False, name='aucroc_score_valid'), 
#          skorch.callbacks.EarlyStopping(monitor='aucroc_score_valid', patience=5, threshold=1e-10, threshold_mode='rel', lower_is_better=False)], 
#          criterion=torch.nn.NLLLoss, 
#          criterion__weight=class_weights, 
#          train_split=skorch.dataset.CVSplit(0.2), 
#          module__rnn_type='LSTM', 
#          module__n_layers=1, 
#          module__n_inputs=X_train.shape[-1], 
#          optimizer=torch.optim.Adam)  

# params = {'lr':[0.001, 0.005, 0.0001], 'module__n_hiddens':[128, 588]} 
# gs = GridSearchCV(rnn, params, refit=False, scoring=roc_auc_scorer) 
# gs.fit(X_train, y_train)

#---------------------------------------------------------------------#
# Skorch LR
#---------------------------------------------------------------------#

#     X_train_ = X_train[:,0,:]
#     logistic = SkorchLogisticRegression(n_features=X_train_.shape[1],
# #                                         l2_penalty_weights=0.1,
# #                                         l2_penalty_bias=0.01,
# #                                         clip=0.2,
# #                                         lr=args.lr,
#                                         batch_size=args.batch_size, 
#                                         max_epochs=args.epochs,
#                                         train_split=skorch.dataset.CVSplit(4),
# #                                         criterion=torch.nn.CrossEntropyLoss,
#                                         criterion=torch.nn.NLLLoss,
#                                         criterion__weight=class_weights,
#                                         optimizer__weight_decay=0.1,
#                                         callbacks=[skorch.callbacks.PrintLog(floatfmt='.2f'),
# #                                                    skorch.callbacks.EarlyStopping(monitor='train_loss', patience=300, threshold=1e-10, threshold_mode='rel', lower_is_better=True)
#                                                    skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=True, name='aucroc_score_train'),
#                                                    skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=False, name='aucroc_score_valid'),
#         ],
#                                        optimizer=torch.optim.SGD)
    
# #     pipe = Pipeline([
# #     ('scale', StandardScaler()),
# #     ('classifier', logistic),
# # ])
    
    
#     best_logistic = logistic.fit(X_train_, y_train)  

#---------------------------------------------------------------------#
# LSTM
#---------------------------------------------------------------------#
    
#     # instantiate RNN
#     rnn = RNNBinaryClassifier(
#         max_epochs=args.epochs,
#         batch_size=args.batch_size,
#         device=device,
# #         criterion=torch.nn.CrossEntropyLoss,
#         criterion=torch.nn.NLLLoss,
#         criterion__weight=class_weights,
#         train_split=skorch.dataset.CVSplit(0.2),
#         callbacks=[
# #             skorch.callbacks.GradientNormClipping(gradient_clip_value=0.2, gradient_clip_norm_type=2) ,
#             skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=True, name='aucroc_score_train'),
#             skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=False, name='aucroc_score_valid'),
#             ComputeGradientNorm(norm_type=2, f_history = args.report_dir + '/%s_running_rnn_classifer_gradient_norm_history.csv'%args.output_filename_prefix),
# #             LSTMtoLogReg(),# transformation to log reg for debugging
#             skorch.callbacks.EarlyStopping(monitor='aucroc_score_valid', patience=5, threshold=1e-10, threshold_mode='rel', lower_is_better=False),
#             skorch.callbacks.Checkpoint(monitor='train_loss', f_history = args.report_dir + '/%s_running_rnn_classifer_history.json'%args.output_filename_prefix),
#             skorch.callbacks.Checkpoint(monitor='aucroc_score_valid', f_params = args.report_dir + '/%s_running_rnn_classifer_model'%args.output_filename_prefix),
#             skorch.callbacks.PrintLog(floatfmt='.2f')
#         ],
#         module__rnn_type='LSTM',
#         module__n_inputs=X_train.shape[-1],
#         module__n_hiddens=args.hidden_units,
#         module__n_layers=1,
#         module__dropout_proba_non_recurrent=args.dropout,
# #         module__dropout_proba=args.dropout,
#         optimizer=torch.optim.SGD,
#         optimizer__weight_decay=0.1,
# #         optimizer__momentum=0.9,
# #         optimizer=torch.optim.Adam,
#         lr=args.lr) 
    
#     # scale input features
#     rnn.fit(X_train, y_train)  
    
#     # get the training history
#     epochs, train_loss, validation_loss, aucroc_score_train, aucroc_score_valid = get_loss_plots_from_training_history(rnn.history)

