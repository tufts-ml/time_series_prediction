import sys, os
import argparse
import numpy as np 
import pandas as pd
import json

import torch
import skorch
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                            balanced_accuracy_score, confusion_matrix, 
                            roc_auc_score, make_scorer)
from yattag import Doc
import matplotlib.pyplot as plt

from dataset_loader import TidySequentialDataCSVLoader
from RNNBinaryClassifier import RNNBinaryClassifier

from joblib import dump

import warnings
warnings.filterwarnings("ignore")

from skorch.callbacks import Callback
from MLPModule import MLPModule
from skorch import NeuralNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from SkorchLogisticRegression import SkorchLogisticRegression

           
def main():
    parser = argparse.ArgumentParser(description='PyTorch RNN with variable-length numeric sequences wrapper')
    
    parser.add_argument('--train_vitals_csv', type=str,
                        help='Location of vitals data for training')
    parser.add_argument('--test_vitals_csv', type=str,
                        help='Location of vitals data for testing')
    parser.add_argument('--metadata_csv', type=str,
                        help='Location of metadata for testing and training')
    parser.add_argument('--data_dict', type=str)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of sequences per minibatch')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='Number of epochs')
    parser.add_argument('--hidden_units', type=int, default=10,
                        help='Number of hidden units')   
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout for optimizer')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--save', type=str,  default='RNNmodel.pt',
                        help='path to save the final model')
    parser.add_argument('--report_dir', type=str, default='html',
                        help='dir in which to save results report')
    parser.add_argument('--simulated_data_dir', type=str, default='simulated_data/2-state/',
                        help='dir in which to simulated data is saved')    
    parser.add_argument('--is_data_simulated', type=bool, default=False,
                        help='boolean to check if data is simulated or from mimic')
    parser.add_argument('--output_filename_prefix', type=str, default='current_config', help='file to save the loss and validation over epochs')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cpu'
    
    # hyperparameter space
#     learning_rate = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
#     hyperparameters = dict(lr=learning_rate)
    
    # extract data
    if not(args.is_data_simulated):
#------------------------Loaded from TidySequentialDataCSVLoader--------------------#
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
    
    # set class weights as (1-Beta)/(1-Beta^(number of training samples in class))
#     beta = (len(y_train)-1)/len(y_train)
#     class_weights = torch.tensor(np.asarray([(1-beta)/(1-beta**((y_train==0).sum())), (1-beta)/(1-beta**((y_train==1).sum()))]))
    
    # set class weights as 1/(number of samples in class) for each class to handle class imbalance
    class_weights = torch.tensor([1/(y_train==0).sum(),
                                  1/(y_train==1).sum()]).double()
    
    
    # define a auc scorer function and pass it as callback of skorch to track training and validation AUROC
    roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                                 needs_threshold=True)
    
        # use only last time step as feature for LR debugging
#     X_train = X_train[:,-1,:][:,np.newaxis,:]
#     X_test = X_test[:,-1,:][:,np.newaxis,:]
    
    # use time steps * features as vectorized feature into RNN for LR debugging
#     X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]*X_train.shape[2]))
#     X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]*X_test.shape[2]))
    
#---------------------------------------------------------------------#
# Pseudo LSTM (hand engineered features through LSTM, collapsed across time)
#---------------------------------------------------------------------#
    # instantiate RNN
    rnn = RNNBinaryClassifier(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__weight=class_weights,
        train_split=skorch.dataset.CVSplit(4),
        callbacks=[
            skorch.callbacks.GradientNormClipping(gradient_clip_value=0.4, gradient_clip_norm_type=2) ,
            skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=True, name='aucroc_score_train'),
            skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=False, name='aucroc_score_valid'),
            ComputeGradientNorm(norm_type=2, f_history = args.report_dir + '/%s_running_rnn_classifer_gradient_norm_history.csv'%args.output_filename_prefix),
#             LSTMtoLogReg(),# transformation to log reg for debugging
            skorch.callbacks.EarlyStopping(monitor='aucroc_score_valid', patience=1000, threshold=1e-10, threshold_mode='rel', lower_is_better=False),
            skorch.callbacks.Checkpoint(monitor='train_loss', f_history = args.report_dir + '/%s_running_rnn_classifer_history.json'%args.output_filename_prefix),
#             skorch.callbacks.Checkpoint(monitor='aucroc_score_valid', f_pickle = args.report_dir + '/%s_running_rnn_classifer_model'%args.output_filename_prefix),
            skorch.callbacks.PrintLog(floatfmt='.2f')
        ],
        module__rnn_type='LSTM',
        module__n_inputs=X_train.shape[-1],
        module__n_hiddens=args.hidden_units,
        module__n_layers=1,
#         module__dropout_proba_non_recurrent=args.dropout,
#         module__dropout_proba=args.dropout,
        optimizer=torch.optim.SGD,
        optimizer__weight_decay=1e-2,
#         optimizer__momentum=0.9,
#         optimizer=torch.optim.Adam,
        lr=args.lr)
    
    from IPython import embed; embed()
    
    # scale input features
    X_train = standard_scaler_3d(X_train)
    X_test = standard_scaler_3d(X_test) 
    rnn.fit(X_train, y_train)
    
    
    # get the training history
    epochs, train_loss, validation_loss, aucroc_score_train, aucroc_score_valid = get_loss_plots_from_training_history(rnn.history)
    
    
    # plot the validation and training error plots and save
    f = plt.figure()
    plt.plot(epochs, train_loss, 'r-.', label = 'Train Loss')
    plt.plot(epochs, validation_loss, 'b-.', label = 'Validation Loss')
    plt.plot(epochs, aucroc_score_train, 'g-.', label = 'AUCROC score (Train)')
    plt.plot(epochs, aucroc_score_valid, 'm-.', label = 'AUCROC score (Valid)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Performance (learning rate : %s, hidden units : %s)'%(str(args.lr), str(args.hidden_units)))
    f.savefig(args.report_dir + '/%s_training_performance_plots.png'%args.output_filename_prefix)
    plt.close()
    
    # save the training and validation loss in a csv
    train_perf_df = pd.DataFrame(data=np.stack([epochs, train_loss,
                                                validation_loss]).T, 
                                 columns=['epochs', 'train_loss','validation_loss']) 
    train_perf_df.to_csv(args.report_dir + '/%s_perf_metrics.csv'%args.output_filename_prefix)
    
    
    # save classifier history to later evaluate early stopping for this model
    dump(rnn, args.report_dir + '/%s_rnn_classifer.pkl'%args.output_filename_prefix)

    
    
    y_pred_proba = rnn.predict_proba(X_test)
    y_pred = convert_proba_to_binary(y_pred_proba)
    
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_pos)
    roc_area = roc_auc_score(y_test, y_pred_proba_pos)
    
    from IPython import embed; embed()
    
    # Brief Summary
#     print('Best lr:', rnn.best_estimator_.get_params()['lr'])
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_pred))
    print('Log Loss:', log_loss(y_test, y_pred_proba))
    print('AUC ROC:', roc_area)
    conf_matrix = confusion_matrix(y_test, y_pred)
    true_neg = conf_matrix[0][0]
    true_pos = conf_matrix[1][1]
    false_neg = conf_matrix[1][0]
    false_pos = conf_matrix[0][1]
    print('True Positive Rate:', float(true_pos) / (true_pos + false_neg))
    print('True Negative Rate:', float(true_neg) / (true_neg + false_pos))
    print('Positive Predictive Value:', float(true_pos) / (true_pos + false_pos))
    print('Negative Predictive Value', float(true_neg) / (true_neg + false_pos))

    create_html_report(args.report_dir, args.output_filename_prefix, y_test, y_pred, y_pred_proba, args.lr)

def create_html_report(report_dir, output_filename_prefix, y_test, y_pred, y_pred_proba, lr):
    try:
        os.mkdir(report_dir)
    except OSError:
        pass

    # Set up HTML report
    doc, tag, text = Doc().tagtext()

    # Metadata
    with tag('h2'):
        text('Random rnn Classifier Results')
    with tag('h3'):
        text('Hyperparameters searched:')
    with tag('p'):
        text('Learning rate: ', str(lr))

#     # Model
#     with tag('h3'):
#         text('Parameters of best model:')
#     with tag('p'):
#         text('Learning rate: ', best_lr)
    
    # Performance
    with tag('h3'):
        text('Performance Metrics:')
    with tag('p'):
        text('Accuracy: ', accuracy_score(y_test, y_pred))
    with tag('p'):
        text('Balanced Accuracy: ', balanced_accuracy_score(y_test, y_pred))
    with tag('p'):
        text('Log Loss: ', log_loss(y_test, y_pred_proba))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    true_neg = conf_matrix[0][0]
    true_pos = conf_matrix[1][1]
    false_neg = conf_matrix[1][0]
    false_pos = conf_matrix[0][1]
    with tag('p'):
        text('True Positive Rate: ', float(true_pos) / (true_pos + false_neg))
    with tag('p'):
        text('True Negative Rate: ', float(true_neg) / (true_neg + false_pos))
    with tag('p'):
        text('Positive Predictive Value: ', float(true_pos) / (true_pos + false_pos))
    with tag('p'):
        text('Negative Predictive Value: ', float(true_neg) / (true_neg + false_pos))
    
    # Confusion Matrix
    columns = ['Predicted 0', 'Predicted 1']
    rows = ['Actual 0', 'Actual 1']
    cell_text = []
    for cm_row, cm_norm_row in zip(conf_matrix, conf_matrix_norm):
        row_text = []
        for i, i_norm in zip(cm_row, cm_norm_row):
            row_text.append('{} ({})'.format(i, i_norm))
        cell_text.append(row_text)

    ax = plt.subplot(111, frame_on=False) 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)

    confusion_table = ax.table(cellText=cell_text,
                               rowLabels=rows,
                               colLabels=columns,
                               loc='center')
    plt.savefig(report_dir + '/%s_confusion_matrix.png'%output_filename_prefix)
    plt.close()

    with tag('p'):
        text('Confusion Matrix:')
    doc.stag('img', src=('%s_confusion_matrix.png'%output_filename_prefix))

    # ROC curve/area
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_pos)
    roc_area = roc_auc_score(y_test, y_pred_proba_pos)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR Test')
    plt.ylabel('TPR Test')
    plt.title('AUC : {}'.format(roc_area))
    plt.savefig(report_dir + '/%s_roc_curve.png'%output_filename_prefix)
    plt.close()

    with tag('p'):
        text('ROC Curve:')
    doc.stag('img', src=('/%s_roc_curve.png'%output_filename_prefix))  
    with tag('p'):
        text('ROC Area: ', roc_area)

    with open(report_dir + '/%s_report.html'%output_filename_prefix, 'w') as f:
        f.write(doc.getvalue())

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


def get_loss_plots_from_training_history(train_history):
    epochs = train_history[:,'epoch']
    train_loss = train_history[:,'train_loss']
    valid_loss = train_history[:,'valid_loss']
    aucroc_score_train = train_history[:,'aucroc_score_train']
    aucroc_score_valid = train_history[:,'aucroc_score_valid']
    
    return epochs, train_loss, valid_loss, aucroc_score_train, aucroc_score_valid


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

                
class LSTMtoLogReg(Callback):
    def __init__(self):
        self.batch_num = 1
        self.epoch_num = 1
        
    def on_epoch_begin(self, net,  dataset_train=None, dataset_valid=None, **kwargs):
        self.batch_num = 1
#         from IPython import embed; embed()
        # set the W_ho to zero remove time-dependence
        net.module_.rnn.weight_hh_l0.requires_grad=False
        net.module_.rnn.weight_hh_l0.data=torch.zeros(4*net.module__n_hiddens,net.module__n_hiddens, dtype=torch.float64)
        # set the W_ho to zero remove time-dependence
        net.module_.rnn.bias_hh_l0.requires_grad = False  
        net.module_.rnn.bias_hh_l0.data = torch.zeros(128, dtype=torch.float64)
        
# class NonRecurrentUnitsDropout(Callback):
#     def __init__(self, dropout_proba=.3):
#         self.dropout=torch.nn.Dropout(p=dropout_proba, inplace=True)
        
#     def on_epoch_end(self, net,  dataset_train=None, dataset_valid=None, **kwargs): 
#         # apply dropout to the non-recurrent layer weights between LSTM layers before output ie is weights for h_(l-1)^t
#         # See https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM for choosing the right weights
#         from IPython import embed; embed()
#         self.dropout(net.module_.rnn.weight_ih_l1)
#         self.dropout(net.module_.rnn.bias_ih_l1)
        
        
        
        
        
#     def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
#         self.epoch_num += 1
        # set the W_ho to zero remove time-dependence

        
#     def on_grad_computed(self, net, named_parameters, **kwargs):  
#         from IPython import embed; embed()        
    

if __name__ == '__main__':
    main()
    

##  SKORCH WITH GRIDSEARCHCV CODE (NOT USED)
#     cv_results = cross_validate(rnn, X_train, y_train, verbose=10, return_estimator=True, scoring=roc_auc_scorer, cv=2, 
#                                 n_jobs=-1)
    # history is in cv_results['estimator'][0].history_    
#     classifier = GridSearchCV(rnn, hyperparameters, n_jobs=-1, cv=5, scoring = roc_auc_scorer, verbose=10, return_train_score = True)
#     best_rnn = classifier.fit(X_train, y_train)
    
    # get the training and validation loss plots over epochs
#     train_history = best_rnn.best_estimator_.history_
#     epochs, train_loss, validation_loss = get_loss_plots_from_training_history(train_history)
    # View best hyperparameters
#     best_lr = best_rnn.best_estimator_.get_params()['lr']


## MLP ## 
#     net = NeuralNet(
#         module=MLPModule,
#         module__num_inputs=X_train.shape[-1],
#         criterion=torch.nn.CrossEntropyLoss,
#         lr=1.0,
#         max_epochs=args.epochs,
#         batch_size=args.batch_size,
# #         batch_size=-1,
#         device=device,
#         criterion__weight=class_weights,
#         train_split=skorch.dataset.CVSplit(4),
#         callbacks=[
#             skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=True, name='aucroc_score_train'),
#             skorch.callbacks.EpochScoring(roc_auc_scorer, lower_is_better=False, on_train=False, name='aucroc_score_valid'),
#             skorch.callbacks.EarlyStopping(monitor='aucroc_score_valid', patience=3000, threshold=1e-10, threshold_mode='rel', lower_is_better=True),
#             skorch.callbacks.PrintLog(floatfmt='.2f')
#         ],
#         optimizer=torch.optim.SGD,

#     )
    
#     from IPython import embed; embed()
#     # use only the last time step as feature for every subject
#     X_train_mlp = X_train[:,-1,:]
#     y_train_mlp = y_train[:,np.newaxis]
#     net.fit(X_train_mlp, y_train)

#---------------------------------------------------------------------#
# Logistic Regression(features : data at last time point, No hidden units, sigmoid at output)
#---------------------------------------------------------------------#
#     logistic = SkorchLogisticRegression(n_features=X_train.shape[-1],
#                                         l2_penalty_weights=0.1,
#                                         l2_penalty_bias=0.01,
#                                         clip=0.2,
#                                         lr=0.01,
#                                         batch_size=-1, 
#                                         max_epochs=1600,
#                                         criterion=torch.nn.CrossEntropyLoss,
#                                         criterion__weight=class_weights,
#                                         callbacks=[skorch.callbacks.PrintLog(floatfmt='.2f'),
#                                                    skorch.callbacks.EarlyStopping(monitor='train_loss', patience=300, threshold=1e-10, threshold_mode='rel', lower_is_better=True),
#                                                    ],
#                                         optimizer=torch.optim.SGD)

#     pipe = Pipeline([
#     ('scale', StandardScaler()),
#     ('classifier', logistic),
#     ])
#     X_train_mlp = X_train[:,-1,:]
    
#     best_logistic = pipe.fit(X_train_mlp, y_train)
    
#     X_test_mlp = X_test[:,-1,:] 
#     y_pred_proba = best_logistic.predict_proba(X_test_mlp) 
#     y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
#     print(roc_auc_score(y_test, y_pred_proba_pos))

