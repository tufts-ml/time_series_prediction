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
                             roc_auc_score, make_scorer, average_precision_score)
PROJECT_REPO_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'rnn'))
from yattag import Doc
import matplotlib.pyplot as plt
import torch.utils.data as utils
from dataset_loader import TidySequentialDataCSVLoader
from RNNBinaryClassifier import RNNBinaryClassifier
from feature_transformation import (parse_id_cols, parse_feature_cols)
from utils import load_data_dict_json
from joblib import dump
from GRU_D import GRUD
from GRUD_model import grud_model

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

def data_dataloader(train_data, train_outcomes, valid_data, valid_outcomes, test_data, test_outcomes, batch_size=512):
    
    
    # ndarray to tensor
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_outcomes[:, np.newaxis])
    dev_data, dev_label = torch.Tensor(valid_data), torch.Tensor(valid_outcomes[:, np.newaxis])
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_outcomes[:, np.newaxis])
    
    # tensor to dataset
    train_dataset = utils.TensorDataset(train_data, train_label)
    dev_dataset = utils.TensorDataset(dev_data, dev_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    # dataset to dataloader 
    train_dataloader = utils.DataLoader(train_dataset)
    dev_dataloader = utils.DataLoader(dev_dataset)
    test_dataloader = utils.DataLoader(test_dataset)
    
    print("train_data.shape : {}\t train_label.shape : {}".format(train_data.shape, train_label.shape))
    print("dev_data.shape : {}\t dev_label.shape : {}".format(dev_data.shape, dev_label.shape))
    print("test_data.shape : {}\t test_label.shape : {}".format(test_data.shape, test_label.shape))
    
    return train_dataloader, dev_dataloader, test_dataloader


'''
def fit(model, criterion, learning_rate,\
        train_dataloader, dev_dataloader, test_dataloader,\
        learning_rate_decay=0, n_epochs=30, weight_decay=0):
    epoch_losses = []
    epoch_perfs = []
    
    # to check the update 
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()
    
    for epoch in range(n_epochs):
        
        if learning_rate_decay != 0:

            # every [decay_step] epoch reduce the learning rate by half
            if  epoch % learning_rate_decay == 0:
                learning_rate = learning_rate/2
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))
        
        # train the model
        losses, acc = [], []
        label, pred = [], []
        y_pred_col= []
        model.train()
        for train_data, train_label in train_dataloader: 
            model.train() 
            #push current sample to GPU or CPU 
            train_data, train_label = train_data.to("cpu"), train_label.to("cpu") 
            # Zero the parameter gradients 
            optimizer.zero_grad() 
            # Forward pass : Compute predicted y by passing train data to the model 
            y_pred = model(train_data)[:,-1,:] #last output 
            
            
            # Save predict and label
            y_pred_col.append(y_pred.detach().numpy())
            pred.append((y_pred.detach().numpy() > 0.5)*1.0)
            label.append(train_label.detach().numpy())
            
            #print('y_pred: {}\t label: {}'.format(y_pred, train_label))

            # Compute loss
            loss = criterion(y_pred, train_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    train_label)
            )
            losses.append(loss.detach().numpy())

            # perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

        
        train_acc = torch.mean(torch.cat(acc).float())
        train_loss = np.mean(losses)
        
        train_pred_out = pred
        train_label_out = label
        
        # save new params
        new_state_dict= {}
        for key in model.state_dict():
            new_state_dict[key] = model.state_dict()[key].clone()
            
        # compare params
        for key in old_state_dict:
            if (old_state_dict[key] == new_state_dict[key]).all():
                print('Not updated in {}'.format(key))
   
        
        # dev loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for dev_data, dev_label in dev_dataloader:
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(dev_data)[:,-1,:]
            
            # Save predict and label
            pred.append((y_pred.detach().numpy() > 0.5)*1.0)
            label.append(dev_label.detach().numpy())

            # Compute loss
            loss = criterion(y_pred, dev_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    dev_label)
            )
            losses.append(loss.detach().numpy())
            
        dev_acc = torch.mean(torch.cat(acc).float())
        dev_loss = np.mean(losses)
        
        dev_pred_out = pred
        dev_label_out = label
        
        # test loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for test_data, test_label in test_dataloader:
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(test_data)[:,-1,:]
            
            # Save predict and label
            pred.append((y_pred.detach().numpy() > 0.5)*1.0)
            label.append(test_label.detach().numpy())

            # Compute loss
            loss = criterion(y_pred, test_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    test_label)
            )
            losses.append(loss.detach().numpy())
            
        test_acc = torch.mean(torch.cat(acc).float())
        test_loss = np.mean(losses)
        
        test_pred_out = pred
        test_label_out = label
          
        pred = np.concatenate(pred)
        label = np.concatenate(label)
        
        train_pred = np.concatenate(train_pred_out)
        train_label = np.concatenate(train_label_out)   
        
        valid_pred = np.concatenate(dev_pred_out)
        valid_label = np.concatenate(dev_label_out) 
        
        auc_score = roc_auc_score(label, pred)
        auprc_score = average_precision_score(label, pred)
        
        valid_auc_score = roc_auc_score(valid_label, valid_pred)
        train_auc_score = roc_auc_score(train_label, train_pred)        
        
        valid_auprc_score = average_precision_score(valid_label, valid_pred)
        train_auprc_score = average_precision_score(train_label, train_pred)
        
        curr_perf_dict = {'train_loss' : train_loss,
                         'valid_loss' : dev_loss,
                         'test_loss' : test_loss,
                         'train_AUC' : train_auc_score,
                         'valid_AUC' : valid_auc_score,
                         'test_AUC' : auc_score,
                         'train_AUPRC' : train_auprc_score,
                         'valid_AUPRC' : valid_auprc_score,
                         'test_AUPRC' : auprc_score}
        
        
        epoch_perfs.append(curr_perf_dict)
        
        epoch_losses.append([
             train_loss, dev_loss, test_loss,
             train_acc, dev_acc, test_acc,
             train_pred_out, dev_pred_out, test_pred_out,
             train_label_out, dev_label_out, test_label_out,
         ])
        
        
        # print("Epoch: {} Train: {:.4f}/{:.2f}%, Dev: {:.4f}/{:.2f}%, Test: {:.4f}/{:.2f}% AUC: {:.4f}".format(
        #     epoch, train_loss, train_acc*100, dev_loss, dev_acc*100, test_loss, test_acc*100, auc_score))
        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Test loss: {:.4f}, Train AUPRC: {:.4f}, Valid AUPRC: {:.4f}, Test AUPRC: {:.4f}".format(
            epoch, train_loss, dev_loss, test_loss, train_auprc_score, valid_auprc_score, auprc_score))
        
        print("Epoch: {} Train AUROC: {:.4f}, Valid AUROC: {:.4f}, Test AUROC: {:.4f}".format(
            epoch, train_auc_score, valid_auc_score, auc_score))
        
        # save the parameters
        train_log = []
        train_log.append(model.state_dict())
        
    perf_df = pd.DataFrame(epoch_perfs)
    return perf_df 
'''

def fit(model, criterion, learning_rate,\
        train_dataloader, dev_dataloader, test_dataloader,\
        learning_rate_decay=0, n_epochs=30, weight_decay=0):
    epoch_losses = []
    epoch_perfs = []
    
    # to check the update 
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()
    
    for epoch in range(n_epochs):
        
        if learning_rate_decay != 0:

            # every [decay_step] epoch reduce the learning rate by half
            if  epoch % learning_rate_decay == 0:
                learning_rate = learning_rate/2
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))
        
        # train the model
        losses, acc = [], []
        label, pred = [], []
        y_pred_col= []
        model.train()
        for train_data, train_label in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            train_data = torch.squeeze(train_data)
            train_label = torch.squeeze(train_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(train_data)
            
            # y_pred = y_pred[:, None]
            # train_label = train_label[:, None]
            
            #print(y_pred.shape)
            #print(train_label.shape)
            
            # Save predict and label
            y_pred_col.append(y_pred.item())
            pred.append(y_pred.item() > 0.5)
            label.append(train_label.item())
            
            #print('y_pred: {}\t label: {}'.format(y_pred, train_label))

            # Compute loss
            loss = criterion(y_pred, train_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    train_label)
            )
            losses.append(loss.item())

            # perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

        
        train_acc = torch.mean(torch.cat(acc).float())
        train_loss = np.mean(losses)
        
        train_pred_out = pred
        train_label_out = label
        
        # save new params
        new_state_dict= {}
        for key in model.state_dict():
            new_state_dict[key] = model.state_dict()[key].clone()
            
        # compare params
        for key in old_state_dict:
            if (old_state_dict[key] == new_state_dict[key]).all():
                print('Not updated in {}'.format(key))
   
        
        # dev loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for dev_data, dev_label in dev_dataloader:
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            dev_data = torch.squeeze(dev_data)
            dev_label = torch.squeeze(dev_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(dev_data)
            
            # Save predict and label
            pred.append(y_pred.item())
            label.append(dev_label.item())

            # Compute loss
            loss = criterion(y_pred, dev_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    dev_label)
            )
            losses.append(loss.item())
            
        dev_acc = torch.mean(torch.cat(acc).float())
        dev_loss = np.mean(losses)
        
        dev_pred_out = pred
        dev_label_out = label
        
        # test loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for test_data, test_label in test_dataloader:
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            test_data = torch.squeeze(test_data)
            test_label = torch.squeeze(test_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(test_data)
            
            # Save predict and label
            pred.append(y_pred.item())
            label.append(test_label.item())

            # Compute loss
            loss = criterion(y_pred, test_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    test_label)
            )
            losses.append(loss.item())
            
        test_acc = torch.mean(torch.cat(acc).float())
        test_loss = np.mean(losses)
        
        test_pred_out = pred
        test_label_out = label
          
        pred = np.asarray(pred)
        label = np.asarray(label)
        
        train_pred = np.asarray(train_pred_out)
        train_label = np.asarray(train_label_out)   
        
        valid_pred = np.asarray(dev_pred_out)
        valid_label = np.asarray(dev_label_out) 
        
        auc_score = roc_auc_score(label, pred)
        auprc_score = average_precision_score(label, pred)
        
        valid_auc_score = roc_auc_score(valid_label, valid_pred)
        train_auc_score = roc_auc_score(train_label, train_pred)        
        
        valid_auprc_score = average_precision_score(valid_label, valid_pred)
        train_auprc_score = average_precision_score(train_label, train_pred)
        
        curr_perf_dict = {'train_loss' : train_loss,
                         'valid_loss' : dev_loss,
                         'test_loss' : test_loss,
                         'train_AUC' : train_auc_score,
                         'valid_AUC' : valid_auc_score,
                         'test_AUC' : auc_score,
                         'train_AUPRC' : train_auprc_score,
                         'valid_AUPRC' : valid_auprc_score,
                         'test_AUPRC' : auprc_score}
        
        
        epoch_perfs.append(curr_perf_dict)
        
        epoch_losses.append([
             train_loss, dev_loss, test_loss,
             train_acc, dev_acc, test_acc,
             train_pred_out, dev_pred_out, test_pred_out,
             train_label_out, dev_label_out, test_label_out,
         ])
        
        
        # print("Epoch: {} Train: {:.4f}/{:.2f}%, Dev: {:.4f}/{:.2f}%, Test: {:.4f}/{:.2f}% AUC: {:.4f}".format(
        #     epoch, train_loss, train_acc*100, dev_loss, dev_acc*100, test_loss, test_acc*100, auc_score))
        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Train loss: {:.4f}, Train AUPRC: {:.4f}, Valid AUPRC: {:.4f}, Test AUPRC: {:.4f}".format(
            epoch, train_loss, dev_loss, test_loss, train_auprc_score, valid_auprc_score, auprc_score))
        
        print("Epoch: {} Train AUROC: {:.4f}, Valid AUROC: {:.4f}, Test AUROC: {:.4f}".format(
            epoch, train_auc_score, valid_auc_score, auc_score))
        
        # save the parameters
        train_log = []
        train_log.append(model.state_dict())
        
    perf_df = pd.DataFrame(epoch_perfs)
    return perf_df 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='PyTorch RNN with variable-length numeric sequences wrapper')
    parser.add_argument('--outcome_col_name', type=str, required=True)
    parser.add_argument('--train_csv_files', type=str, required=True)
    parser.add_argument('--valid_csv_files', type=str, default=None)
    parser.add_argument('--test_csv_files', type=str, required=True)
    parser.add_argument('--data_dict_files', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of sequences per minibatch')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--hidden_units', type=int, default=32,
                        help='Number of hidden units')
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help='Number of hidden layers')
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
    parser.add_argument('--is_data_simulated', type=bool, default=False,
                        help='boolean to check if data is simulated or from mimic')
    parser.add_argument('--simulated_data_dir', type=str, default='simulated_data/2-state/',
                        help='dir in which to simulated data is saved.Must be provide if is_data_simulated = True')
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
    
    if args.valid_csv_files is None:
        # split train into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.validation_size, random_state=213)
    else:
        x_valid_csv_filename, y_valid_csv_filename = args.valid_csv_files.split(',')
        valid_vitals = TidySequentialDataCSVLoader(
            x_csv_path=x_valid_csv_filename,
            y_csv_path=y_valid_csv_filename,
            x_col_names=feature_cols,
            idx_col_names=id_cols,
            y_col_name=args.outcome_col_name,
            y_label_type='per_sequence'
        )
        X_valid, y_valid = valid_vitals.get_batch_data(batch_id=0)
    
    
    
    N_train, T, F = X_train.shape
    N_valid, T, F = X_valid.shape
    N_test, T, F = X_test.shape
    
    # the GRU-D needs the input in shape (n_samples, 3, n_feats, n_timepoints) 
    # second dimension is the (raw, mask, time_since_missing) 
    num_true_feats = int(F/3)
    X_train_reshaped = np.zeros((N_train, 3, T, num_true_feats))
    X_valid_reshaped = np.zeros((N_valid, 3, T, num_true_feats))
    X_test_reshaped = np.zeros((N_test, 3, T, num_true_feats))
    for ii in range(3):
        X_train_reshaped[:, ii, :, :] = X_train[:, :, num_true_feats*ii:num_true_feats*(ii+1)]
        X_valid_reshaped[:, ii, :, :] = X_valid[:, :, num_true_feats*ii:num_true_feats*(ii+1)]
        X_test_reshaped[:, ii, :, :] = X_test[:, :, num_true_feats*ii:num_true_feats*(ii+1)]
    X_train_reshaped = np.transpose(X_train_reshaped, (0, 1, 3, 2))
    X_valid_reshaped = np.transpose(X_valid_reshaped, (0, 1, 3, 2))
    X_test_reshaped = np.transpose(X_test_reshaped, (0, 1, 3, 2))
    
    
    train_dataloader, valid_dataloader, test_dataloader = data_dataloader(X_train_reshaped[:5000], y_train[:5000], 
                                                                          X_valid_reshaped[:5000], y_valid[:5000],
                                                                          X_test_reshaped[:5000], y_test[:5000],
                                                                          args.batch_size)
    
    
    # get the mean of each dimension for the GRU-D to decay to
    x_mean_F = np.zeros(num_true_feats)
    for f in range(num_true_feats):
        x_mean_F[f] = np.mean(X_train[:, :, f])
    x_mean_F = torch.Tensor(x_mean_F)
    
    
    
    if args.output_filename_prefix==None:
        output_filename_prefix = ('hiddens=%s-layers=%s-lr=%s-dropout=%s-weight_decay=%s'%(args.hidden_units, 
                                                                       args.hidden_layers, 
                                                                       args.lr, 
                                                                       args.dropout, args.weight_decay))
    else:
        output_filename_prefix = args.output_filename_prefix
        
    
    input_size = num_true_feats # num of variables base on the paper
    hidden_size = num_true_feats # same as inputsize
    output_size = 1
    num_layers = T # num of step or layers base on the paper
    model = GRUD(input_size = input_size, hidden_size= hidden_size, output_size=output_size, dropout=0, 
                 dropout_type='mloss', x_mean=x_mean_F, num_layers=num_layers)
    
#     from IPython import embed; embed()
#     model = grud_model(input_size = input_size, 
#                        hidden_size= args.hidden_units, 
#                        output_size=output_size, 
#                        dropout=0, 
#                        x_mean=x_mean_F, 
#                        num_layers=args.hidden_layers)  

    count = count_parameters(model)
    print('number of parameters : ' , count)
    print(list(model.parameters())[0].grad)
    
#     case_prev = y_train.sum()/len(y_train)
#     class_imb = torch.tensor(1/case_prev)
#     criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_imb)
    
    
    criterion = torch.nn.BCELoss()
    
    learning_rate = args.lr
    learning_rate_decay = 125
    n_epochs = 50
    
    # learning_rate = 0.1 learning_rate_decay=True
    perf_df = fit(model, criterion, learning_rate,\
                       train_dataloader, valid_dataloader, test_dataloader,\
                       learning_rate_decay, n_epochs, weight_decay=args.weight_decay)
    
    
    from IPython import embed; embed()
    test_perf_csv = os.path.join(args.output_dir, output_filename_prefix+'.csv')
    test_perf_df.to_csv(perf_df, index=False)
    
'''
def fit(model, criterion, learning_rate,\
        train_dataloader, dev_dataloader, test_dataloader,\
        learning_rate_decay=0, n_epochs=30, weight_decay=0):
    epoch_losses = []
    epoch_perfs = []
    
    # to check the update 
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()
    
    for epoch in range(n_epochs):
        
        if learning_rate_decay != 0:

            # every [decay_step] epoch reduce the learning rate by half
            if  epoch % learning_rate_decay == 0:
                learning_rate = learning_rate/2
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))
        
        # train the model
        losses, acc = [], []
        label, pred = [], []
        y_pred_col= []
        model.train()
        for train_data, train_label in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            train_data = torch.squeeze(train_data)
            train_label = torch.squeeze(train_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(train_data)
            
            # y_pred = y_pred[:, None]
            # train_label = train_label[:, None]
            
            #print(y_pred.shape)
            #print(train_label.shape)
            
            # Save predict and label
            y_pred_col.append(y_pred.item())
            pred.append(y_pred.item() > 0.5)
            label.append(train_label.item())
            
            #print('y_pred: {}\t label: {}'.format(y_pred, train_label))

            # Compute loss
            loss = criterion(y_pred, train_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    train_label)
            )
            losses.append(loss.item())

            # perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

        
        train_acc = torch.mean(torch.cat(acc).float())
        train_loss = np.mean(losses)
        
        train_pred_out = pred
        train_label_out = label
        
        # save new params
        new_state_dict= {}
        for key in model.state_dict():
            new_state_dict[key] = model.state_dict()[key].clone()
            
        # compare params
        for key in old_state_dict:
            if (old_state_dict[key] == new_state_dict[key]).all():
                print('Not updated in {}'.format(key))
   
        
        # dev loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for dev_data, dev_label in dev_dataloader:
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            dev_data = torch.squeeze(dev_data)
            dev_label = torch.squeeze(dev_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(dev_data)
            
            # Save predict and label
            pred.append(y_pred.item())
            label.append(dev_label.item())

            # Compute loss
            loss = criterion(y_pred, dev_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    dev_label)
            )
            losses.append(loss.item())
            
        dev_acc = torch.mean(torch.cat(acc).float())
        dev_loss = np.mean(losses)
        
        dev_pred_out = pred
        dev_label_out = label
        
        # test loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for test_data, test_label in test_dataloader:
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            test_data = torch.squeeze(test_data)
            test_label = torch.squeeze(test_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(test_data)
            
            # Save predict and label
            pred.append(y_pred.item())
            label.append(test_label.item())

            # Compute loss
            loss = criterion(y_pred, test_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    test_label)
            )
            losses.append(loss.item())
            
        test_acc = torch.mean(torch.cat(acc).float())
        test_loss = np.mean(losses)
        
        test_pred_out = pred
        test_label_out = label
          
        pred = np.asarray(pred)
        label = np.asarray(label)
        
        train_pred = np.asarray(train_pred_out)
        train_label = np.asarray(train_label_out)   
        
        valid_pred = np.asarray(dev_pred_out)
        valid_label = np.asarray(dev_label_out) 
        
        auc_score = roc_auc_score(label, pred)
        auprc_score = average_precision_score(label, pred)
        
        valid_auc_score = roc_auc_score(valid_label, valid_pred)
        train_auc_score = roc_auc_score(train_label, train_pred)        
        
        valid_auprc_score = average_precision_score(valid_label, valid_pred)
        train_auprc_score = average_precision_score(train_label, train_pred)
        
        curr_perf_dict = {'train_loss' : train_loss,
                         'valid_loss' : dev_loss,
                         'test_loss' : test_loss,
                         'train_AUC' : train_auc_score,
                         'valid_AUC' : valid_auc_score,
                         'test_AUC' : auc_score,
                         'train_AUPRC' : train_auprc_score,
                         'valid_AUPRC' : valid_auprc_score,
                         'test_AUPRC' : auprc_score}
        
        
        epoch_perfs.append(curr_perf_dict)
        
        epoch_losses.append([
             train_loss, dev_loss, test_loss,
             train_acc, dev_acc, test_acc,
             train_pred_out, dev_pred_out, test_pred_out,
             train_label_out, dev_label_out, test_label_out,
         ])
        
        
        # print("Epoch: {} Train: {:.4f}/{:.2f}%, Dev: {:.4f}/{:.2f}%, Test: {:.4f}/{:.2f}% AUC: {:.4f}".format(
        #     epoch, train_loss, train_acc*100, dev_loss, dev_acc*100, test_loss, test_acc*100, auc_score))
        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Train loss: {:.4f}, Train AUPRC: {:.4f}, Valid AUPRC: {:.4f}, Test AUPRC: {:.4f}".format(
            epoch, train_loss, dev_loss, test_loss, train_auprc_score, valid_auprc_score, auprc_score))
        
        print("Epoch: {} Train AUROC: {:.4f}, Valid AUROC: {:.4f}, Test AUROC: {:.4f}".format(
            epoch, train_auc_score, valid_auc_score, auc_score))
        
        # save the parameters
        train_log = []
        train_log.append(model.state_dict())
        
    perf_df = pd.DataFrame(epoch_perfs)
    return perf_df 
'''

if __name__ == '__main__':
    main()
    
