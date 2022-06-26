from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import sys
import numpy as np
PROJECT_REPO_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math

# import torch.utils as utils
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from models import TapNet
from utils import *


PROJECT_SRC_DIR = '/cluster/tufts/hugheslab/prath01/projects/time_series_prediction/src/'
sys.path.append(PROJECT_SRC_DIR)
from sklearn.metrics import (roc_auc_score, average_precision_score)



parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
# parser.add_argument('--train_np_files', type=str, required=True)
# parser.add_argument('--test_np_files', type=str, required=True)
# parser.add_argument('--valid_np_files', type=str, default=None)
parser.add_argument('--epochs', default=35, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=512, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--seed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--train-iteration', type=int, default=1024,
                        help='Number of iteration per epoch')
parser.add_argument('--output_dir', default='result',
                        help='output directory')
parser.add_argument('--output_filename_prefix', default='result',
                        help='prefix of outcome file')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--perc_labelled', default='33.3')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
np.random.seed(args.seed)

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
    train_dataloader = utils.DataLoader(train_dataset, batch_size=batch_size)
    dev_dataloader = utils.DataLoader(dev_dataset, batch_size=batch_size)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=batch_size)
    
    print("train_data.shape : {}\t train_label.shape : {}".format(train_data.shape, train_label.shape))
    print("dev_data.shape : {}\t dev_label.shape : {}".format(dev_data.shape, dev_label.shape))
    print("test_data.shape : {}\t test_label.shape : {}".format(test_data.shape, test_label.shape))
    
    return train_dataloader, dev_dataloader, test_dataloader

# training function
def train(model, X, y, X_val, y_val, optimizer):
    loss_list = [1e+5]
    test_best_possible, best_so_far = 0.0, 0.0
    for epoch in range(100):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        output, proto_dist = model(X, y)

        loss_train = F.cross_entropy(output, torch.tensor(y).long())
#         if args.use_metric:
#             loss_train = loss_train + args.metric_param * proto_dist


        loss_list.append(loss_train.item())
        
        auprc_train = average_precision_score(y, output[:, 1].detach().numpy())
        loss_train.backward()
        optimizer.step()
        
        output_val, _ = model(X_val, y_val)
        loss_val = F.cross_entropy(output_val, torch.tensor(y_val).long())
        auprc_val = average_precision_score(y_val, output_val[:, 1].detach().numpy())
        
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.8f}'.format(loss_train.item()),
              'AUPRC_train: {:.4f}'.format(auprc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'AUPRC_val: {:.4f}'.format(auprc_val.item()))

        if auprc_val.item() > test_best_possible:
            test_best_possible = auprc_val.item()

        if best_so_far > auprc_val.item():
            best_so_far = auprc_val.item()
    
    print("val AUPRC: " + str(auprc_val.item()))
    print("best possible: " + str(test_best_possible))

# # test function
# def test():
#     output, proto_dist = model(input)
#     loss_test = F.cross_entropy(output[idx_test], torch.squeeze(labels[idx_test]))
#     if args.use_metric:
#         loss_test = loss_test - args.metric_param * proto_dist

#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print(args.dataset, "Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))
    

def main():
    
    data_save_dir = '/cluster/tufts/hugheslab/prath01/datasets/mimic3_ssl/percentage_labelled_sequnces=100/'
    x_train_np_filename = os.path.join(data_save_dir, 'X_train.npy')
    x_valid_np_filename = os.path.join(data_save_dir, 'X_valid.npy')
    x_test_np_filename = os.path.join(data_save_dir, 'X_test.npy')
    y_train_np_filename = os.path.join(data_save_dir, 'y_train.npy')
    y_valid_np_filename = os.path.join(data_save_dir, 'y_valid.npy')
    y_test_np_filename = os.path.join(data_save_dir, 'y_test.npy')    


#     x_train_np_filename, y_train_np_filename = args.train_np_files.split(',')
#     x_test_np_filename, y_test_np_filename = args.test_np_files.split(',')
#     x_valid_np_filename, y_valid_np_filename = args.valid_np_files.split(',')
    
    
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
    
    _, T, D = X_test_labelled.shape
    for d in range(D):
        
        # impute missing values by mean filling using estimates from full training set
        fill_vals = np.nanmean(X_train[:, :, d])
        
        X_train_labelled[:, :, d] = np.nan_to_num(X_train_labelled[:, :, d], nan=fill_vals)
        X_valid_labelled[:, :, d] = np.nan_to_num(X_valid_labelled[:, :, d], nan=fill_vals)
        X_test_labelled[:, :, d] = np.nan_to_num(X_test_labelled[:, :, d], nan=fill_vals)
        X_train_unlabelled[:, :, d] = np.nan_to_num(X_train_unlabelled[:, :, d], nan=fill_vals)
        
        
        # min max normalization
#         den = np.nanmax(X_train[:, :, d])-np.nanmin(X_train[:, :, d])
#         X_train_labelled[:, :, d] = (X_train_labelled[:, :, d] - np.nanmin(X_train[:, :, d]))/den
#         X_valid_labelled[:, :, d] = (X_valid_labelled[:, :, d] - np.nanmin(X_train[:, :, d]))/den
#         X_test_labelled[:, :, d] = (X_test_labelled[:, :, d] - np.nanmin(X_train[:, :, d]))/den
#         X_train_unlabelled[:, :, d] = (X_train_unlabelled[:, :, d] - np.nanmin(X_train[:, :, d]))/den
        
        
#         # zscore normalization
        den = np.nanstd(X_train[:, :, d])
    
        X_train_labelled[:, :, d] = (X_train_labelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den
        X_valid_labelled[:, :, d] = (X_valid_labelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den
        X_test_labelled[:, :, d] = (X_test_labelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.sparse = True
    args.layers = [128, 300]
    args.kernels = [8, 5, 3]
    args.filters = [256, 256, 128]
    args.rp_params = [-1, 3]
    args.use_cnn = False
    args.use_lstm = True
    args.dilation = -1
    if not args.use_lstm and not args.use_cnn:
        print("Must specify one encoding method: --use_lstm or --use_cnn")
        print("Program Exiting.")
        exit(-1)

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))


    # Model and optimizer
    model_type = "TapNet" 
    
    N_tr = len(X_train_labelled)
    N_va = len(X_valid_labelled)
    N_te = len(X_test_labelled)    
    
    X_train_labelled = torch.tensor(X_train_labelled.reshape(N_tr, D, T))
    X_valid_labelled = torch.tensor(X_valid_labelled.reshape(N_va, D, T))
    X_test_labelled = torch.tensor(X_test_labelled.reshape(N_te, D, T))
    
    
    if model_type == "TapNet":
        
        # update random permutation parameter
        if args.rp_params[0] < 0:
            dim = X_train_labelled.shape[1]
            args.rp_params = [3, math.floor(dim / (3 / 2))]
        else:
            dim = X_train_labelled.shape[1]
            args.rp_params[1] = math.floor(dim / args.rp_params[1])

        args.rp_params = [int(l) for l in args.rp_params]
        print("rp_params:", args.rp_params)

        # update dilation parameter
        if args.dilation == -1:
            args.dilation = math.floor(X_train_labelled.shape[2] / 24)

        print("Data shape:", X_train_labelled.size())
        model = TapNet(nfeat=X_train_labelled.shape[1],
                       len_ts=X_train_labelled.shape[2],
                       layers=args.layers,
                       nclass=2,
                       dropout=0,
                       use_lstm=args.use_lstm,
                       use_cnn=args.use_cnn,
                       filters=args.filters,
                       dilation=args.dilation,
                       kernels=args.kernels,
                       use_metric=False,
                       use_rp=True,
                       rp_params=args.rp_params,
                       lstm_dim=128
                       )

    # init the optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    t_total = time.time()
    train(model, X_train_labelled, y_train_labelled, X_valid_labelled, y_valid_labelled, optimizer)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
              
    from IPython import embed; embed()
    # Testing
#     test()
    
    perf_dict = {'epoch': epoch,
                'train_loss' : train_loss,
                'labelled_train_loss' : train_loss_x,
                'unlabelled_train_loss' : train_loss_u,
                'valid_loss' : valid_loss,
                'test_loss' : test_loss,
                'train_auprc' : train_auprc,
                'valid_auprc' : valid_auprc,
                'test_auprc' : test_auprc,
                'train_auroc' : train_auroc,
                'valid_auroc' : valid_auroc,
                'test_auroc' : test_auroc,
                }
    print('-----------------------------Performance after epoch %s ------------------'%epoch)
    print(perf_dict)
    print('--------------------------------------------------------------------------')
    perf_dict_list.append(perf_dict)
    
    
    model_fname = os.path.join(args.output_dir, args.output_filename_prefix+'.pt')
    torch.save(model.state_dict(), model_fname)
    
    perf_df = pd.DataFrame(perf_dict_list)
    training_file = os.path.join(args.output_dir, args.output_filename_prefix+'.csv')
    print('Save model training performance to %s'%training_file)
    perf_df.to_csv(training_file, index=False)
    
    final_perf_save_file = os.path.join(args.output_dir, 'final_perf_'+args.output_filename_prefix+'.csv')
    final_model_perf_df = pd.DataFrame([{'train_AUPRC' : train_auprc, 
                                       'valid_AUPRC' : valid_auprc, 
                                       'test_AUPRC' : test_auprc,
                                       'train_AUROC' : train_auroc, 
                                       'valid_AUROC' : valid_auroc, 
                                       'test_AUROC' : test_auroc}])
    
    final_model_perf_df.to_csv(final_perf_save_file, index=False)
    

if __name__ == '__main__':
    main()
