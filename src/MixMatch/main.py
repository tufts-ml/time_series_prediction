from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import sys
import numpy as np
PROJECT_REPO_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'MixMatch', 'MixMatch-pytorch'))
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pandas as pd

# import torch.utils as utils
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

# import models.wideresnet as models
# import dataset.cifar10 as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
# from tensorboardX import SummaryWriter
PROJECT_SRC_DIR = '/cluster/tufts/hugheslab/prath01/projects/time_series_prediction/src/'
sys.path.append(PROJECT_SRC_DIR)
sys.path.append(os.path.join(PROJECT_SRC_DIR, "rnn"))
from RNNBinaryClassifierModule import RNNBinaryClassifierModule
from sklearn.metrics import (roc_auc_score, average_precision_score)



parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--train_np_files', type=str, required=True)
parser.add_argument('--test_np_files', type=str, required=True)
parser.add_argument('--valid_np_files', type=str, default=None)
parser.add_argument('--epochs', default=35, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=512, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
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
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--perc_labelled', default='33.3')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

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


def main():
    
#     data_save_dir = '/cluster/tufts/hugheslab/prath01/datasets/mimic3_ssl/percentage_labelled_sequnces=11.1/'
#     x_train_np_filename = os.path.join(data_save_dir, 'X_train.npy')
#     x_valid_np_filename = os.path.join(data_save_dir, 'X_valid.npy')
#     x_test_np_filename = os.path.join(data_save_dir, 'X_test.npy')
#     y_train_np_filename = os.path.join(data_save_dir, 'y_train.npy')
#     y_valid_np_filename = os.path.join(data_save_dir, 'y_valid.npy')
#     y_test_np_filename = os.path.join(data_save_dir, 'y_test.npy')    


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
    
    _, _, D = X_test_labelled.shape
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
        X_train_unlabelled[:, :, d] = (X_train_unlabelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den
    
    
    # get them into the dataloader
    train_data, train_label = torch.Tensor(X_train_labelled), torch.Tensor(y_train_labelled[:, np.newaxis])
    train_labeled_set = data.TensorDataset(train_data, train_label)
    
    valid_data, valid_label = torch.Tensor(X_valid_labelled), torch.Tensor(y_valid_labelled[:, np.newaxis])
    valid_labeled_set = data.TensorDataset(valid_data, valid_label)    

    test_data, test_label = torch.Tensor(X_test_labelled), torch.Tensor(y_test_labelled[:, np.newaxis])
    test_labeled_set = data.TensorDataset(test_data, test_label)  
    
    # set unlabelled data labels as -1 for now
    N_unlabelled_tr = len(X_train_unlabelled)
    dummy_labels_tr = -1*np.ones(N_unlabelled_tr)
    train_unlabelled_data, train_unlabelled_dummy_labels = torch.Tensor(X_train_unlabelled), torch.Tensor(dummy_labels_tr[:, np.newaxis])
    
    # section 3.1 of the MixMatch paper says for every unlabelled batch there needs to be an augmented version. Here I'm just adding noise as the augmented version (source : https://proceedings.neurips.cc/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Paper.pdf)
    train_unlabelled_aug_data = train_unlabelled_data + torch.randn_like(train_unlabelled_data)
    
    train_unlabeled_set = data.TensorDataset(train_unlabelled_data, 
                                             train_unlabelled_aug_data)  
    
    
    
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    
    val_loader = data.DataLoader(valid_labeled_set, batch_size=len(valid_labeled_set), shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_labeled_set, batch_size=len(test_labeled_set), shuffle=False, num_workers=0)

    # Model
    print("==> creating GRU")
    def create_model(ema=False):
#         model = models.WideResNet(num_classes=2)
        model = RNNBinaryClassifierModule(rnn_type='GRU', 
                                          n_inputs=D, 
                                          n_hiddens=32, 
                                          n_layers=1,
                                          dropout_proba=0.0, 
                                          dropout_proba_non_recurrent=0.0, 
                                          bidirectional=False)
#         model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

#     cudnn.benchmark = True
    print('Total params: %d' % (sum(p.numel() for p in model.parameters())))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    step = 0
    perf_dict_list = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
    
        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, use_cuda)
        train_loss, train_auprc, train_auroc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        valid_loss, valid_auprc, valid_auroc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_auprc, test_auroc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')

        step = args.train_iteration * (epoch + 1)

        # append logger file
    
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

        
def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

#     bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(len(labeled_trainloader)):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 2).scatter_(1, targets_x.view(-1,1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
#             p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            p = (outputs_u + outputs_u2) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        
        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/len(labeled_trainloader))
        
        loss = Lx + w * Lu
        
        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        print('({batch}/{size}) Data: {data:.3f}s | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(labeled_trainloader),
                    data=data_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    w=ws.avg,
                    ))

    return (losses.avg, losses_x.avg, losses_u.avg,)

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
#     bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            
            batch_size = inputs.size(0)

            # Transform label to one-hot
            targets = torch.zeros(batch_size, 2).scatter_(1, targets.view(-1,1).long(), 1)
            loss = criterion(outputs, targets[:, 1].long()) 
            
            roc_auc_valid = roc_auc_score(targets, outputs)
            auprc_valid = average_precision_score(targets, outputs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # plot progress
            print('({batch}/{size}) Data: {data:.3f}s | Loss: {loss:.4f} | AUROC: {auroc: .4f} | AUPRC: {auprc: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        loss=loss,
                        auroc=roc_auc_valid,
                        auprc=auprc_valid,
                        ))
    return (float(loss), auprc_valid, roc_auc_valid)

# def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
#     filepath = os.path.join(checkpoint, filename)
#     torch.save(state, filepath)
#     if is_best:
#         shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

if __name__ == '__main__':
    main()
