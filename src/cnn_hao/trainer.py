import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tsPredCNNBinaryClassifier import tsPredCNNBinaryClassifier
from dataloader import create_datasets
import json

def train_test_model(config_path):
    if torch.cuda.is_available():  
        dev = torch.device("cuda:0")
    else:  
        dev = torch.device("cpu")
    with open(config_path) as f:
        setting = json.load(f)
        
    train_ds, val_ds, test_ds = create_datasets(
        setting['data_folder'] + 'x_train.csv', 
        setting['data_folder'] + 'y_train.csv',
        setting['data_folder'] + 'x_test.csv', 
        setting['data_folder'] + 'y_test.csv',
        setting['seq_id'],
        setting['x_columns'],
        setting['y_column'],
        ts_steps=setting['ts_steps']
    )
    
    train_dl = DataLoader(
        train_ds, batch_size = setting['bs'], 
        shuffle=True, num_workers=1)
    val_dl = DataLoader(
        val_ds, batch_size = setting['bs'] * 4, 
        shuffle=True, num_workers=1)
    test_dl = DataLoader(
        test_ds, batch_size = setting['bs'] * 4, 
        shuffle=False, num_workers=1)
    
    model = tsPredCNNBinaryClassifier(
        setting['channels'], 
        setting['kernels'], 
        setting['strides'], 
        setting['paddings'], 
        setting['pools'],
        setting['linear_layers']).to(dev)
    
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=setting['train_lr'], weight_decay=setting['train_l2'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25)
    model.train()
    
    train_losses = []
    train_aurocs = []
    val_losses = []
    val_aurocs = []
    for epoch in range(setting['train_epoch']):
        train_loss = 0
        train_target = []
        train_pred = []
        for batch_id, (x, y) in enumerate(train_dl):
            x = x.to(dev).float()
            y = y.to(dev).float()
            prediction = model(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item() /  setting['bs']
            train_target.append(y.cpu().detach().numpy())
            train_pred.append(prediction.cpu().detach().numpy())
        train_loss /= (batch_id + 1)
        train_target = np.concatenate(train_target)
        train_pred = np.concatenate(train_pred)
        train_auroc = roc_auc_score(train_target, train_pred)
        train_losses.append(train_loss)
        train_aurocs.append(train_auroc)
    #     train_acc /= (batch_id + 1)
        
        val_loss = 0
        val_target = []
        val_pred = []
        for batch_id, (x, y) in enumerate(val_dl):
            x = x.to(dev).float()
            y = y.to(dev).float()
            prediction = model(x)
            loss = loss_func(prediction, y)
            val_loss += loss.detach().item() /  setting['bs'] / 4
            val_target.append(y.cpu().detach().numpy())
            val_pred.append(prediction.cpu().detach().numpy())
        val_loss /= (batch_id + 1)
        val_target = np.concatenate(val_target)
        val_pred = np.concatenate(val_pred)
        val_auroc = roc_auc_score(val_target, val_pred)
        val_losses.append(val_loss)
        val_aurocs.append(val_auroc)
    
        inline_log = 'Epoch {} / {}, train_loss: {:.4f}, train_auroc: {:.3f}, val_loss: {:.4f}, val_auroc: {:.3f}'.format(
            epoch + 1, setting['train_epoch'], train_loss, train_auroc, val_loss, val_auroc
        )
        print(inline_log)
    
        scheduler.step()
        
    test_target = []
    test_pred = []
    for batch_id, (x, y) in enumerate(test_dl):
        x = x.to(dev).float()
        y = y.to(dev).float()
        prediction = model(x)
        test_target.append(y.cpu().detach().numpy())
        test_pred.append(prediction.cpu().detach().numpy())
    test_target = np.concatenate(test_target)
    test_pred = np.concatenate(test_pred)
    test_auroc = roc_auc_score(test_target, test_pred)
    
    print("Test AUROC: {:.3f}".format(test_auroc))
    
    return model, (train_losses, train_aurocs, val_losses, val_aurocs), test_auroc