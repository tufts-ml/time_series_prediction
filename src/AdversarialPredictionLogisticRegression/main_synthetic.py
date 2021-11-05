import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler
from ap_perf import PerformanceMetric, MetricLayer

import time
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler

from ap_perf import PerformanceMetric, MetricLayer


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[idx, :], self.y[idx]]

    
class LinearClassifier(nn.Module):
    def __init__(self, nvar):
        super().__init__()
        self.fc1 = nn.Linear(nvar, 1)
        self.double()
        
    def forward(self, x):
        x = self.fc1(x)
        return x.squeeze()



class RecallGvPrecision(PerformanceMetric):
    def __init__(self, th):
        self.th = th

    def define(self, C):
        return C.tp / C.ap  

    def constraint(self, C):
        return C.tp / C.pp  >= self.th   




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
    parser.add_argument('--lr', type=float, help='learning rate')    
    parser.add_argument('--weight_decay', type=float, help='penalty for weights')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--n_splits', type=int, default=2)
    args = parser.parse_args()
    
    # read the data dictionaries
    print('Reading train-test data...')
    
     
    x_train_csv, y_train_csv = args.train_csv_files.split(',')
    x_valid_csv, y_valid_csv = args.valid_csv_files.split(',')
    x_test_csv, y_test_csv = args.test_csv_files.split(',')
    outcome_col_name = args.outcome_col_name
    
    # Prepare data for classification
    x_train = pd.read_csv(x_train_csv).values
    y_train = pd.read_csv(y_train_csv)[outcome_col_name].values
    
    x_valid = pd.read_csv(x_valid_csv).values
    y_valid = pd.read_csv(y_valid_csv)[outcome_col_name].values
    
    x_test = pd.read_csv(x_test_csv).values
    y_test = pd.read_csv(y_test_csv)[outcome_col_name].values    
    
    # normalize data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_transformed = scaler.transform(x_train) 
    x_valid_transformed = scaler.transform(x_valid)
    x_test_transformed = scaler.transform(x_test)
    
    
    # store fixed validaiton set into dataset object object
#     valid_ds = Dataset(x_valid_transformed, y_valid) 
    
    # set random seed
    torch.manual_seed(args.seed)
    
    # set max_epochs 
    max_epochs=1000
    fixed_precision = 0.8
    
    recall_gv_precision_pm = RecallGvPrecision(fixed_precision)
    recall_gv_precision_pm.initialize()
    recall_gv_precision_pm.enforce_special_case_positive()
    recall_gv_precision_pm.set_cs_special_case_positive(True)
    
    
    trainset = TabularDataset(x_train_transformed, y_train)
    
    
    batch_size = x_train_transformed.shape[0] # full batch
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    method = "ap-perf"              # uncomment if we want to use ap-perf objective 

    torch.manual_seed(46364)#189
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    nvar = x_train_transformed.shape[1]
    model = LinearClassifier(nvar).to(device)

    criterion = MetricLayer(recall_gv_precision_pm).to(device)
    lr = args.lr
    weight_decay = args.weight_decay


    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    training_perf_dict_list = []
    for epoch in range(50): # epoch 2 was the best checkpoint after running 50 epochs
            # transform data
        t1 = time.time()

        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

        # evaluate after each epoch
        model.eval()
        
        # get valid and test loss
#         output_va = model(torch.from_numpy(x_valid_transformed))
#         loss_va = criterion(output_va, torch.from_numpy(y_valid))
        
        # train
        train_data = torch.tensor(x_train_transformed).to(device)
        tr_output = model(train_data)
        tr_pred = (tr_output >= 0.0).float()
        tr_pred_np = tr_pred.cpu().numpy()
        train_acc = np.sum(y_train == tr_pred_np) / len(y_train)
        train_metric = recall_gv_precision_pm.compute_metric(tr_pred_np, y_train)
        train_constraint = recall_gv_precision_pm.compute_constraints(tr_pred_np, y_train)
        
        # valid
        valid_data = torch.tensor(x_valid_transformed).to(device)
        va_output = model(valid_data)
        va_pred = (va_output >= 0.0).float()
        va_pred_np = va_pred.cpu().numpy()
        valid_acc = np.sum(y_valid == va_pred_np) / len(y_valid)
        valid_metric = recall_gv_precision_pm.compute_metric(va_pred_np, y_valid)
        valid_constraint = recall_gv_precision_pm.compute_constraints(va_pred_np, y_valid)
        
        # test
        test_data = torch.tensor(x_test_transformed).to(device)
        te_output = model(test_data)
        te_pred = (te_output >= 0.0).float()
        te_pred_np = te_pred.cpu().numpy()
        test_acc = np.sum(y_test == te_pred_np) / len(y_test)
        test_metric = recall_gv_precision_pm.compute_metric(te_pred_np, y_test)
        test_constraint = recall_gv_precision_pm.compute_constraints(te_pred_np, y_test)
        
        model.train()

        print('#{} loss : {:.2f}\nAcc tr: {:.5f} | Recall tr: {:.5f} | Precision tr: {:.5f} \nAcc va: {:.5f} | Recall va: {:.5f} | Precision va: {:.5f} \nAcc te: {:.5f} | Recall te: {:.5f} | Precision te: {:.5f}'.format(
            epoch, loss.detach().numpy()[0], train_acc, train_metric, train_constraint[0], valid_acc, valid_metric, valid_constraint[0], test_acc, test_metric, test_constraint[0]))
        
        t2 = time.time()
        time_taken = t2-t1
        print('time taken : {} seconds'.format(time_taken))
        
        training_perf_dict = {'epoch':epoch,
            'loss' : loss.detach().numpy()[0],
            'N_train':x_train_transformed.shape[0],
            'precision_train':train_constraint[0],
            'recall_train':train_metric,
            'N_valid': x_valid_transformed.shape[0],
            'precision_valid': valid_constraint[0],
            'recall_valid':valid_metric,
            'N_test':x_test_transformed.shape[0],
            'precision_test':test_constraint[0],
            'recall_test': test_metric,
            'time taken' : time_taken}
        
        training_perf_dict_list.append(training_perf_dict)
        
        if train_constraint[0]>=fixed_precision:
            print('Convergence Reached...')
            break
        

    
    training_perf_df = pd.DataFrame(training_perf_dict_list)
    train_output_csv = os.path.join(args.output_dir, args.output_filename_prefix+'training_hist.csv')
    print('Saved training history to : \n%s'%train_output_csv)
    
    training_perf_df.to_csv(train_output_csv, index=False)
    
    train_model_pt = os.path.join(args.output_dir, args.output_filename_prefix+'_model.pt')
    print('Saved trained model to : \n%s'%train_model_pt)
    torch.save(model.state_dict(), train_model_pt)
    
    
    # compute the TP's, FP's, TN's and FN's

    TP_train = np.sum(np.logical_and(y_train == 1, tr_pred_np == 1))
    FP_train = np.sum(np.logical_and(y_train == 0, tr_pred_np == 1))
    TN_train = np.sum(np.logical_and(y_train == 0, tr_pred_np == 0))
    FN_train = np.sum(np.logical_and(y_train == 1, tr_pred_np == 0))

    TP_valid = np.sum(np.logical_and(y_valid == 1, va_pred_np == 1))
    FP_valid = np.sum(np.logical_and(y_valid == 0, va_pred_np == 1))
    TN_valid = np.sum(np.logical_and(y_valid == 0, va_pred_np == 0))
    FN_valid = np.sum(np.logical_and(y_valid == 1, va_pred_np == 0))          

    TP_test = np.sum(np.logical_and(y_test == 1, te_pred_np == 1))
    FP_test = np.sum(np.logical_and(y_test == 0, te_pred_np == 1))
    TN_test = np.sum(np.logical_and(y_test == 0, te_pred_np == 0))
    FN_test = np.sum(np.logical_and(y_test == 1, te_pred_np == 0))    

    
    
    perf_dict = {'N_train':x_train_transformed.shape[0],
                'precision_train':train_constraint[0],
                'recall_train':train_metric,
                'TP_train':TP_train,
                'FP_train':FP_train,
                'TN_train':TN_train,
                'FN_train':FN_train,
                'N_valid': x_valid_transformed.shape[0],
                'precision_valid': valid_constraint[0],
                'recall_valid':valid_metric,
                'TP_valid':TP_valid,
                'FP_valid':FP_valid,
                'TN_valid':TN_valid,
                'FN_valid':FN_valid,
                'N_test':x_test_transformed.shape[0],
                'precision_test':test_constraint[0],
                'recall_test': test_metric,
                'TP_test':TP_test,
                'FP_test':FP_test,
                'TN_test':TN_test,
                'FN_test':FN_test,
                'Time taken' : time_taken}
    perf_df = pd.DataFrame([perf_dict])
    perf_csv = os.path.join(args.output_dir, args.output_filename_prefix+'_perf.csv')
    print('Saving performance on train and test set to %s'%(perf_csv))
    perf_df.to_csv(perf_csv, index=False)
