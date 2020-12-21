import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Model/dataloader definitions
class tsPredDataset(Dataset):
    def __init__(self, x_df, y_df, seq_id, x_cols, y_col, ts_steps=None):
        list_of_x = x_df[[seq_id] + x_cols] \
                    .groupby(seq_id) \
                    .apply(pd.DataFrame.to_numpy)
        if ts_steps is None:
            self.x = np.swapaxes(np.stack(list_of_x)[:, :, 1:], 1, 2).astype('float64')
        else:
            list_of_padded_x = []
            for i in list_of_x.keys():
                padded_x = np.zeros((ts_steps, len(x_cols) + 1))
                padded_x[:list_of_x[i].shape[0], :list_of_x[i].shape[1]] = list_of_x[i]
                list_of_padded_x.append(padded_x)
            self.x = np.swapaxes(np.stack(list_of_padded_x)[:, :, 1:], 1, 2).astype('float64')
        self.y = np.expand_dims(y_df[y_col].to_numpy().astype('float64'), axis = 1)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx, :, :], self.y[idx, :]

def load_train_val(x_path, y_path, seq_id, random_seed=None, train_p=0.8):
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    all_seq_id = pd.unique(x_df[seq_id])
    if random_seed is not None:
        np.random.seed(random_seed)
    train_msk = np.random.random(all_seq_id.shape)
    train_id = [i for i, msk in zip(all_seq_id, train_msk) if msk <= train_p]
    val_id = [i for i, msk in zip(all_seq_id, train_msk) if msk > train_p]
    
    return (
        x_df[x_df[seq_id].isin(train_id)].reset_index(drop=True), 
        y_df[y_df[seq_id].isin(train_id)].reset_index(drop=True),
        x_df[x_df[seq_id].isin(val_id)].reset_index(drop=True),
        y_df[y_df[seq_id].isin(val_id)].reset_index(drop=True)
    )

def load_test(x_path, y_path):
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path) 
    return (x_df, y_df)

def create_datasets(train_x_path, train_y_path, test_x_path, test_y_path,
                    seq_id, x_cols, y_col,
                   random_seed=None, train_p=0.8, ts_steps=None):
    train_x, train_y, val_x, val_y = load_train_val(
        train_x_path, train_y_path, seq_id, random_seed, train_p
    )
    test_x, test_y = load_test(test_x_path, test_y_path)
    train_ds = tsPredDataset(train_x, train_y, seq_id, x_cols, y_col, ts_steps=ts_steps)
    val_ds = tsPredDataset(val_x, val_y, seq_id, x_cols, y_col, ts_steps=ts_steps)
    test_ds = tsPredDataset(test_x, test_y, seq_id, x_cols, y_col, ts_steps=ts_steps)
    return train_ds, val_ds, test_ds