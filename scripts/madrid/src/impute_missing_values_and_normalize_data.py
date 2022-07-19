import os
import numpy as np
import pandas as pd
import sys
import argparse
sys.path.append(os.path.join(os.path.abspath('../'), 'predictions_collapsed'))
from config_loader import PROJECT_REPO_DIR
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from utils import load_data_dict_json
from feature_transformation import (parse_id_cols, parse_feature_cols, parse_time_col, get_fenceposts)
from progressbar import ProgressBar
import copy
import json

def get_time_since_last_observed_features(df, id_cols, time_col, feature_cols):
    fp = get_fenceposts(df, id_cols)
    n_fences = len(fp)-1
    t_arr = np.asarray(df[time_col].values.copy(), dtype=np.float64) 
    pbar = ProgressBar()
    for feature_col in pbar(feature_cols):
        f_mask_arr = df['mask_'+feature_col].values
        deltas_arr = np.zeros_like(t_arr)
        for p in range(n_fences):
            curr_stay_times = t_arr[fp[p]:fp[p+1]]
            curr_stay_feature_mask = f_mask_arr[fp[p]:fp[p+1]]
            deltas_arr[fp[p]:fp[p+1]] = compute_time_since_last_observed(curr_stay_times, curr_stay_feature_mask)
        df.loc[:, 'delta_'+feature_col] = deltas_arr
    return df
    
def compute_time_since_last_observed(timestamp_arr, mask_arr):
    deltas_arr = np.zeros_like(timestamp_arr)
    for ind in range(len(mask_arr)):
        if ind>0:
            if mask_arr[ind-1]==0:
                deltas_arr[ind] = timestamp_arr[ind] - timestamp_arr[ind-1] + deltas_arr[ind-1]
            elif mask_arr[ind-1]==1:
                deltas_arr[ind] = timestamp_arr[ind] - timestamp_arr[ind-1]
    return deltas_arr

def update_data_dict_with_mask_features(x_data_dict):
    # add the new features to the data dict
    new_fields=[]
    for col in x_data_dict['schema']['fields']:
        if col['name']==time_col:
            new_fields.append(dict(col))
        
        if col['name'] in id_cols:
            new_fields.append(dict(col))
        
        if col['name'] in non_medication_feature_cols:
            for missing_feature in ['', 'mask_', 'delta_']:
                new_dict = dict(col)
                new_dict['name'] = '{}{}'.format(missing_feature, col['name'])
                new_fields.append(new_dict)
        elif col['name'] in medication_feature_cols:
            new_fields.append(dict(col))
     

    new_data_dict = copy.deepcopy(x_data_dict)
    new_data_dict['schema']['fields'] = new_fields
    new_data_dict['fields'] = new_fields
        
    return new_data_dict

def normalize_df(df, feature_cols, scaling='minmax', train_df=None):
    ''' Normalizes the dataframe if needed. Useful to normalize train and test sets separately after splitting dataset'''
    if train_df is None:
        train_df = df.copy()
    
    
    scaling_dict_list = []
    normalized_df = df.copy()
    
    if scaling=='zscore':
        for col in feature_cols:
            den_scaling = train_df[col].std()
            num_scaling = train_df[col].mean()
            
            if den_scaling==0:
                den_scaling = 1
            
            # scale the data
            normalized_df[col] = (df[col]-num_scaling)/den_scaling
            
            # store the normalization estimates in a list and save them for later evaluation
            scaling_dict_list.append({'feature':col, 'numerator_scaling':num_scaling, 
                                      'denominator_scaling':den_scaling})
            
    elif scaling=='minmax':
        for col in feature_cols:
            den_scaling = train_df[col].max()-train_df[col].min()
            num_scaling = train_df[col].min()
            
            if den_scaling==0:
                den_scaling = 1  
                
            # scale the data
            normalized_df[col] = (df[col]-num_scaling)/den_scaling
            
            # store the normalization estimates in a list and save them for later evaluation
            scaling_dict_list.append({'feature':col, 'numerator_scaling':num_scaling, 
                                      'denominator_scaling':den_scaling})
        
        
    elif scaling=='minmax_with_outliers':
         for col in feature_cols:
            den_scaling = np.nanpercentile(train_df[col], 99)-train_df[col].min()
            num_scaling = train_df[col].min()
            
            if den_scaling==0:
                den_scaling = 1  
                
            # scale the data
            normalized_df[col] = (df[col]-num_scaling)/den_scaling
            
            # store the normalization estimates in a list and save them for later evaluation
            scaling_dict_list.append({'feature':col, 'numerator_scaling':num_scaling, 
                                      'denominator_scaling':den_scaling})
    scaling_df = pd.DataFrame(scaling_dict_list) 
    return normalized_df, scaling_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test_split_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--x_train_csv', type=str)
    parser.add_argument('--x_valid_csv', type=str)
    parser.add_argument('--x_test_csv', type=str)
    parser.add_argument('--x_data_dict', type=str)
    parser.add_argument('--normalization', type=str, default='minmax')
    args = parser.parse_args()
    
    # get the train test features
#     x_train_csv=os.path.join(args.train_test_split_dir, 'x_train.csv.gz')
#     x_valid_csv=os.path.join(args.train_test_split_dir, 'x_valid.csv.gz')
#     x_test_csv=os.path.join(args.train_test_split_dir, 'x_test.csv.gz')
#     x_dict_json=os.path.join(args.train_test_split_dir, 'x_dict.json')

    x_train_csv=args.x_train_csv
    x_valid_csv=args.x_valid_csv
    x_test_csv=args.x_test_csv
    x_dict_json=args.x_data_dict
    
    # impute values by carry forward and then pop mean on train and test sets separately
    x_data_dict = load_data_dict_json(x_dict_json)
    x_train_df = pd.read_csv(x_train_csv)
    x_valid_df = pd.read_csv(x_valid_csv)
    x_test_df = pd.read_csv(x_test_csv)
    
    id_cols = parse_id_cols(x_data_dict)
    feature_cols = parse_feature_cols(x_data_dict)
    time_col = 'stop'#parse_time_col(x_data_dict)
    
    # add mask features
    non_medication_feature_cols = [feature_col for feature_col in feature_cols if 'medication' not in feature_col]
    medication_feature_cols = [feature_col for feature_col in feature_cols if 'medication' in feature_col]
    
    print('Adding missing values mask as features...')
    for feature_col in non_medication_feature_cols:
        x_train_df.loc[:, 'mask_'+feature_col] = (~x_train_df[feature_col].isna())*1.0
        x_valid_df.loc[:, 'mask_'+feature_col] = (~x_valid_df[feature_col].isna())*1.0
        x_test_df.loc[:, 'mask_'+feature_col] = (~x_test_df[feature_col].isna())*1.0
    print('Adding time since last missing value is observed as features...')
    x_train_df = get_time_since_last_observed_features(x_train_df, id_cols, time_col, non_medication_feature_cols)
    x_valid_df = get_time_since_last_observed_features(x_valid_df, id_cols, time_col, non_medication_feature_cols)
    x_test_df = get_time_since_last_observed_features(x_test_df, id_cols, time_col, non_medication_feature_cols)
    
    
    # impute values
    print('Imputing values in train, valid and test sets by forward filling, and then with training set median..')
    x_train_df = x_train_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()
    x_valid_df = x_valid_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()
    x_test_df = x_test_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()
    for feature_col in feature_cols:
        x_train_df[feature_col].fillna(x_train_df[feature_col].median(), inplace=True)
        
        # impute population mean of training set to test set
        x_valid_df[feature_col].fillna(x_train_df[feature_col].median(), inplace=True)
        x_test_df[feature_col].fillna(x_train_df[feature_col].median(), inplace=True)
    
    # Update data dict with missing value features
    new_data_dict = update_data_dict_with_mask_features(x_data_dict)
    
    # get feature columns with mask features
    feature_cols_with_mask_features = parse_feature_cols(new_data_dict)
    
    # normalize the data
    x_train_normalized_df, scaling_df = normalize_df(x_train_df, feature_cols_with_mask_features, scaling=args.normalization)
    x_valid_normalized_df, scaling_df = normalize_df(x_valid_df, feature_cols_with_mask_features, scaling=args.normalization, train_df=x_train_df)
    x_test_normalized_df, scaling_df = normalize_df(x_test_df, feature_cols_with_mask_features, scaling=args.normalization, train_df=x_train_df)
    
    
    print('Saving imputed and normalized train-valid-test splits to :\n%s \n%s \n%s'%(x_train_csv, x_valid_csv, x_test_csv))
    x_train_normalized_df.to_csv(x_train_csv, index=False, compression='gzip') 
    x_valid_normalized_df.to_csv(x_valid_csv, index=False, compression='gzip') 
    x_test_normalized_df.to_csv(x_test_csv, index=False, compression='gzip')
    
    norm_estimates_csv = os.path.join(args.output_dir, 'normalization_estimates.csv') 
    print('Saving normalization estimates to : \n%s'%norm_estimates_csv)
    scaling_df.to_csv(norm_estimates_csv, index=False)
    
    #save the new data dict
    print('Saving data dict with masking features to : \n%s'%x_dict_json)
    with open(x_dict_json, 'w') as f:
        json.dump(new_data_dict, f, indent=4)
