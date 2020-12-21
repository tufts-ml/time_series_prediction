import os
import numpy as np
import pandas as pd
import sys
import argparse
sys.path.append(os.path.join(os.path.abspath('../'), 'predictions_collapsed'))
from config_loader import PROJECT_REPO_DIR
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from utils import load_data_dict_json
from feature_transformation import (parse_id_cols, parse_feature_cols, parse_time_cols, get_fenceposts)
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
        deltas_arr = np.zeros((len(f_mask_arr),1))
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
        
        if col['name'] in feature_cols:
            for missing_feature in ['', 'mask_', 'delta_']:
                new_dict = dict(col)
                new_dict['name'] = '{}{}'.format(missing_feature, col['name'])
                new_fields.append(new_dict)
     

    new_data_dict = copy.deepcopy(x_data_dict)
    new_data_dict['schema']['fields'] = new_fields
    new_data_dict['fields'] = new_fields
        
    return new_data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_csv', type=str)
    parser.add_argument('--features_data_dict', type=str)
    args = parser.parse_args()
    
    # get the train test features
    features_df = pd.read_csv(args.features_csv)
     
    # impute values by carry forward and then pop mean on train and test sets separately
    features_data_dict = load_data_dict_json(args.features_data_dict)
    
    id_cols = parse_id_cols(features_data_dict)
    feature_cols = parse_feature_cols(features_data_dict)
    time_col = parse_time_cols(features_data_dict)
    
    # add mask features
    print('Adding missing values mask as features...')
    for feature_col in feature_cols:
        features_df.loc[:, 'mask_'+feature_col] = (~features_df[feature_col].isna())*1.0
    
    print('Adding time since last missing value is observed as features...')
    features_df = get_time_since_last_observed_features(features_df, id_cols, time_col, feature_cols)    
    
    # impute values
    print('Imputing missing values...')
    features_df = features_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()

    for feature_col in feature_cols:
        features_df[feature_col].fillna(features_df[feature_col].mean(), inplace=True)
    
    # Update data dict with missing value features
    new_data_dict = update_data_dict_with_mask_features(features_data_dict)    
    
    print('Saving imputed data to :\n %s'%(args.features_csv))
    features_df.to_csv(args.features_csv, index=False) 
    
    #save the new data dict
    print('Saving data dict with masking features to : \n%s'%args.features_data_dict)
    with open(args.features_data_dict, 'w') as f:
        json.dump(new_data_dict, f, indent=4)