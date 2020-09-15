import os
import numpy as np
import pandas as pd
import sys
import argparse
sys.path.append(os.path.join(os.path.abspath('../'), 'predictions_collapsed'))
from config_loader import PROJECT_REPO_DIR
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from utils import load_data_dict_json
from feature_transformation import (parse_id_cols, parse_feature_cols)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test_split_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    
    # get the train test features
    x_train_csv=os.path.join(args.train_test_split_dir, 'x_train.csv')
    x_test_csv=os.path.join(args.train_test_split_dir, 'x_test.csv')
    x_dict_json=os.path.join(args.train_test_split_dir, 'x_dict.json')
    
    # impute values by carry forward and then pop mean on train and test sets separately
    x_data_dict = load_data_dict_json(x_dict_json)
    x_train_df = pd.read_csv(x_train_csv)
    x_test_df = pd.read_csv(x_test_csv)
    
    id_cols = parse_id_cols(x_data_dict)
    feature_cols = parse_feature_cols(x_data_dict)
    
    # impute values
    print('Imputing values in train and test sets separately..')
    x_train_df = x_train_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()
    x_test_df = x_test_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()
    for feature_col in feature_cols:
        x_train_df[feature_col].fillna(x_train_df[feature_col].mean(), inplace=True)
        
        # impute population mean of training set to test set
        x_test_df[feature_col].fillna(x_train_df[feature_col].mean(), inplace=True)
    
    print('Saving imputed data to :\n%s \n%s'%(x_train_csv, x_test_csv))
    x_train_df.to_csv(x_train_csv, index=False) 
    x_test_df.to_csv(x_test_csv, index=False)