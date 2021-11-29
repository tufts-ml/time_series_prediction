# split_dataset.py

# Input:  --input: (required) a time-series file with one or more columns of
#              role 'id'
#         --data_dict: (required) data dictionary for that file
#         --test_size: (required) fractional size of the test set, expressed as
#              a number between 0 and 1
#         --output_dir: (required) directory where output files are saved
#         --group_cols: (optional) columns to group by, specified as a
#             space-separated list
#         Additionally, a seed used for randomization is hard-coded.
# Output: train.csv and test.csv, where grouping is by all specified columns,
#         or all columns of role 'id' if --group_cols is not specified.

import argparse
import json
import pandas as pd
import os
import numpy as np
import copy
import datetime
from sklearn.model_selection import GroupShuffleSplit
DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))
import sys
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from feature_transformation import (parse_id_cols, parse_time_cols, parse_feature_cols)
from utils import load_data_dict_json



def read_csv_with_float32_dtypes(filename):
    # Sample 100 rows of data to determine dtypes.
    df_test = pd.read_csv(filename, nrows=100)

    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    df = pd.read_csv(filename, dtype=float32_cols)
    
    return df



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--data_dict', type=str, required=True)
    parser.add_argument('--test_size', required=True, type=float)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--train_csv_filename', default='train.csv')
    parser.add_argument('--valid_csv_filename', default='valid.csv')
    parser.add_argument('--test_csv_filename', default='test.csv')
    parser.add_argument('--output_data_dict_filename', required=False, type=str, default=None)
#     parser.add_argument('--group_cols', nargs='*', default=[None])
#     parser.add_argument('--random_state', required=False, type=int, default=20190206)
    args = parser.parse_args()
    
    print('Creating train-test splits for %s'%args.input)
    # Import data
    df = read_csv_with_float32_dtypes(args.input)
    data_dict = load_data_dict_json(args.data_dict)
    
    '''
    # Split dataset
    if len(args.group_cols) == 0 or args.group_cols[0] is not None:
        group_cols = args.group_cols
    elif args.group_cols[0] is None:
        try:
            fields = data_dict['fields']
        except KeyError:
            fields = data_dict['schema']['fields']
        group_cols = [c['name'] for c in fields
                      if c['role'] in ('id', 'key') and c['name'] in df.columns]
    '''  
    # sort the dataframe by timestamp
    
    if 'window_start' in df.columns:
        df.rename(columns = {'window_start':'start',
                            'window_end' : 'stop'}, inplace=True)
    
    if 'start' in df.columns:
        df_timesorted = df.sort_values(by=['admission_timestamp', 'start', 'stop']) 
    else:
        id_cols = parse_id_cols(data_dict)       
        df_timesorted = df.sort_values(by=['admission_timestamp'] + id_cols + ['hours_since_admission']) 
    
    # set the first 3 years of admissions for training
    train_admission_ts_start = df_timesorted['admission_timestamp'].min()
    
    # set the 4th year as validation
    valid_admission_ts_start = str(pd.to_datetime(train_admission_ts_start) + datetime.timedelta(hours=24*365*3))
    
    # set the 5th year for test
    test_admission_ts_start = str(pd.to_datetime(train_admission_ts_start) + datetime.timedelta(hours=24*365*4))
    
    # split train, validation, test based on timestamps
    train_df = df_timesorted[(df_timesorted['admission_timestamp']>=train_admission_ts_start)&(df_timesorted['admission_timestamp']<valid_admission_ts_start)]
    
    valid_df = df_timesorted[(df_timesorted['admission_timestamp']>=valid_admission_ts_start)&(df_timesorted['admission_timestamp']<test_admission_ts_start)]
    
    test_df = df_timesorted[(df_timesorted['admission_timestamp']>=test_admission_ts_start)]
    
    # Write split data frames to CSV
    fdir_train_test = args.output_dir
    if fdir_train_test is not None:
        if not os.path.exists(fdir_train_test):
            os.mkdir(fdir_train_test)
        args.train_csv_filename = os.path.join(fdir_train_test, args.train_csv_filename)
        args.valid_csv_filename = os.path.join(fdir_train_test, args.valid_csv_filename)
        args.test_csv_filename = os.path.join(fdir_train_test, args.test_csv_filename)
        if args.output_data_dict_filename is not None:
            args.output_data_dict_filename = os.path.join(fdir_train_test, args.output_data_dict_filename)    
    
    if args.train_csv_filename[-3:] == '.gz':
        print('saving compressed train test files to :\n%s\n%s\n%s'%(args.train_csv_filename, args.valid_csv_filename,
                                                                     args.test_csv_filename))
        train_df.to_csv(args.train_csv_filename, index=False, compression='gzip')
        valid_df.to_csv(args.valid_csv_filename, index=False, compression='gzip')
        test_df.to_csv(args.test_csv_filename, index=False, compression='gzip')
    else:
        print('saving train test files to :\n%s\n%s\n%s'%(args.train_csv_filename, args.valid_csv_filename, args.test_csv_filename))
        train_df.to_csv(args.train_csv_filename, index=False)
        valid_df.to_csv(args.valid_csv_filename, index=False)
        test_df.to_csv(args.test_csv_filename, index=False)
    
    if args.output_data_dict_filename is not None:
        with open(args.output_data_dict_filename, 'w') as f:
            json.dump(data_dict, f, indent=4)

