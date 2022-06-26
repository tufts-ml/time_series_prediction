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

from sklearn.model_selection import GroupShuffleSplit

class Splitter:
    def __init__(self, n_splits=1, size=0, random_state=0, cols_to_group=None):
        self.n_splits = n_splits
        self.size = size
        self.cols_to_group = cols_to_group
        if hasattr(random_state, 'rand'):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(int(random_state))

    def make_groups_from_df(self, data_df):
        grp = data_df[self.cols_to_group]
        grp = [' '.join(row) for row in grp.astype(str).values]
        return grp

    def split(self, X, y=None, groups=None):
        gss1 = GroupShuffleSplit(random_state=copy.deepcopy(self.random_state), test_size=self.size, n_splits=self.n_splits)
        for tr_inds, te_inds in gss1.split(X, y=y, groups=groups):
            yield tr_inds, te_inds

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits

def split_dataframe_by_keys(data_df=None, size=0, random_state=0, cols_to_group=None):
    gss1 = Splitter(n_splits=1, size=size, random_state=random_state, cols_to_group=cols_to_group)
    for a, b in gss1.split(data_df, groups=gss1.make_groups_from_df(data_df)):
        train_df = data_df.iloc[a].copy()
        test_df = data_df.iloc[b].copy()
    return train_df, test_df

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
        
    scaling_df = pd.DataFrame(scaling_dict_list) 
    return normalized_df, scaling_df

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--data_dict', type=str, required=True)
    parser.add_argument('--test_size', required=True, type=float)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--train_csv_filename', default='train.csv')
    parser.add_argument('--valid_csv_filename', default=None)    
    parser.add_argument('--test_csv_filename', default='test.csv')
    parser.add_argument('--output_data_dict_filename', required=False, type=str, default=None)
    parser.add_argument('--group_cols', nargs='*', default=[None])
    parser.add_argument('--random_state', required=False, type=int, default=20190206)
    args = parser.parse_args()
    
    print('Creating train-test splits for %s'%args.input)
    # Import data
    df = pd.read_csv(args.input)
    data_dict = json.load(open(args.data_dict))

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
        
    train_df, test_df = split_dataframe_by_keys(
        df, cols_to_group=group_cols, size=args.test_size, random_state=args.random_state)
    
    if args.valid_csv_filename is not None:
        train_df, valid_df = split_dataframe_by_keys(
            train_df, cols_to_group=group_cols, size=args.test_size, random_state=args.random_state)    
    
    # Write split data frames to CSV
    fdir_train_test = args.output_dir
    if fdir_train_test is not None:
        if not os.path.exists(fdir_train_test):
            os.mkdir(fdir_train_test)
        args.train_csv_filename = os.path.join(fdir_train_test, args.train_csv_filename)
        args.test_csv_filename = os.path.join(fdir_train_test, args.test_csv_filename)
        
        if args.valid_csv_filename is not None:
            args.valid_csv_filename = os.path.join(fdir_train_test, args.valid_csv_filename)
            
        if args.output_data_dict_filename is not None:
            args.output_data_dict_filename = os.path.join(fdir_train_test, args.output_data_dict_filename)    
    
    if args.train_csv_filename[-3:] == '.gz':
        print('saving compressed train test files to :\n%s\n%s\n%s'%(args.train_csv_filename, args.test_csv_filename, args.valid_csv_filename))
        train_df.to_csv(args.train_csv_filename, index=False, compression='gzip')
        if args.valid_csv_filename is not None:
            valid_df.to_csv(args.valid_csv_filename, index=False, compression='gzip')
        
        test_df.to_csv(args.test_csv_filename, index=False, compression='gzip')
    else:
        print('saving train test files to :\n%s\n%s\n%s'%(args.train_csv_filename, args.test_csv_filename, args.valid_csv_filename))
        train_df.to_csv(args.train_csv_filename, index=False)
        if args.valid_csv_filename is not None:
            valid_df.to_csv(args.valid_csv_filename, index=False)
        test_df.to_csv(args.test_csv_filename, index=False)
    
    if args.output_data_dict_filename is not None:
        with open(args.output_data_dict_filename, 'w') as f:
            json.dump(data_dict, f, indent=4)