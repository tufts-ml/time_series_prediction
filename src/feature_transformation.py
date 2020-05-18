# feature_transformation.py

# Input: Requires a dataframe, specification of whether to collapse
#        into summary statistics and which statistics, or whether to
#        add a transformation of a column and the operation with which
#        to transform

#        Requires a json data dictionary to accompany the dataframe,
#        data dictionary columns must all have a defined "role" field

# Output: Puts transformed dataframe into ts_transformed.csv and 
#         updated data dictionary into transformed.json


import sys
import pandas as pd
import argparse
import json
import numpy as np
from scipy import stats
import ast
import time
import copy

def main():
    parser = argparse.ArgumentParser(description="Script for collapsing"
                                                 "time features or adding"
                                                 "new features.")
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to csv dataframe of readings')
    parser.add_argument('--data_dict', type=str, required=True,
                        help='Path to json data dictionary file')
    parser.add_argument('--output', type=str, required=False, default=None)
    parser.add_argument('--data_dict_output', type=str, required=False, 
                        default=None)

    parser.add_argument('--collapse', default=False, action='store_true')
    parser.add_argument('--collapse_features', type=str, required=False,
                        default='count mean median std min max', 
                        help="Enclose options with 's, choose "
                             "from mean, std, min, max, "
                             "median, slope, count, present")
    parser.add_argument('--collapse_range_features', type=str, required=False,
                        default='slope std', 
                        help="Enclose options with 's, choose "
                             "from mean, std, min, max, "
                             "median, slope, count, present")
    parser.add_argument('--range_pairs', type=str, required=False,
                        default='[(0, 10), (0, 25), (0, 50), (50, 100), (75, 100), (90, 100), (0, 100)]',
                        help="Enclose pairs list with 's and [], list all desired ranges in "
                             "parentheses like this: '[(0, 50), (25, 75), (50, 100)]'")
    parser.add_argument('--max_time_step', type=int, required=False,
                        default=None, help="Specify the maximum number of time "
                                         "steps to collapse on, for example, "
                                         "input 48 for 48 hours at 1 hour time steps. "
                                         "Set to -1 for no limit.")

    # TODO: Add arithmetic opertions (ie column1 * column2 / column3)
    parser.add_argument('--add_feature', default=False, action='store_true')
    parser.add_argument('--add_from', type=str, required=False)
    parser.add_argument('--new_feature', type=str, required=False,
                        default='z-score', 
                        choices=['square', 'floor', 'int', 'sqrt', 
                                 'abs', 'z-score', 'ceiling', 'float'])

    args = parser.parse_args()

    args.data_dict_path = args.data_dict
    with open(args.data_dict_path, 'r') as f:
        args.data_dict = json.load(f)
    try:
        args.data_dict['fields'] = args.data_dict['schema']['fields']
    except KeyError:
        pass

    print('done parsing...')
    print('reading csv...')
    ts_df = pd.read_csv(args.input)
    print('done reading csv...')
    data_dict = None

    # transform data
    print('collapsing data..')
    t1 = time.time()
    if args.collapse:
        ts_df = collapse_np(ts_df, args)
        data_dict = update_data_dict_collapse(args)
    elif args.add_feature:
        ts_df = add_new_feature(ts_df, args)
        data_dict = update_data_dict_add_feature(args)
    t2 = time.time()
    print('done collapsing data..')
    print('time taken to collapse data : {} seconds'.format(t2-t1))
   
    # save data to file
    print('saving data to output...')
    if args.output is None:
        file_name = args.input.split('/')[-1].split('.')[0]
        data_output = '{}_transformed.csv'.format(file_name)
    elif args.output[-4:] == '.csv':
        data_output = args.output
    else:
        data_output = '{}.csv'.format(args.output)
    ts_df.to_csv(data_output, index=False)
    print("Wrote to output CSV:\n%s" % (data_output))

    # save data dictionary to file
    if args.data_dict_output is None:
        file_name = args.data_dict_path.split('/')[-1].split('.')[0]
        dict_output = '{}_transformed.json'.format(file_name)
    elif args.data_dict_output[-5:] == '.json':
        dict_output = args.data_dict_output
    else:
        dict_output = '{}.json'.format(args.data_dict_output)
    with open(dict_output, 'w') as f:
        json.dump(data_dict, f, indent=4)


def collapse_np(ts_df, args):
    id_cols = parse_id_cols(args.data_dict)
    id_cols = remove_col_names_from_list_if_not_in_df(id_cols, ts_df)
    feature_cols = parse_feature_cols(args.data_dict)
    feature_cols = remove_col_names_from_list_if_not_in_df(feature_cols, ts_df)
    time_col = parse_time_col(args.data_dict)

    # Obtain fenceposts based on where any key differs
    # Be sure keys are converted to a numerical datatype (so fencepost detection is possible)
    keys_df = ts_df[id_cols].copy()
    for col in id_cols:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    fp = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0).any(axis=1)), keys_df.shape[0]]) 

    list_of_collapsed_feat_arr = list()
    list_of_collapsed_feat_names = list()

    # Start timer
    total_time = 0

    timestamp_arr = np.asarray(ts_df[time_col].values.copy(), dtype=np.float64)
    features_arr = ts_df[feature_cols].values

    for op in args.collapse_range_features.split(' '):
        for low, high in ast.literal_eval(args.range_pairs):           
            print('Collapsing with func %s in %d to %d percentile range'%(op, low, high))
            t1 = time.time()
            # initialize collapsed dataframe for the current summary function
            n_rows = len(fp) - 1
            n_feats = len(feature_cols)
            collapsed_feat_arr = np.zeros([n_rows, n_feats])

            # loop through all the subject episode fenceposts
            for p in range(n_rows):
                # get the data for the current fencepost
                fp_start = fp[p]
                fp_end = fp[p+1]
                lower_bound, upper_bound = calc_start_and_stop_indices_from_percentiles(
                    timestamp_arr[fp_start:fp_end], start_percentile=low, end_percentile=high, max_timestamp=args.max_time_step)

                cur_feat_arr = features_arr[fp_start:fp_end,:].copy()
                cur_timestamp_arr = timestamp_arr[fp_start:fp_end]

                # compute summary function on that particular subject episode dataframe
                collapsed_feat_arr[p,:] = COLLAPSE_FUNCTIONS_np[op](cur_feat_arr, lower_bound, upper_bound, cur_timestamp_arr=cur_timestamp_arr)

            t2 = time.time()
            print('done in %d seconds'%(t2-t1)) 
            total_time = total_time + t2-t1 
            
            list_of_collapsed_feat_arr.append(collapsed_feat_arr)
            list_of_collapsed_feat_names.extend([x+'_'+op+'_'+str(low)+'_to_'+str(high) for x in feature_cols])

    print('-----------------------------------------')
    print('total time taken = %d seconds'%total_time)
    print('-----------------------------------------')
    collapsed_df = pd.DataFrame(np.hstack(list_of_collapsed_feat_arr), columns=list_of_collapsed_feat_names)

    for col_name in id_cols[::-1]:
        collapsed_df.insert(0, col_name, ts_df[col_name].values[fp[:-1]].copy())
    return collapsed_df


def calc_start_and_stop_indices_from_percentiles(timestamp_arr, start_percentile, end_percentile, max_timestamp=None):
    ''' Find start and stop indices to select specific percentile range
    
    Args
    ----
    timestamp_df : pd.DataFrame

    Returns
    -------
    lower_bound : int
        First index to consider in current sequence 
    upper_bound : int
        Last index to consider in current sequence

    Examples
    --------
    >>> timestamp_arr = np.arange(100)
    >>> calc_start_and_stop_indices_from_percentiles(timestamp_arr, 0, 10, None)
    (0, 10)
    >>> calc_start_and_stop_indices_from_percentiles(timestamp_arr, 25, 33, None)
    (25, 33)
    >>> calc_start_and_stop_indices_from_percentiles(timestamp_arr, 0, 0, 100)
    (0, 1)

    >>> timestamp_arr = np.asarray([0.7, 0.8, 0.9, 0.95, 0.99, 50.0, 98.1])
    >>> calc_start_and_stop_indices_from_percentiles(timestamp_arr, 0, 1, 100)
    (0, 5)
    >>> calc_start_and_stop_indices_from_percentiles(timestamp_arr, 1, 100, 100)
    (5, 7)
    '''

    # Treat the last data point as 100 percentile.
    if max_timestamp is None:
        max_timestamp = timestamp_arr[-1]
    lower_bound = np.searchsorted(timestamp_arr, max_timestamp * start_percentile/100)
    upper_bound = np.searchsorted(timestamp_arr, max_timestamp * (end_percentile + 0.001)/100)
    # if lower bound and upper bound are the same, add 1 to the upper bound
    if lower_bound >= upper_bound:
        upper_bound = lower_bound + 1
    assert upper_bound <= timestamp_arr.size + 1
    return int(lower_bound), int(upper_bound)

    # TODO handle this case??
    # Add nan values to list until it is the length of the max time step. 
    # Treat that as 100 percentile.
    #else: 
    #    data = data.append(pd.Series([np.nan for i in range(max_time_step - len(data))]),
    #                       ignore_index=True)
    #    lower_bound = (len(df)*start_percentile)//100
    #    upper_bound = (len(df)*end_percentile)//100


def all_id_combinations(cols, df, combos, ids=[]):
    if len(cols) == 0:
        combos.append(ids)
        return

    for i in df[cols[0]].unique():
        ids_copy = list(ids)
        ids_copy.append(i)
        all_id_combinations(cols[1:], df.loc[df[cols[0]] == i], 
                            combos, ids_copy)


# ADD NEW FEATURE COLUMN
def add_new_feature(ts_df, args):
    new_col_name = '{}_{}'.format(args.new_feature, args.add_from) 
    original_values = ts_df[args.add_from].tolist()
    new_values = None 

    if args.new_feature == 'z-score':
        new_values = stats.zscore(original_values)
    elif args.new_feature == 'square':
        new_values = np.square(original_values)
    elif args.new_feature == 'sqrt':
        new_values = np.sqrt(original_values)
    elif args.new_feature == 'floor':
        new_values = np.floor(original_values)
    elif args.new_feature == 'ceiling':
        new_values = np.ceil(original_values)
    elif args.new_feature == 'float':
        new_values = np.array(original_values).astype(float)
    elif args.new_feature == 'int':
        new_values = np.array(original_values).astype(int)
    elif args.new_feature == 'abs':
        new_values = np.absolute(original_values)

    ts_df[new_col_name] = new_values
    return ts_df


# DATA DICTIONARY STUFF: PARSING FUNCTIONS AND DICT UPDATING
def update_data_dict_collapse(args): 
    data_dict = args.data_dict

    id_cols = parse_id_cols(args.data_dict)
    feature_cols = parse_feature_cols(args.data_dict)

    new_fields = []
    for name in id_cols:
        for col in data_dict['fields']:
            if col['name'] == name: 
                new_fields.append(col)
    '''
    for op in args.collapse_features.split(' '):
        for name in feature_cols:
            for col in data_dict['fields']:
                if col['name'] == name: 
                    new_dict = dict(col)
                    new_dict['name'] = '{}_{}'.format(name, op)
                    new_fields.append(new_dict)
    '''
    for op in args.collapse_range_features.split(' '):
        for low, high in ast.literal_eval(args.range_pairs):
            for name in feature_cols:
                for col in data_dict['fields']:
                    if col['name'] == name: 
                        new_dict = dict(col)
                        new_dict['name'] = '{}_{}_{}_to_{}'.format(name, op, low, high)
                        new_fields.append(new_dict)

    new_data_dict = copy.deepcopy(data_dict)
    if 'schema' in new_data_dict:
        new_data_dict['schema']['fields'] = new_fields
        del new_data_dict['fields']
    else:
        new_data_dict['fields'] = new_fields

    return new_data_dict

def update_data_dict_add_feature(args): 
    data_dict = args.data_dict

    new_fields = []
    for col in data_dict['fields']:
        if 'name' in col and col['name'] == args.add_from:
            new_dict = dict(col)
            new_dict['name'] = '{}_{}'.format(args.new_feature, col['name'])
            new_fields.append(new_dict)
        else: 
            new_fields.append(col)

    new_data_dict = dict()
    new_data_dict['fields'] = new_fields
    return new_data_dict

def parse_id_cols(data_dict):
    cols = []
    for col in data_dict['fields']:
        if 'role' in col and (col['role'] == 'id' or 
                              col['role'] == 'key'):
            cols.append(col['name'])
    return cols

def parse_output_cols(data_dict):
    cols = []
    for col in data_dict['fields']:
        if 'role' in col and (col['role'] == 'outcome' or 
                              col['role'] == 'output'):
            cols.append(col['name'])
    return cols

def parse_feature_cols(data_dict):
    non_time_cols = []
    for col in data_dict['fields']:
        if 'role' in col and col['role'] in ('feature', 'measurement', 'covariate'):
            non_time_cols.append(col['name'])
    return non_time_cols

def parse_time_col(data_dict):
    for col in data_dict['fields']:
        # TODO avoid hardcoding a column name
        if (col['role'].count('time') or col['name'] == 'hours'):
            return col['name']
    

def remove_col_names_from_list_if_not_in_df(col_list, df):
    ''' Remove column names from provided list if not in dataframe

    Examples
    --------
    >>> df = pd.DataFrame(np.eye(3), columns=['a', 'b', 'c'])
    >>> remove_col_names_from_list_if_not_in_df(['q', 'c', 'a', 'e', 'f'], df)
    ['c', 'a']
    '''
    assert isinstance(col_list, list)
    for cc in range(len(col_list))[::-1]:
        col = col_list[cc]
        if col not in df.columns:
            col_list.remove(col)
    return col_list


# NEW collapse functions
def replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound, **kwargs):
    percentile_data_np = data_np[lower_bound:upper_bound,:]
    all_nan_col_ind = np.isnan(percentile_data_np).all(axis = 0)
    if sum(all_nan_col_ind)>0:
        percentile_data_np[:,all_nan_col_ind] = 0 
    return percentile_data_np

def collapse_mean_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)
    return np.nanmean(percentile_data_np, axis=0)
    
def collapse_median_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)  
    return np.nanmedian(percentile_data_np, axis=0)
    
def collapse_standard_dev_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)   
    return np.nanstd(percentile_data_np, axis=0)
    
def collapse_min_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)  
    return np.nanmin(percentile_data_np, axis=0)
    
def collapse_max_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)   
    return np.nanmax(percentile_data_np, axis=0)

def collapse_count_np(data_np, lower_bound, upper_bound, **kwargs):
    return (~np.isnan(data_np[lower_bound:upper_bound,:])).sum(axis=0)
 
def collapse_present_np(data_np, lower_bound, upper_bound, **kwargs):
    return (~np.isnan(data_np[lower_bound:upper_bound,:])).any(axis=0)

def collapse_slope_np(data_np, lower_bound, upper_bound, cur_timestamp_arr=None, **kwargs): 
    percentile_t_np = cur_timestamp_arr[lower_bound:upper_bound]
    percentile_data_np = data_np[lower_bound:upper_bound,:]
    n_cols = percentile_data_np.shape[1]
    collapsed_slope = np.zeros(n_cols)
    
    for col in range(n_cols):
        mask = ~np.isnan(percentile_data_np[:,col])
        if mask.sum():
            xs = percentile_data_np[mask,col]
            ts = percentile_t_np[mask]
            x_mean = np.mean(xs)
            ts -= np.mean(ts)
            xs -= x_mean
            numer = np.sum(ts * xs)
            denom = np.sum(np.square(xs))
            if denom == 0:
                collapsed_slope[col] = 0
            else:
                collapsed_slope[col] = numer/denom
        else:
            collapsed_slope[col] = 0
    return collapsed_slope  
    
COLLAPSE_FUNCTIONS_np = {
    "mean": collapse_mean_np,
    "std":  collapse_standard_dev_np,
    "median": collapse_median_np,
    "min": collapse_min_np,
    "max": collapse_max_np,
    "slope": collapse_slope_np, 
    "count": collapse_count_np,
    "present": collapse_present_np
}


if __name__ == '__main__':
    main()