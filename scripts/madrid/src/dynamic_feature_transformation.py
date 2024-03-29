# dynamic_feature_transformation.py

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
import os
from scipy.stats import skew
DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from utils import load_data_dict_json
from progressbar import ProgressBar

def get_fenceposts(ts_df, id_cols):
    keys_df = ts_df[id_cols].copy()
    for col in id_cols:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    fp = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0).any(axis=1)), keys_df.shape[0]]) 
    return fp

def main():
    parser = argparse.ArgumentParser(description="Script for collapsing"
                                                 "time features or adding"
                                                 "new features.")
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to csv dataframe of readings')
    parser.add_argument('--data_dict', type=str, required=True,
                        help='Path to json data dictionary file')
    parser.add_argument('--outcomes', type=str, required=True, 
                        help='Path to csv dataframe of outcomes')
    parser.add_argument('--data_dict_outcomes', type=str, required=True,
                        help='Path to json data dictionary file for outcomes')
    parser.add_argument('--dynamic_collapsed_features_csv', type=str, required=False, default=None)
    parser.add_argument('--dynamic_collapsed_features_data_dict', type=str, required=False, default=None)
    parser.add_argument('--dynamic_outcomes_csv', type=str, required=False, default=None)    
#     parser.add_argument('--dynamic_outcomes_data_dict', type=str, required=False, default=None) 
    parser.add_argument('--collapse_features', type=str, required=False,
                        default='count mean median std min max', 
                        help="Enclose options with 's, choose "
                             "from mean, std, min, max, "
                             "median, slope, count, present")
    parser.add_argument('--collapse_range_features', type=str, required=False,
                        default='slope std', 
                        help="Enclose options with 's, choose "
                             "from mean, std, min, max, "
                             "median, slope, count, present, skew, hours_since_measured")
    parser.add_argument('--range_pairs', type=str, required=False,
                        default='[(0, 10), (0, 25), (0, 50), (50, 100), (75, 100), (90, 100), (0, 100)]',
                        help="Enclose pairs list with 's and [], list all desired ranges in "
                             "parentheses like this: '[(0, 50), (25, 75), (50, 100)]'")


    args = parser.parse_args()

    print('reading features...')
    ts_df = pd.read_csv(args.input)
    data_dict = load_data_dict_json(args.data_dict)
    print('done reading features...')
    
    print('reading outcomes...')
    outcomes_df = pd.read_csv(args.outcomes)
    data_dict_outcomes = load_data_dict_json(args.data_dict_outcomes)
    print('done reading outcomes...')    

    # transform data
    t1 = time.time()
    dynamic_collapsed_df, dynamic_outcomes_df = collapse_dynamic(ts_df=ts_df, data_dict=data_dict,                                           
                                                                 collapse_range_features=args.collapse_range_features, 
                                                                 range_pairs=args.range_pairs, outcomes_df=outcomes_df, 
                                                                 data_dict_outcomes=data_dict_outcomes)
    
    
    dynamic_collapsed_features_data_dict = update_data_dict_collapse(data_dict, args.collapse_range_features, args.range_pairs)
    t2 = time.time()
    print('done collapsing data..')
    print('time taken to collapse data : {} seconds'.format(t2-t1))
    
    # save data to file
    dynamic_collapsed_df.to_csv(args.dynamic_collapsed_features_csv, index=False, compression='gzip')
    print('Saved dynamic collapsed features to :\n%s'%args.dynamic_collapsed_features_csv)
    
    dynamic_outcomes_df.to_csv(args.dynamic_outcomes_csv, index=False, compression='gzip')
    print('Saved dynamic outcomes to :\n%s'%args.dynamic_outcomes_csv)
    
    
    # save data dictionary to file
    with open(args.dynamic_collapsed_features_data_dict, 'w') as f:
        json.dump(dynamic_collapsed_features_data_dict, f, indent=4)

    print ('Saved dynamic collapsed features dict to :\n%s'%args.dynamic_collapsed_features_data_dict)
    

def collapse_dynamic(ts_df, data_dict, collapse_range_features, range_pairs, outcomes_df, data_dict_outcomes):
    id_cols = parse_id_cols(data_dict)
    id_cols = remove_col_names_from_list_if_not_in_df(id_cols, ts_df)

    feature_cols = parse_feature_cols(data_dict)
    feature_cols = remove_col_names_from_list_if_not_in_df(feature_cols, ts_df)

    time_cols = parse_time_cols(data_dict)
    time_cols = remove_col_names_from_list_if_not_in_df(time_cols, ts_df)
    
    if len(time_cols) == 0:
        raise ValueError("Expected at least one variable with role='time'")
    elif len(time_cols) > 1:
#         raise ValueError("More than one time variable found. Expected exactly one.")
          print("More than one time variable found. Choosing %s"%time_cols[-1])
    time_col = time_cols[-1]

    # Obtain fenceposts based on where any key differs
    # Be sure keys are converted to a numerical datatype (so fencepost detection is possible)
    keys_df = ts_df[id_cols].copy()
    for col in id_cols:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    fp = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0).any(axis=1)), keys_df.shape[0]]) 
    
    list_of_dynamic_collapsed_feat_arr = list()
    list_of_collapsed_feat_names = list()
    
    # define outcome column (TODO : Avoid hardcording by loading from config.json)
    outcome_col = 'clinical_deterioration_outcome'
    
    # Start timer
    total_time = 0
    #ts_df[feature_cols] = ts_df[feature_cols].astype(np.float32)
    timestamp_arr = np.asarray(ts_df[time_col].values.copy(), dtype=np.float32)
    features_arr = ts_df[feature_cols].values
    ids_arr = ts_df[id_cols].values
#     ts_with_max_tstop_df = ts_df[id_cols + [time_col]].groupby(id_cols, as_index=False).max().rename(columns={time_col:'max_tstop'})
#     tstops_arr = np.asarray(pd.merge(ts_df, ts_with_max_tstop_df, on=id_cols, how='left')['max_tstop'], dtype=np.float32)
    prediction_window = 12
    prediction_horizon = 24
    max_hrs_data_observed = 504
    t_start=-24 # start time 
    dynamic_collapsed_feat_id_list = list()
    dynamic_outcomes_list = list()
    dynamic_window_list = list()
    dynamic_stay_lengths_list = list()
    for op_ind, op in enumerate(collapse_range_features.split(' ')):
        print('Collapsing with func %s'%op)
        t1=time.time()
        
        list_of_dynamic_collapsed_feat_cur_op = list()
        for rp_ind, (low, high) in enumerate(ast.literal_eval(range_pairs)):           
            #print('Collapsing with func %s in %d to %d percentile range'%(op, low, high))
            #t1 = time.time()
            # initialize collapsed dataframe for the current summary function
            n_rows = len(fp) - 1
            n_feats = len(feature_cols)
            
            list_of_dynamic_collapsed_feat_cur_op_range_pair = list()
            
            # loop through all the subject episode fenceposts
            empty_arrays = 0
            pbar=ProgressBar()
            
            # potentially extract the list of all unique ids and avoid using merge
            
            for p in pbar(range(n_rows)):
#             for p in pbar(range(100)):
                # Get features and times for the current fencepost
                fp_start = fp[p]
                fp_end = fp[p+1]
                cur_feat_arr = features_arr[fp_start:fp_end,:].copy()
                cur_timestamp_arr = timestamp_arr[fp_start:fp_end]
#                 cur_tstops_arr = tstops_arr[fp_start:fp_end]
                
                # get the current stay id (Do this outside the loop)
                cur_id_df = ts_df[id_cols].iloc[fp[p]:fp[p+1]].drop_duplicates(subset=id_cols)
                
                # get the stay length of the current 
                cur_outcomes_df = pd.merge(outcomes_df, cur_id_df, on=id_cols, how='inner')
                cur_stay_length = cur_outcomes_df['stay_length'].values[0]
                cur_final_outcome = int(cur_outcomes_df[outcome_col].values[0])
                
                # create windows from start to length of stay (0-prediction_window, 0-2*prediction_window, ... 0-length_of_stay)
                t_end = min(cur_stay_length, max_hrs_data_observed)
                window_ends = np.arange(t_start+prediction_window, t_end+prediction_window, prediction_window)
                
                # collapse features in each window
                cur_dynamic_collapsed_feat_arr = np.zeros([len(window_ends), n_feats], dtype=np.float32)
#                 cur_dynamic_collapsed_feat_list = list()
                
                for q, window_end in enumerate(window_ends):
                    cur_dynamic_idx = (cur_timestamp_arr>t_start)&(cur_timestamp_arr<=window_end)
                    cur_dynamic_timestamp_arr = cur_timestamp_arr[cur_dynamic_idx]
                    cur_dynamic_feat_arr = cur_feat_arr[cur_dynamic_idx]
                    
                     
                    if len(cur_dynamic_timestamp_arr)>0:
                        start, stop = calc_start_and_stop_indices_from_percentiles(
                            cur_dynamic_timestamp_arr,
                            start_percentile=int(low[:-1]),
                            end_percentile=int(high[:-1]))
                            
                        cur_dynamic_collapsed_feat_arr[q,:] = COLLAPSE_FUNCTIONS_np[op](cur_dynamic_feat_arr, start, stop,
                                                                                        cur_timestamp_arr=cur_dynamic_timestamp_arr)
#                         cur_dynamic_collapsed_feat_list.append(COLLAPSE_FUNCTIONS_np[op](cur_dynamic_feat_arr, start, stop,
#                                                                                          cur_timestamp_arr=cur_dynamic_timestamp_arr))
                    
                    if (op_ind==0) & (rp_ind==0):
                        # keep track of stay ids
                        dynamic_collapsed_feat_id_list.append(cur_id_df.values[0])

                        # keep track of windows
                        dynamic_window_list.append(np.array([t_start, window_end]))

                        # keep track of the stay lengths
                        dynamic_stay_lengths_list.append(cur_stay_length)

                        # if the length of stay is within the prediction horizon, set the outcome as the clinical deterioration outcome, else set 0
                        if window_end>=cur_stay_length-prediction_horizon:
                            dynamic_outcomes_list.append(cur_final_outcome)
                        else:
                            dynamic_outcomes_list.append(0)
                
#             if op_ind==0:
#                 print('Percentage of empty slices in %s to %s is %.2f'%(
#                     low, high, (empty_arrays/n_rows)*100))
                
                list_of_dynamic_collapsed_feat_cur_op_range_pair.append(cur_dynamic_collapsed_feat_arr)
                               
            # vertically stack collapsed features for this op and range pair across all stays
            dynamic_collapsed_feat_cur_op_range_pair = np.vstack(list_of_dynamic_collapsed_feat_cur_op_range_pair)
            
            # get the collapsed feature for all range pairs for current op
            list_of_dynamic_collapsed_feat_cur_op.append(dynamic_collapsed_feat_cur_op_range_pair)
#             del dynamic_collapsed_feat_cur_op_range_pair
            
            # keep track of the collapsed feature op range pair
            list_of_collapsed_feat_names.extend([x+'_'+op+'_'+str(low)+'_to_'+str(high) for x in feature_cols])
        
        # horizontally stack across all ops range pairs for current op
        list_of_dynamic_collapsed_feat_arr.append(np.hstack(list_of_dynamic_collapsed_feat_cur_op))
        t2 = time.time()
        print('done in %d seconds'%(t2-t1))
        total_time = total_time + t2-t1
    print('-----------------------------------------')
    print('total time taken to collapse features = %d seconds'%total_time)
    print('-----------------------------------------')
    
    # horizontally stack across all ops
    dynamic_collapsed_df = pd.DataFrame(np.hstack(list_of_dynamic_collapsed_feat_arr), columns=list_of_collapsed_feat_names)
    
    # add the ids back to the collapsed features
    ids_df = pd.DataFrame(dynamic_collapsed_feat_id_list, columns=id_cols) 
    
    # add the window start and ends
    dynamic_window_df = pd.DataFrame(np.vstack(dynamic_window_list), columns=['window_start','window_end'])
    dynamic_stay_lengths_df = pd.DataFrame(np.vstack(dynamic_stay_lengths_list), columns=['stay_length'])
    
    dynamic_collapsed_df = pd.concat([ids_df, dynamic_collapsed_df, dynamic_window_df], axis=1)
    
    dynamic_outcomes_df = pd.DataFrame(np.array(dynamic_outcomes_list), columns=[outcome_col])
    dynamic_outcomes_df = pd.concat([ids_df, dynamic_outcomes_df, dynamic_window_df, dynamic_stay_lengths_df], axis=1)  
    
    return dynamic_collapsed_df, dynamic_outcomes_df

def calc_start_and_stop_indices_from_percentiles(
        timestamp_arr, start_percentile, end_percentile, max_timestamp=None):
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

    # If we want the first 1% of the sequence, we'd get first 5 values
    >>> calc_start_and_stop_indices_from_percentiles(timestamp_arr, 0, 1, 100)
    (0, 5)

    # If we want the slice from 1% to 100%, we'd get the last 2 values
    >>> calc_start_and_stop_indices_from_percentiles(timestamp_arr, 1, 100, 100)
    (5, 7)
    '''

    # Consider last data point as first timestamp + step size input by the user. For eg. If the step size is 1hr, then consider only 
    # first hour of time series data
    
    min_timestamp = timestamp_arr[0]
    if max_timestamp is None:
        max_timestamp = timestamp_arr[-1]
    max_timestamp = np.minimum(max_timestamp, timestamp_arr[-1])
    
    lower_tstamp = (min_timestamp +
        (max_timestamp - min_timestamp) * float(start_percentile) / 100)
    lower_bound = np.searchsorted(timestamp_arr, lower_tstamp)

    upper_tstamp = (min_timestamp +
        (max_timestamp - min_timestamp) * float(end_percentile) / 100)
    upper_bound = np.searchsorted(timestamp_arr, upper_tstamp)
    
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



# DATA DICTIONARY STUFF: PARSING FUNCTIONS AND DICT UPDATING
def update_data_dict_collapse(data_dict, collapse_range_features, range_pairs): 

    id_cols = parse_id_cols(data_dict)
    feature_cols = parse_feature_cols(data_dict)

    new_fields = []
    for name in id_cols:
        for col in data_dict['fields']:
            if col['name'] == name: 
                new_fields.append(col)
                
    for op in collapse_range_features.split(' '):
        for low, high in ast.literal_eval(range_pairs):
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
    non_time_cols.sort()
    return non_time_cols

def parse_time_cols(data_dict):
    time_cols = []
    for col in data_dict['fields']:
        # TODO avoid hardcoding a column name
        if (col['name'] == 'hours' or col['role'].count('time') > 0):
            time_cols.append(col['name'])
    return time_cols
            
def parse_time_col(data_dict):
    time_cols = []
    for col in data_dict['fields']:
        # TODO avoid hardcoding a column name
        if (col['name'] == 'hours' or col['role']=='timestamp_relative'):
            time_cols.append(col['name'])
    return time_cols[-1]

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

    if len(percentile_data_np)==0:
        percentile_data_np = np.zeros((1, percentile_data_np.shape[1]))
    else:
        all_nan_col_ind = np.isnan(percentile_data_np).all(axis = 0)
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

def collapse_skew_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)
    return skew(percentile_data_np, axis=0, nan_policy='omit')

def collapse_count_np(data_np, lower_bound, upper_bound, **kwargs):
    return (~np.isnan(data_np[lower_bound:upper_bound,:])).sum(axis=0)
 
def collapse_present_np(data_np, lower_bound, upper_bound, **kwargs):
    return (~np.isnan(data_np[lower_bound:upper_bound,:])).any(axis=0)

def collapse_hours_since_measured_np(data_np, lower_bound, upper_bound, cur_timestamp_arr=None, **kwargs):
    '''
    Computes the time since last value was observed from the last stamps
    Example : 
    data = [0, 1 , nan, 4, 5, nan, nan]
    tstamp = [30, 32, 36, 40, 45, 51, 60]
    
    output : 60-45=15
    '''
    percentile_t_np = cur_timestamp_arr[lower_bound:upper_bound]
    percentile_data_np = data_np[lower_bound:upper_bound,:]
    n_cols = percentile_data_np.shape[1]

    collapsed_hours_since_missing = np.zeros(n_cols)
    for col in range(n_cols):
        mask = ~np.isnan(percentile_data_np[:,col])
        if mask.sum():
            xs = percentile_data_np[mask,col]
            ts = percentile_t_np[mask]
            collapsed_hours_since_missing[col] = percentile_t_np[-1] - ts[-1]
        else: # set to large value if no measurement is observed in the sequence
            collapsed_hours_since_missing[col] = 120
    return collapsed_hours_since_missing

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
    "present": collapse_present_np,
    "skew":collapse_skew_np,
    "hours_since_measured":collapse_hours_since_measured_np
}

if __name__ == '__main__':
    main()