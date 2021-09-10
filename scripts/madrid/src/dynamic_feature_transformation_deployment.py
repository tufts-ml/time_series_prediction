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
from feature_transformation import (get_fenceposts, parse_time_cols, parse_id_cols, parse_feature_cols, 
                                    update_data_dict_collapse, remove_col_names_from_list_if_not_in_df) 


sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from utils import load_data_dict_json
from progressbar import ProgressBar


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
    parser.add_argument('--features_to_summarize', type=str, required=False,
                        default='slope std', 
                        help="Enclose options with 's, choose "
                             "from mean, std, min, max, "
                             "median, slope, count, present, skew, hours_since_measured")
    parser.add_argument('--percentile_ranges_to_summarize', type=str, required=False,
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
                                                                 features_to_summarize=args.features_to_summarize, 
                                                                 percentile_ranges_to_summarize=args.percentile_ranges_to_summarize, outcomes_df=outcomes_df, 
                                                                 data_dict_outcomes=data_dict_outcomes)
    
    
    dynamic_collapsed_features_data_dict = update_data_dict_collapse(data_dict, args.features_to_summarize, args.percentile_ranges_to_summarize)
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
    

    
def featurize_ts(
        time_arr_by_var,
        val_arr_by_var,
        var_cols=[],
        start_numerictime=None,
        stop_numerictime=None,
        summary_ops=['count', 'mean', 'std', 'slope'],
        percentile_slices_to_featurize=[(0., 100.)],
        ):
    ''' Featurize provided multivariate irregular time series into flat vector

    Args
    ----
    time_arr_by_var : dict of 1D NumPy arrays
    val_arr_by_var : dict of 1D NumPy arrays
    var_cols : list of strings
        Indicates the name of each desired variable
    start_numerictime : float
        Indicates numerical time value at which current window *starts*
    stop_numerictime : float
        Indicates numerical time that current window *stops*

    Returns
    -------
    feat_vec_1F : 2D NumPy array, shape (1, F)
        One entry for each combination of {variable, summary op, subwindow slice}
    feat_names : list of strings
        Gives the names (in order) of the collapsed features
    '''
    if start_numerictime is None:
        start_numerictime = time_arr_by_var.get('__START_TIME__', None)
    if stop_numerictime is None:
        stop_numerictime = time_arr_by_var.get('__PREDICTION_TIME__', None)
    time_range = stop_numerictime - start_numerictime

    F = len(percentile_slices_to_featurize) * len(var_cols) * len (summary_ops)
    feat_vec_1F = np.zeros((1, F))
    feat_names = list()
    ff = 0
    for rp_ind, (low, high) in enumerate(percentile_slices_to_featurize):
        cur_window_start_time = start_numerictime + float(low) / 100 * time_range
        cur_window_stop_time = start_numerictime + float(high) / 100 * time_range

        for var_id, var_name in enumerate(var_cols):           
            try:
                cur_feat_arr = val_arr_by_var[var_name]
                cur_numerictime_arr = time_arr_by_var[var_name]
                start, stop = calc_start_and_stop_indices_from_percentiles(
                    cur_numerictime_arr,
                    start_percentile=low,
                    end_percentile=high)
                cur_numerictime_arr = cur_numerictime_arr[start:stop]
                cur_feat_arr = cur_feat_arr[start:stop]

            except KeyError:
                # Current variable never measured in provided data
                cur_numerictime_arr = np.zeros(0)
                cur_feat_arr = np.zeros(0)

            for op_ind, op in enumerate(summary_ops):
                summary_func, empty_val = SUMMARY_OPERATIONS[op]

                cur_isfinite_arr = np.isfinite(cur_feat_arr)
                if cur_feat_arr.size < 1 or cur_isfinite_arr.sum() < 1:
                    feat_vec_1F[0,ff] = empty_val
                else:
                    feat_vec_1F[0,ff] = summary_func(
                        cur_feat_arr, cur_numerictime_arr, cur_isfinite_arr,
                        cur_window_start_time, cur_window_stop_time)
                feat_names.append("%s_%s_%.0f-%.0f" % (var_name, op, float(low), float(high)))
                ff += 1

    return feat_vec_1F, feat_names

def collapse_dynamic(ts_df, data_dict, features_to_summarize, percentile_ranges_to_summarize, outcomes_df, data_dict_outcomes):
    ''' Featurize multiple patient stays slices, and extract the outcome for each slice

    Args
    ----
    ts_df : Dataframe containing raw per-time-step features
    data_dict : dict of spec for every column in ts_df
    features_to_summarize : list of strings
        Indicates the list of all the summary functions
    percentile_ranges_to_summarize : list of tuples (example : [(0, 100), (0, 50)])
        Indicates numerical time value at which current window *starts*
    outcomes_df : Dataframe containing outcomes-per-sequence
    data_dict_outcomes : dict of spec for every column in outcomes_df

    Returns
    -------
    dynamic_collapsed_df : Dataframe containing the collapsed features for all patient-stay slices
    dynamic_outcomes_df : Dataframe containing the outcomes for all patient-stay slices
    '''    
    
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
    
    # define outcome column (TODO : Avoid hardcording by loading from config.json)
    outcome_col = 'clinical_deterioration_outcome'
    
    # Start timer
    total_time = 0
    timestamp_arr = np.asarray(ts_df[time_col].values.copy(), dtype=np.float32)
    features_arr = ts_df[feature_cols].values
    ids_arr = ts_df[id_cols].values

    prediction_window = 12 # change to spacing_between_endpoints
    prediction_horizon = 24
    max_hrs_data_observed = 504
    t_start=-24 # start time 
    dynamic_collapsed_feat_id_list = list()
    dynamic_outcomes_list = list()
    dynamic_window_list = list()
    dynamic_stay_lengths_list = list()
    pbar=ProgressBar()
    n_rows = len(fp) - 1
    
    dynamic_collapsed_feats_all_fps_list = []
    dynamic_outcomes_all_fps_list = []
    t1=time.time()
    
    for p in pbar(range(n_rows)):
        
        # Get features and times for the current fencepost
        fp_start = fp[p]
        fp_end = fp[p+1]

        # get the current stay id (Do this outside the loop)
        cur_id_df = ts_df[id_cols].iloc[fp[p]:fp[p+1]].drop_duplicates(subset=id_cols)

        # get the stay length of the current 
        cur_outcomes_df = pd.merge(outcomes_df, cur_id_df, on=id_cols, how='inner')
        cur_stay_length = cur_outcomes_df['stay_length'].values[0]
        cur_final_outcome = int(cur_outcomes_df[outcome_col].values[0])

        # create windows from start to length of stay (0-prediction_window, 0-2*prediction_window, ... 0-length_of_stay)
        t_end = min(cur_stay_length, max_hrs_data_observed)
        window_ends = np.arange(t_start+prediction_window, t_end+prediction_window, prediction_window)
        
        # create a dictionary of times and values for each feature
        cur_fp_df = ts_df.loc[fp[p]:fp[p+1], [time_col]+feature_cols] 
        v = cur_fp_df.set_index(time_col).agg(lambda x: x.dropna().to_dict()) 
        res = v[v.str.len() > 0].to_dict()
        
        time_arr_by_var = dict()
        val_arr_by_var = dict()
        for feature_col in feature_cols:
            if feature_col in res.keys():
                time_arr_by_var[feature_col] = np.array(list(res[feature_col].keys()))
                val_arr_by_var[feature_col] = np.array(list(res[feature_col].values()))
        
        # get the summary operations and the percentile ranges to collapse in to pre-allocate the collapsed feature matrix
        percentile_slices_to_featurize=ast.literal_eval(percentile_ranges_to_summarize)
        
        summary_ops = features_to_summarize.split(' ')
        F = len(percentile_slices_to_featurize) * len(feature_cols) * len (summary_ops)
        
        cur_dynamic_collapsed_feat_arr = np.zeros([len(window_ends), F], dtype=np.float32)
        cur_dynamic_final_outcomes = np.zeros([len(window_ends), 1], dtype=np.int64)
        cur_dynamic_window_starts_and_ends = np.zeros([len(window_ends), 2])
        
        for q, window_end in enumerate(window_ends):
            cur_dynamic_collapsed_feat_arr[q, :], feat_names = featurize_ts(time_arr_by_var, val_arr_by_var, 
                                                   var_cols=feature_cols, start_numerictime=t_start,
                                                   stop_numerictime=window_end,
                                                   summary_ops=summary_ops,
                                                   percentile_slices_to_featurize=percentile_slices_to_featurize)
            
            # if the length of stay is within the prediction horizon, set the outcome as the clinical deterioration outcome, else set 0
            if window_end>=cur_stay_length-prediction_horizon:
                cur_dynamic_final_outcomes[q] = cur_final_outcome
            else:
                cur_dynamic_final_outcomes[q] = 0
            
            
            cur_dynamic_window_starts_and_ends[q, :] = np.array([t_start, window_end])
        
        
        # stack across all fps
        dynamic_collapsed_feats_all_fps_list.append(cur_dynamic_collapsed_feat_arr)
        
        dynamic_outcomes_all_fps_list.append(cur_dynamic_final_outcomes)
        
        # keep track of ids
        dynamic_collapsed_feat_id_list.append(np.tile(cur_id_df.values[0], (len(window_ends), 1)))
        
        # keep track of window ends
        dynamic_window_list.append(cur_dynamic_window_starts_and_ends)
        
        # keep track of the stay lengths
        dynamic_stay_lengths_list.append(np.tile(cur_stay_length, (len(window_ends), 1)))
        
    
    # horizontally stack across all ops
    dynamic_collapsed_df = pd.DataFrame(np.vstack(dynamic_collapsed_feats_all_fps_list), columns=feat_names)    
    
    # add the ids back to the collapsed features
    ids_df = pd.DataFrame(np.vstack(dynamic_collapsed_feat_id_list), columns=id_cols) 
    
    # add the window start and ends
    dynamic_window_df = pd.DataFrame(np.vstack(dynamic_window_list), columns=['window_start','window_end'])
    dynamic_stay_lengths_df = pd.DataFrame(np.vstack(dynamic_stay_lengths_list), columns=['stay_length'])
    
    dynamic_collapsed_df = pd.concat([ids_df, dynamic_collapsed_df, dynamic_window_df], axis=1)
    
    dynamic_outcomes_df = pd.DataFrame(np.vstack(dynamic_outcomes_all_fps_list), columns=[outcome_col])
    dynamic_outcomes_df = pd.concat([ids_df, dynamic_outcomes_df, dynamic_window_df, dynamic_stay_lengths_df], axis=1)     
    
    t2 = time.time()
    print('done in %d seconds'%(t2-t1))
    total_time = total_time + t2-t1    

    print('-----------------------------------------')
    print('total time taken to collapse features = %d seconds'%total_time)
    print('-----------------------------------------')    
    
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


def collapse_mean(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    '''
    '''
    return np.nanmean(data_arr)

def collapse_std(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    '''
    '''
    return np.nanstd(data_arr)
    
def collapse_median(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    '''
    '''
    return np.nanmedian(data_arr)
    
def collapse_min(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    '''
    '''
    return np.nanmin(data_arr)

def collapse_max(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    '''
    '''
    return np.nanmax(data_arr)

def collapse_count(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    return np.sum(np.isfinite(data_arr))


def collapse_slope(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    ''' Compute slope within current window of time

    Treat time as *relative* between 0 and 1 within the window

    Examples
    --------
    >>> ts = np.asarray([0., 1., 2., 3.])
    >>> ys = np.asarray([0., 1., 2., 3.])
    >>> fs = np.ones(4, dtype=np.bool)
    >>> "%.5f" % collapse_slope(ys, ts, fs, 0.0, 3.0)
    '3.00000'

    >>> B = 77.7
    >>> "%.5f" % collapse_slope(ys, ts + B, fs, 0.0 + B, 3.0 + B)
    '3.00000'
    '''
    ys = data_arr[isfinite_arr]
    ts = (timestamp_arr[isfinite_arr] - tstart) / float(tstop - tstart)

    ymean = np.mean(ys)
    tmean = np.mean(ts)
    ys -= ymean
    ts -= tmean

    numer = np.sum(ts * ys)
    denom = np.sum(np.square(ts))
    slope = numer / (1e-10 + denom)
    return slope
    #intercept = ymean - slope * tmean
    #return slope, intercept


def collapse_elapsed_time_since_last_measured(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    '''
    Computes the time since last value was observed from the last stamps

    Example
    -------
    >>> data = np.asarray([0, 1 , np.nan, 4, 5, np.nan, np.nan])
    >>> times = np.asarray([30, 32, 36, 40, 45, 51, 60])
    >>> collapse_elapsed_time_since_last_measured(data, times, None, 0, 60)
    15.0
    '''
    if isfinite_arr is None:
        isfinite_arr = np.isfinite(data_arr)

    if np.sum(isfinite_arr) == 0:
        tlast = tstart
    else:
        tlast = timestamp_arr[np.flatnonzero(isfinite_arr)[-1]]
    return float(tstop - tlast)



# Map each desired summary function to
# 1) the function to call for non-empty input arrays w/ at least one finite val
# 2) the numerical value to use when the target time slice is empty

SUMMARY_OPERATIONS = {
    'mean' : (collapse_mean, 0.0),
    'median' : (collapse_median, 0.0),
    'std' : (collapse_std, -1.0),
    'slope' : (collapse_slope, 0.0),
    'count': (collapse_count, 0.0),
    'hours_since_measured' : (collapse_elapsed_time_since_last_measured, -1.0),
    'max' : (collapse_max, 0.0),
    'min' : (collapse_min, 0.0)
}    
    
if __name__ == '__main__':
    main()