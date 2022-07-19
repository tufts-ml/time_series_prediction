
'''
featurize_single_time_series.py
Code for producing a flat feature vector from one multivariate time series.
'''

import pandas as pd
import numpy as np
import time
import os

def featurize_ts(
        time_arr_by_var,
        val_arr_by_var,
        var_cols=[],
        var_spec_dict=None,
        var_to_minmax_dict=None,
        start_numerictime=None,
        stop_numerictime=None,
        percentile_slices_to_featurize=[(0., 100.)],
        summary_ops=['count', 'mean', 'std', 'slope'],
        ):
    ''' Featurize provided multivariate irregular time series into flat vector
    Args
    ----
    time_arr_by_var : dict of 1D NumPy arrays
    val_arr_by_var : dict of 1D NumPy arrays
    var_cols : list of strings
        Indicates the name of each desired variable
    var_spec_dict : dict or None
        Optional data specification (data dictionary) for each measured var.
        Follows ts_pred data specification format
        Provides optional min/max feasible values for each variable
        When provided, will override the var_to_minmax_dict argument
    var_to_minmax_dict : dict or None
        Optional min/max allowed values for each numerical variable
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
    Examples
    --------
    >>> time_arr_by_var = {'hr':np.asarray([- 5.0,  0.0,  5.0, 10.0])}
    >>> val_arr_by_var = {'hr': np.asarray([ 20.0, 25.0, 30.0, 55.0])}
    >>> feat_vec, names = featurize_ts(
    ...     time_arr_by_var, val_arr_by_var, var_cols=['hr'],
    ...     start_numerictime=-10.0,
    ...     stop_numerictime =+ 5.0,
    ...     percentile_slices_to_featurize=[(0,100), (0, 25)],
    ...     summary_ops=['mean', 'slope', 'last_value_measured'])
    >>> _ = [print("%5.2f %s" % (val, name))
    ...         for (name,val) in zip(names, feat_vec[0])];
    25.00 hr_mean_0-100
     1.00 hr_slope_0-100
    30.00 hr_last_value_measured_0-100
     0.00 hr_mean_0-25
     0.00 hr_slope_0-25
     0.00 hr_last_value_measured_0-25
    '''
    if start_numerictime is None:
        start_numerictime = time_arr_by_var.get('__START_TIME__', None)
    if stop_numerictime is None:
        stop_numerictime = time_arr_by_var.get('__PREDICTION_TIME__', None)
    time_range = stop_numerictime - start_numerictime

    # Prep to do filtering by outliers
    if var_spec_dict is not None:
        var_to_minmax_dict = make_var_to_minmax_from_spec(var_spec_dict, var_cols)
    elif var_to_minmax_dict is None:
        var_to_minmax_dict = {}

    F = len(percentile_slices_to_featurize) * len(var_cols) * len (summary_ops)
    feat_vec_1F = np.zeros((1, F))
    feat_names = list()
    ff = 0

    SUMMARY_OPERATIONS = make_summary_ops()

    for rp_ind, (low, high) in enumerate(percentile_slices_to_featurize):
        cur_window_start_time = start_numerictime + float(low) / 100 * time_range
        cur_window_stop_time = start_numerictime + float(high) / 100 * time_range

        for var_id, var_name in enumerate(var_cols):

            # Obtain measurements and times for current window         
            try:
                cur_feat_arr = val_arr_by_var[var_name].astype('float')
                cur_numerictime_arr = time_arr_by_var[var_name]

                # Keep only the entries whose times occur within current window
                start = np.searchsorted(
                    cur_numerictime_arr, cur_window_start_time, side='left')
                stop = np.searchsorted(
                    cur_numerictime_arr, cur_window_stop_time, side='right')
                cur_numerictime_arr = cur_numerictime_arr[start:stop]
                cur_feat_arr = cur_feat_arr[start:stop]

                # Keep only entries whose values fall within allowed range
                # This step will discard outliers
                if var_name in var_to_minmax_dict:
                    min_val_allowed, max_val_allowed = var_to_minmax_dict[var_name]
                    keep_mask = np.logical_and(
                        cur_feat_arr >= min_val_allowed,
                        cur_feat_arr <= max_val_allowed)
                    cur_feat_arr = cur_feat_arr[keep_mask]
                    cur_numerictime_arr = cur_numerictime_arr[keep_mask]

            except KeyError:
                # Current variable never measured in provided data
                cur_numerictime_arr = np.zeros(0)
                cur_feat_arr = np.zeros(0)
            
            cur_isfinite_arr = np.isfinite(cur_feat_arr)
            for op_ind, op in enumerate(summary_ops):
                summary_func, empty_val = SUMMARY_OPERATIONS[op]
                if cur_feat_arr.size < 1 or cur_isfinite_arr.sum() < 1:
                    feat_vec_1F[0,ff] = empty_val
                else:
                    feat_vec_1F[0,ff] = summary_func(
                        cur_feat_arr, cur_numerictime_arr, cur_isfinite_arr,
                        cur_window_start_time, cur_window_stop_time)
                feat_names.append("%s_%s_%.0f-%.0f" % (var_name, op, float(low), float(high)))
                ff += 1
    return feat_vec_1F, feat_names

def make_summary_ops():
    ''' Create defaults for summarization functions
    Returns
    -------
    summary_ops : dict
        Keys are names of the summary operations
        Values are a tuple, containing:
        - function to execute to compute the summary
        - default numerical value to use when the target time slice is empty
    '''
    summary_ops = {
        'count': (collapse_count, 0.0),
        'last_value_measured' : (collapse_value_last_measured, 0.0),
        'time_since_measured' : (collapse_elapsed_time_since_last_measured, -24.0),
        'min' : (collapse_min, 0.0),
        'max' : (collapse_max, 0.0),
        'median' : (collapse_median, 0.0),
        'mean' : (collapse_mean, 0.0),
        'std' : (collapse_std, 10.0),
        'slope' : (collapse_slope, 0.0),
        }    
    return summary_ops


def collapse_mean(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    ''' Evaluate the mean of present values in provided window
    '''
    return np.nanmean(data_arr)

def collapse_std(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    ''' Evaluate the standard deviation of present values in provided window
    '''
    return np.nanstd(data_arr)
    
def collapse_median(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    ''' Evaluate the median of present values in provided window
    '''
    return np.nanmedian(data_arr)
    
def collapse_min(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    ''' Evaluate the minimum present value in provided window
    '''
    return np.nanmin(data_arr)

def collapse_max(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    ''' Evaluate the maximum present value in provided window
    '''
    return np.nanmax(data_arr)

def collapse_count(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    ''' Count the number of present values in window
    '''
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
    '1.00000'
    # Verify that slope is invariant to any additive offset to times
    >>> B = 77.7
    >>> "%.5f" % collapse_slope(ys, ts + B, fs, 0.0 + B, 3.0 + B)
    '1.00000'
    '''
    ys = data_arr[isfinite_arr]
    ts = timestamp_arr[isfinite_arr]

    ymean = np.mean(ys)
    tmean = np.mean(ts)
    ys -= ymean
    ts -= tmean

    # If desired, could also compute the intercept
    #intercept = ymean - slope * tmean
    #return slope, intercept

    numer = np.sum(ts * ys)
    denom = np.sum(np.square(ts))
    slope = numer / (1e-10 + denom)
    return slope

def collapse_value_last_measured(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    ''' Computes the latest value measured within this window
    Example
    -------
    >>> data = np.asarray([0, 1 , np.nan, 4, 5, np.nan, np.nan])
    >>> times = np.asarray([30, 32, 36, 40, 45, 51, 60])
    >>> collapse_value_last_measured(data, times, None, 0, 60)
    5.0
    '''
    if isfinite_arr is None:
        isfinite_arr = np.isfinite(data_arr)

    assert np.sum(isfinite_arr) > 0
    value = data_arr[np.flatnonzero(isfinite_arr)[-1]]
    return value


def collapse_elapsed_time_since_last_measured(data_arr, timestamp_arr, isfinite_arr, tstart, tstop):
    ''' Computes the time elapsed since the last value was observed
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


def make_var_to_minmax_from_spec(var_spec_dict, var_cols):
    ''' Create dict that provides min/max allowed values for each variable
    Returns
    -------
    var_to_minmax_dict : dict
        Keys : variable names
        Values : tuple of minimum,maximum numeric values allowed (inclusive)
    '''
    schema_dict = var_spec_dict.get('schema', var_spec_dict)
    list_of_var_info_dict = schema_dict.get('fields', [])

    def parse_val(v):
        if v == 'INF':
            return np.inf
        elif v == '-INF':
            return -np.inf
        return np.asarray(v, dtype=np.float64)

    var_to_minmax_dict = dict()
    for info_dict in list_of_var_info_dict:
        name = info_dict['name']
        if info_dict['role'] == 'measurement' and name in var_cols:
            if 'constraints' not in info_dict:
                continue
            
            contraints = info_dict['constraints']
            min_val_allowed = parse_val(contraints.get('minimum', -np.inf))
            max_val_allowed = parse_val(contraints.get('maximum', +np.inf))
            if min_val_allowed > -np.inf or max_val_allowed < np.inf:
                var_to_minmax_dict[name] = (min_val_allowed, max_val_allowed)
    return var_to_minmax_dict