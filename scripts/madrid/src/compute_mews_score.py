import sys
import pandas as pd
import argparse
import json
import numpy as np
from progressbar import ProgressBar
import copy

def parse_id_cols(data_dict):
    cols = []
    for col in data_dict['fields']:
        if 'role' in col and (col['role'] == 'id' or 
                              col['role'] == 'key'):
            cols.append(col['name'])
    return cols

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

def parse_time_col(data_dict):
    time_cols = []
    for col in data_dict['fields']:
        # TODO avoid hardcoding a column name
        if (col['name'] == 'hours' or col['role'].count('time')):
            time_cols.append(col['name'])
    return time_cols[-1]

def parse_feature_cols(data_dict):
    non_time_cols = []
    for col in data_dict['fields']:
        if 'role' in col and col['role'] in ('feature', 'measurement', 'covariate'):
            non_time_cols.append(col['name'])
    return non_time_cols

def calc_start_and_stop_indices_from_percentiles(timestamp_arr, start_percentile, end_percentile, max_time_step=None):
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

    # Consider last data point as first timestamp + step size input by the user. For eg. If the step size is 1hr, then consider only 
    # first hour of time series data
    min_timestamp = timestamp_arr[0]
    if (max_time_step is None) or (max_time_step==-1):
        max_timestamp = timestamp_arr[-1]
    elif (min_timestamp + max_time_step > timestamp_arr[-1]):
        max_timestamp = timestamp_arr[-1]
    else :
        max_timestamp = min_timestamp + max_time_step
    lower_bound = np.searchsorted(timestamp_arr, (min_timestamp + (max_timestamp - min_timestamp)*start_percentile/100))
    upper_bound = np.searchsorted(timestamp_arr, (min_timestamp + (max_timestamp - min_timestamp)*(end_percentile + 0.001)/100))
    # if lower bound and upper bound are the same, add 1 to the upper bound
    if lower_bound >= upper_bound:
        upper_bound = lower_bound + 1
    assert upper_bound <= timestamp_arr.size + 1
    return int(lower_bound), int(upper_bound)


def compute_mews(ts_df, args, mews_df):
    id_cols = parse_id_cols(args.data_dict)
    id_cols = remove_col_names_from_list_if_not_in_df(id_cols, ts_df)
    #feature_cols = parse_feature_cols(args.data_dict)
    feature_cols = ['systolic_blood_pressure', 'heart_rate', 'respiratory_rate', 'body_temperature']
    #feature_cols = remove_col_names_from_list_if_not_in_df(feature_cols, ts_df)
    time_col = parse_time_col(args.data_dict)

    # Obtain fenceposts based on where any key differs
    # Be sure keys are converted to a numerical datatype (so fencepost detection is possible)
    keys_df = ts_df[id_cols].copy()
    for col in id_cols:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    fp = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0).any(axis=1)), keys_df.shape[0]])
    nrows = len(fp)- 1

    timestamp_arr = np.asarray(ts_df[time_col].values.copy(), dtype=np.float64)
    mews_features_df = ts_df[feature_cols].copy()
    mews_scores = np.zeros(nrows)
    
    # impute missing values per feature to population median for that feature
    mews_features_df.fillna(mews_features_df.median(), inplace=True)
    pbar=ProgressBar()
    for p in pbar(range(nrows)):
    #for p in range(100):
        # get the data for the current fencepost
        fp_start = fp[p]
        fp_end = fp[p+1]
        lower_bound, upper_bound = calc_start_and_stop_indices_from_percentiles(timestamp_arr[fp_start:fp_end], 
                start_percentile=0, end_percentile=100, max_time_step=args.max_time_step)

        cur_timestamp_arr = timestamp_arr[fp_start:fp_end][lower_bound:upper_bound]
        cur_features_df = mews_features_df.iloc[fp_start:fp_end,:].reset_index(drop=True).iloc[lower_bound:upper_bound,:]
        
        cur_mews_scores = np.zeros(len(cur_timestamp_arr))
        for feature in feature_cols:
            feature_vals_np = cur_features_df[feature].astype(float)
            mews_df_cur_feature = mews_df[mews_df['vital']==feature].reset_index(drop=True)
            feature_maxrange_np = mews_df_cur_feature['range_max'].to_numpy().astype(float)
            scores_idx = np.searchsorted(feature_maxrange_np, feature_vals_np)
            cur_mews_scores += mews_df_cur_feature.loc[scores_idx, 'score'].to_numpy().astype(float)
        
        # set mews score as median across timesteps
        mews_scores[p]=np.median(cur_mews_scores) 
    mews_scores_df = pd.DataFrame(data=mews_scores, columns=['mews_score'])

    for col_name in id_cols[::-1]:
        mews_scores_df.insert(0, col_name, ts_df[col_name].values[fp[:-1]].copy())
    return mews_scores_df   
            
def update_data_dict_mews(args): 
    data_dict = args.data_dict

    id_cols = parse_id_cols(args.data_dict)
    feature_cols = parse_feature_cols(args.data_dict)

    new_fields = []
    for name in id_cols:
        for col in data_dict['fields']:
            if col['name'] == name: 
                new_fields.append(col)
    
    new_fields.append({'name': 'mews_score','role': 'feature','type': 'numeric',
        'description': 'Modified Early Warning Score','units': 'NONE',
        'constraints': {'required': 'FALSE', 'minimum': '0', 'maximum': 'INF'}})

    new_data_dict = copy.deepcopy(data_dict)
    if 'schema' in new_data_dict:
        new_data_dict['schema']['fields'] = new_fields
        del new_data_dict['fields']
    else:
        new_data_dict['fields'] = new_fields

    return new_data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for computing mews score for a subject-episode")
    parser.add_argument('--input', type=str, required=True,
                        help='Path to csv dataframe of readings')
    parser.add_argument('--data_dict', type=str, required=True,
                        help='Path to json data dictionary file')
    parser.add_argument('--max_time_step', type=int, required=False,
                        default=None, help="Specify the maximum number of time "
                                         "steps to compute mews, for example, "
                                         "input 48 for 48 hours at 1 hour time steps. "
                                         "Set to -1 for no limit.")
    parser.add_argument('--output', type=str, required=False, default=None)
    parser.add_argument('--data_dict_output', type=str, required=False, 
                        default=None)


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

    
    # define the mews score dataframe
    max_val=np.inf
    mews_list = [['systolic_blood_pressure', 0, 70, 3],['systolic_blood_pressure', 70, 80, 2],['systolic_blood_pressure', 80, 100, 1],
            ['systolic_blood_pressure', 100, 199, 0],['systolic_blood_pressure', 199, max_val, 2],
            ['heart_rate', 0, 40, 2],['heart_rate', 40, 50, 1],['heart_rate', 50, 100, 0],['heart_rate', 100, 110, 1],
            ['heart_rate', 110, 129, 2], ['heart_rate', 129, max_val, 3],
            ['respiratory_rate', 0, 28, 2], ['respiratory_rate', 28, 44, 0], ['respiratory_rate', 44, 64, 1], 
            ['respiratory_rate', 64, 92, 2], ['respiratory_rate', 92, max_val, 3],
            ['body_temperature', 0, 35, 2], ['body_temperature', 35, 38.4, 0], ['body_temperature', 38.4, max_val, 2]]
    mews_df = pd.DataFrame(columns=['vital', 'range_min', 'range_max', 'score'], data=np.vstack(mews_list))
    
    print('computing mews scores..')
    mews_scores_df = compute_mews(ts_df, args, mews_df)
    
    data_dict = update_data_dict_mews(args)
    
    # save data to file
    print('saving data to output...')
    if args.output is None:
        file_name = args.input.split('/')[-1].split('.')[0]
        data_output = '{}_transformed.csv'.format(file_name)
    elif args.output[-4:] == '.csv':
        data_output = args.output
    else:
        data_output = '{}.csv'.format(args.output)
    mews_scores_df.to_csv(data_output, index=False)
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

