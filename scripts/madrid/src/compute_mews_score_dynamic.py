import sys
import pandas as pd
import argparse
import json
import numpy as np
from progressbar import ProgressBar
import copy
import os

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from feature_transformation import (parse_id_cols, remove_col_names_from_list_if_not_in_df, parse_time_cols, parse_feature_cols, calc_start_and_stop_indices_from_percentiles)
from utils import load_data_dict_json

def compute_mews_dynamic(ts_df, data_dict, mews_df, outcomes_df):
    id_cols = parse_id_cols(data_dict)
    id_cols = remove_col_names_from_list_if_not_in_df(id_cols, ts_df)
    
    feature_cols = ['systolic_blood_pressure', 'heart_rate', 'respiratory_rate', 'body_temperature']

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
    nrows = len(fp)- 1
    
    timestamp_arr = np.asarray(ts_df[time_col].values.copy(), dtype=np.float32)
    features_arr = ts_df[feature_cols].values
    ids_arr = ts_df[id_cols].values
    prediction_window = 12
    prediction_horizon = 24
    max_hrs_data_observed = 504
    t_start=-24 # start time 
    dynamic_mews_id_list = list()
    dynamic_outcomes_list = list()
    dynamic_window_list = list()
    dynamic_stay_lengths_list = list()
    
    # define outcome column (TODO : Avoid hardcording by loading from config.json)
    outcome_col = 'clinical_deterioration_outcome'
    
    # impute missing values per feature to population median for that feature
    print('Imputing missing values with forward fill for MEWS computation...')
    ts_df_imputed = ts_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad'))
    ts_df_imputed.fillna(ts_df_imputed.median(), inplace=True)
    mews_features_df = ts_df_imputed[feature_cols].copy()
    
    print('Computing mews scores dynamically...')
    pbar=ProgressBar()
    dynamic_mews_scores_list = list()
    
    for p in pbar(range(nrows)):
        # get the data for the current fencepost
        fp_start = fp[p]
        fp_end = fp[p+1]

        cur_timestamp_arr = timestamp_arr[fp_start:fp_end]
        cur_mews_features_df = mews_features_df.iloc[fp_start:fp_end,:].reset_index(drop=True)
        
        # get the current stay id (Do this outside the loop)
        cur_id_df = ts_df[id_cols].iloc[fp[p]:fp[p+1]].drop_duplicates(subset=id_cols)

        # get the stay length of the current 
        cur_outcomes_df = pd.merge(outcomes_df, cur_id_df, on=id_cols, how='inner')
        cur_stay_length = cur_outcomes_df['stay_length'].values[0]
        cur_final_outcome = int(cur_outcomes_df[outcome_col].values[0])

        # create windows from start to length of stay (0-prediction_window, 0-2*prediction_window, ... 0-length_of_stay)
        t_end = min(cur_stay_length, max_hrs_data_observed)
#         window_ends = np.arange(t_start+prediction_window, t_end+prediction_window, prediction_window)
        window_ends = np.arange(
            t_start + prediction_window,
            t_end + 0.01 * prediction_window,
            prediction_window)
        
        cur_dynamic_mews_scores = np.zeros([len(window_ends), 1], dtype=np.float32)
        
        for q, window_end in enumerate(window_ends):
            cur_dynamic_idx = (cur_timestamp_arr>t_start)&(cur_timestamp_arr<=window_end)
            cur_dynamic_timestamp_arr = cur_timestamp_arr[cur_dynamic_idx]
            cur_dynamic_mews_features_df = cur_mews_features_df[cur_dynamic_idx]
        
            
            cur_mews_scores = np.zeros(len(cur_dynamic_timestamp_arr))
            
            if len(cur_dynamic_timestamp_arr)>0:
                for feature in feature_cols:
                    feature_vals_np = cur_dynamic_mews_features_df[feature].astype(float)
                    mews_df_cur_feature = mews_df[mews_df['vital']==feature].reset_index(drop=True)
                    feature_maxrange_np = mews_df_cur_feature['range_max'].to_numpy().astype(float)
                    scores_idx = np.searchsorted(feature_maxrange_np, feature_vals_np)
                    cur_mews_scores += mews_df_cur_feature.loc[scores_idx, 'score'].to_numpy().astype(float)


                cur_dynamic_mews_scores[q] = cur_mews_scores[-1]
            
            # set mews score as last observed mews score over all timesteps
            
            # keep track of stay ids
            dynamic_mews_id_list.append(cur_id_df.values[0])

            # keep track of windows
            dynamic_window_list.append(np.array([t_start, window_end]))

            # keep track of the stay lengths
            dynamic_stay_lengths_list.append(cur_stay_length)

            # if the length of stay is within the prediction horizon, set the outcome as the clinical deterioration outcome, else set 0
            if window_end>=cur_stay_length-prediction_horizon:
                dynamic_outcomes_list.append(cur_final_outcome)
            else:
                dynamic_outcomes_list.append(0)
            
        
        dynamic_mews_scores_list.append(cur_dynamic_mews_scores)
        
    # horizontally stack across all slices
    dynamic_mews_df = pd.DataFrame(np.vstack(dynamic_mews_scores_list), columns=['mews_score'])
    
    # add the ids back to the collapsed features
    ids_df = pd.DataFrame(dynamic_mews_id_list, columns=id_cols) 
    
    # add the window start and ends
    dynamic_window_df = pd.DataFrame(np.vstack(dynamic_window_list), columns=['window_start','window_end'])
    dynamic_stay_lengths_df = pd.DataFrame(np.vstack(dynamic_stay_lengths_list), columns=['stay_length'])
    
    dynamic_mews_df = pd.concat([ids_df, dynamic_mews_df, dynamic_window_df], axis=1)
    
    
    dynamic_outcomes_df = pd.DataFrame(np.array(dynamic_outcomes_list), columns=[outcome_col])
    dynamic_outcomes_df = pd.concat([ids_df, dynamic_outcomes_df, dynamic_window_df, dynamic_stay_lengths_df], axis=1)  
    
    return dynamic_mews_df, dynamic_outcomes_df


def update_data_dict_mews(data_dict): 
    id_cols = parse_id_cols(data_dict)

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
                        help='Path to vitals csv dataframe of readings')
    parser.add_argument('--data_dict', type=str, required=True,
                        help='Path to vitals json data dictionary file')
    parser.add_argument('--outcomes', type=str, required=True, 
                        help='Path to csv dataframe of outcomes')
    parser.add_argument('--data_dict_outcomes', type=str, required=True,
                        help='Path to json data dictionary file for outcomes')
    parser.add_argument('--dynamic_mews_csv', type=str, required=False, default=None)
    parser.add_argument('--dynamic_mews_data_dict', type=str, required=False, default=None)
    parser.add_argument('--dynamic_outcomes_csv', type=str, required=False, default=None)   

    
    args = parser.parse_args()
    
    print('reading features...')
    ts_df = pd.read_csv(args.input)
    data_dict = load_data_dict_json(args.data_dict)
    print('done reading features...')
    
    print('reading outcomes...')
    outcomes_df = pd.read_csv(args.outcomes)
    data_dict_outcomes = load_data_dict_json(args.data_dict_outcomes)
    print('done reading outcomes...')

    
    # define the mews score dataframe
    max_val=np.inf
    mews_list = [['systolic_blood_pressure', 0, 70, 3],['systolic_blood_pressure', 70, 80, 2],['systolic_blood_pressure', 80, 100, 1],
            ['systolic_blood_pressure', 100, 199, 0],['systolic_blood_pressure', 199, max_val, 2],
            ['heart_rate', 0, 40, 2],['heart_rate', 40, 50, 1],['heart_rate', 50, 100, 0],['heart_rate', 100, 110, 1],
            ['heart_rate', 110, 129, 2], ['heart_rate', 129, max_val, 3],
            ['respiratory_rate', 0, 9, 2], ['respiratory_rate', 9, 15, 0], ['respiratory_rate', 15, 20, 1], 
            ['respiratory_rate', 20, 30, 2], ['respiratory_rate', 30, max_val, 3],
            ['body_temperature', 0, 35, 2], ['body_temperature', 35, 38.5, 0], ['body_temperature', 38.5, max_val, 2]]
    mews_df = pd.DataFrame(columns=['vital', 'range_min', 'range_max', 'score'], data=np.vstack(mews_list))
    
    dynamic_mews_df, dynamic_outcomes_df = compute_mews_dynamic(ts_df, data_dict, mews_df, outcomes_df)
    
    dynamic_mews_data_dict = update_data_dict_mews(data_dict)
    
    # save data to file
    dynamic_mews_df.to_csv(args.dynamic_mews_csv, index=False, compression='gzip')
    print('Saved dynamic mews to :\n%s'%args.dynamic_mews_csv)
    
    dynamic_outcomes_df.to_csv(args.dynamic_outcomes_csv, index=False, compression='gzip')
    print('Saved dynamic outcomes to :\n%s'%args.dynamic_outcomes_csv)
    
    
    # save data dictionary to file
    with open(args.dynamic_mews_data_dict, 'w') as f:
        json.dump(dynamic_mews_data_dict, f, indent=4)

    print ('Saved dynamic mews dict to :\n%s'%args.dynamic_mews_data_dict)

