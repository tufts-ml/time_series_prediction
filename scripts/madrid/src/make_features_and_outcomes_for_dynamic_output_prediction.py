import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.abspath('../'), 'predictions_collapsed'))
sys.path.append(os.path.join(os.path.abspath('../'), 'src'))
from config_loader import PROJECT_REPO_DIR
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'rnn'))
from feature_transformation import *
# from filter_admissions_by_tslice import get_preprocessed_data
# from merge_features_all_tslices import merge_data_dicts, get_all_features_data
from utils_preprocessing import get_preprocessed_data, get_all_features_data

import argparse
from progressbar import ProgressBar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc_data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--include_medications', type=str, default='True')
    args = parser.parse_args()
    
    # Get all the labs, vitals, demographics and outcomes
    labs_df, labs_data_dict, vitals_df, vitals_data_dict, demographics_df, demographics_data_dict, medications_df, medications_data_dict, outcomes_df, outcomes_data_dict = get_preprocessed_data(args.preproc_data_dir)
    
    # merge the labs, vitals and demographics to get a single features table
    if args.include_medications=='True':
        print('Getting labs, vitals, medications and demographics...')
        features_df,features_data_dict = get_all_features_data(labs_df, labs_data_dict, 
                                                               vitals_df, vitals_data_dict, 
                                                               demographics_df, demographics_data_dict, 
                                                               medications_df, medications_data_dict, True)
    else:
        print('Getting labs, vitalsand demographics...')
        features_df,features_data_dict = get_all_features_data(labs_df, labs_data_dict, 
                                                               vitals_df, vitals_data_dict, 
                                                               demographics_df, demographics_data_dict, 
                                                               medications_df, medications_data_dict, False)
    
    
    
    
    
    feature_cols = parse_feature_cols(features_data_dict)
    id_cols = parse_id_cols(features_data_dict)
    time_col = parse_time_col(features_data_dict)

    # sort by ids and timestamp
    features_df.sort_values(by=id_cols+[time_col], inplace=True)
    outcomes_df.sort_values(by=id_cols, inplace=True)
    
    features_df = features_df.reset_index(drop=True)
    outcomes_df = outcomes_df.reset_index(drop=True)
    
    
    discretized_window_length=12
    print('Discretizing to bins of %d hours'%discretized_window_length)
    # get the fenceposts
    fp = get_fenceposts(features_df, id_cols)
    n_iters = len(fp)-1
    timestamp_arr = np.asarray(features_df[time_col].values.copy(), dtype=np.float32)
    
    
    outcome_seq_duration_col = 'stay_length'
    outcome_col = 'clinical_deterioration_outcome'
    feat_arr_per_seq = list()
    windows_per_seq = list()
    outcomes_per_seq = list()
    durations_per_seq = list()
    missingness_density_per_seq = list()
    ids_per_seq = list()
    adm_ts_per_seq = list()
    pbar=ProgressBar()
    t_start = -24
    prediction_horizon=24
    max_hrs_data_observed=504
    
    F = len(feature_cols)
    n_seqs = len(fp) - 1
    
    
    start_time_sec = time.time()
    for p in pbar(range(n_seqs)):
        
        
        # Get features and times for the current fencepost
        fp_start = fp[p]
        fp_end = fp[p+1]
        
        
        # get the timestamps for this fp
        t = features_df.iloc[fp_start:fp_end][time_col].values
        
        # get outcome of current sequence
        cur_final_outcome = outcomes_df[outcome_col].values[p]
        cur_seq_duration = outcomes_df[outcome_seq_duration_col].values[p]
        cur_timestamp_arr = timestamp_arr[fp_start:fp_end]
        cur_features_arr = features_df[feature_cols].values[fp_start:fp_end]
        
        cur_id_df = features_df[id_cols].iloc[fp_start:fp_end].drop_duplicates(subset=id_cols)
        
        curr_adm_timestamp = features_df.iloc[fp_start]['admission_timestamp']
        
        # get the time bins
        t_end = min(cur_seq_duration, max_hrs_data_observed)
        
        window_ends = np.arange(
        t_start + discretized_window_length,
        t_end + 0.01 * discretized_window_length,
        discretized_window_length)
        
        window_starts = window_ends-discretized_window_length
        
        W = len(window_ends)
        window_features_WF = np.zeros([W, F], dtype=np.float32)
        window_starts_stops_W2 = np.zeros([W, 2], dtype=np.float32)
        if outcomes_df is not None:
            window_outcomes_W1 = np.zeros([W, 1], dtype=np.int64)
        
        for ww, window_end in enumerate(window_ends):
            window_starts_stops_W2[ww, 0] = window_starts[ww]
            window_starts_stops_W2[ww, 1] = window_end
            
            cur_dynamic_idx = (cur_timestamp_arr>window_starts[ww])&(cur_timestamp_arr<=window_end)
            cur_dynamic_timestamp_arr = cur_timestamp_arr[cur_dynamic_idx]
            cur_dynamic_features_arr = cur_features_arr[cur_dynamic_idx]
            
            
            window_features_WF[ww, :] = np.nanmean(cur_dynamic_features_arr, axis=0)
            
            if outcomes_df is not None:
                # Determine the outcome for this window
                # Set outcome as final outcome if within the provided horizon
                # Otherwise, set to zero
                if window_end >= cur_seq_duration - prediction_horizon:
                    window_outcomes_W1[ww] = cur_final_outcome
                else:
                    window_outcomes_W1[ww] = 0
        
        # Append all windows from this sequence to the big lists
        feat_arr_per_seq.append(window_features_WF)
        windows_per_seq.append(window_starts_stops_W2)
        ids_per_seq.append(np.tile(cur_id_df.values[0], (W, 1)))
        adm_ts_per_seq.append(np.tile(curr_adm_timestamp, (W, 1)))
        durations_per_seq.append(np.tile(cur_seq_duration, (W, 1)))
        
        
        if outcomes_df is not None:
            outcomes_per_seq.append(window_outcomes_W1)

    # Produce final data frames
    features_df = pd.DataFrame(np.vstack(feat_arr_per_seq), columns=feature_cols)    
    ids_df = pd.DataFrame(np.vstack(ids_per_seq), columns=id_cols) 
    adm_ts_df = pd.DataFrame(np.vstack(adm_ts_per_seq), columns=['admission_timestamp']) 
    windows_df = pd.DataFrame(np.vstack(windows_per_seq), columns=['start', 'stop'])
    all_features_df = pd.concat([ids_df, windows_df, features_df, adm_ts_df], axis=1)

    if outcomes_df is not None:
        durations_df = pd.DataFrame(np.vstack(durations_per_seq), columns=[outcome_seq_duration_col])
        outcomes_df = pd.DataFrame(np.vstack(outcomes_per_seq), columns=[outcome_col])
        all_outcomes_df = pd.concat([ids_df, windows_df, durations_df, outcomes_df, adm_ts_df], axis=1)
    else:
        durations_df = pd.DataFrame(np.vstack(durations_per_seq), columns=[outcome_seq_duration_col])
        all_outcomes_df = pd.concat([ids_df, windows_df, durations_df], axis=1)

    seq_lengths = np.vstack([a[0] for a in durations_per_seq])
    elapsed_time_sec = time.time() - start_time_sec
    
    print('-----------------------------------------')
    print('Processed %d sequences of duration %.1f-%.1f in %.1f sec' % (
        n_seqs,
        np.percentile(seq_lengths, 5),
        np.percentile(seq_lengths, 95),
        elapsed_time_sec,
        ))      
    
    features_csv_filename = os.path.join(args.output_dir, 'features_per_tstep.csv.gz')
    features_data_dict_filename = os.path.join(args.output_dir, 'features_dict.json')
    print('Saving features to :\n%s \n%s'%(features_csv_filename, features_data_dict_filename))
    all_features_df.to_csv(features_csv_filename, index=False, compression='gzip') 
    with open(features_data_dict_filename, 'w') as f:
        json.dump(features_data_dict, f, indent=4)
    
    outcomes_csv_filename = os.path.join(args.output_dir, 'outcomes_per_tstep.csv.gz')
    outcomes_data_dict_filename = os.path.join(args.output_dir, 'outcomes_dict.json')
    print('Saving outcomes to :\n%s \n%s'%(outcomes_csv_filename, outcomes_data_dict_filename))
    all_outcomes_df.to_csv(outcomes_csv_filename, index=False, compression='gzip')
    with open(outcomes_data_dict_filename, 'w') as f:
        json.dump(outcomes_data_dict, f, indent=4)    