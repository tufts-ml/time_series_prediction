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
from filter_admissions_by_tslice import get_preprocessed_data
from merge_features_all_tslices import merge_data_dicts, get_all_features_data
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
    
    # discretize to bins 
    bin_hrs = 16
    
    print('Discretizing to bins of %d hours'%bin_hrs)
    # get the fenceposts
    fp = get_fenceposts(features_df, id_cols)
    n_iters = len(fp)-1
    
    features_per_bin_list = []
    times_per_bin_list = []
    ids_per_bin_list = []
    pbar=ProgressBar()
    
    for p in pbar(range(n_iters)):
        
        # get the timestamps for this fp
        t = features_df.iloc[fp[p]:fp[p+1]][time_col].values
        
        # get the features values
        features_ND = features_df.iloc[fp[p]:fp[p+1]][feature_cols].values
        ids = features_df.iloc[fp[p]][id_cols].values
        
        # get the time bins
        t_bins = np.arange(0, max(t)+bin_hrs, bin_hrs)
        
        # assign each timepoint to bins
        t_bin_inds = np.searchsorted(t_bins, t)
        
        for bin_ind in range(1, max(t_bin_inds)+1):
            curr_bin_t = t_bins[bin_ind]
            keep_inds = t_bin_inds==bin_ind
            curr_bin_features = features_ND[keep_inds]
            if keep_inds.sum()>0:
                features_per_bin_list.append(np.nanmean(curr_bin_features, axis=0))
            else:
                features_per_bin_list.append(np.nan*np.zeros(125))
            times_per_bin_list.append(curr_bin_t)
            ids_per_bin_list.append(ids)
        
    features_np = np.hstack([np.vstack(ids_per_bin_list), np.vstack(times_per_bin_list), np.vstack(features_per_bin_list)])              
    features_df = pd.DataFrame(features_np, columns=id_cols+[time_col]+feature_cols)
    
    
    keep_ids_df = features_df[id_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)
    outcomes_df = pd.merge(keep_ids_df, outcomes_df, on=id_cols, how='inner')
    
    features_csv_filename = os.path.join(args.output_dir, 'features.csv.gz')
    features_data_dict_filename = os.path.join(args.output_dir, 'features_dict.json')
    print('Saving features to :\n%s \n%s'%(features_csv_filename, features_data_dict_filename))
    features_df.to_csv(features_csv_filename, index=False, compression='gzip') 
    with open(features_data_dict_filename, 'w') as f:
        json.dump(features_data_dict, f, indent=4)
    
    outcomes_csv_filename = os.path.join(args.output_dir, 'outcomes.csv.gz')
    outcomes_data_dict_filename = os.path.join(args.output_dir, 'outcomes_dict.json')
    print('Saving outcomes to :\n%s \n%s'%(outcomes_csv_filename, outcomes_data_dict_filename))
    outcomes_df.to_csv(outcomes_csv_filename, index=False, compression='gzip')
    with open(outcomes_data_dict_filename, 'w') as f:
        json.dump(outcomes_data_dict, f, indent=4)    