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
    
