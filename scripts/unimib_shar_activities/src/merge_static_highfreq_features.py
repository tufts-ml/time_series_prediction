import argparse
import json
import numpy as np
import os
import pandas as pd
import sys
DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from utils import merge_data_dicts, load_data_dict_json
from feature_transformation import (parse_id_cols, remove_col_names_from_list_if_not_in_df, parse_time_cols, parse_feature_cols)


if __name__ == '__main__':

    # Parse pre-specified command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--static_features_csv', type=str, required=True)
    parser.add_argument('--static_features_json', type=str, required=True)
    parser.add_argument('--highfreq_features_csv', type=str, required=True)
    parser.add_argument('--highfreq_features_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=False)
    args=parser.parse_args()
    
    print('merging static and high-frequency features..')
    # load static features
    static_df = pd.read_csv(args.static_features_csv) 
    static_data_dict = load_data_dict_json(args.static_features_json)
    
    # load high-frequency features
    highfreq_df = pd.read_csv(args.highfreq_features_csv)
    highfreq_data_dict = load_data_dict_json(args.highfreq_features_json)
    
    # merge the static and high-freq_df
    merge_cols = parse_id_cols(static_data_dict)
    merged_df = pd.merge(static_df, highfreq_df, on=merge_cols)
    merged_data_dict = merge_data_dicts([static_data_dict, highfreq_data_dict])
    
    print(merged_df.head())
    
    # save 
    output_csv_filename = os.path.join(args.output_dir, 'features_per_sequence.csv')
    merged_df.to_csv(output_csv_filename, index=False)
    
    output_data_dict_filename = os.path.join(args.output_dir, 'Spec_FeaturesPerSequence.json')
    with open(output_data_dict_filename, "w") as outfile:
        json.dump(merged_data_dict, outfile)
    