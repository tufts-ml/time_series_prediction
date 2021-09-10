import os
import argparse
import pandas as pd
import numpy as np
import sys
import json
DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from feature_transformation import (parse_id_cols, remove_col_names_from_list_if_not_in_df, parse_time_col, parse_feature_cols)
from utils import load_data_dict_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamic_collapsed_features_folder', type=str, 
            help='folder where collapsed features from each tslice are stored')
    parser.add_argument('--static_data_dict_dir', type=str,
            help='directory where data dict for demographics and outcomes')
    parser.add_argument('--output_dir',  type=str,
        help='folder to save merged features and outcomes from all tslices')

    args = parser.parse_args()
    
    print('Loading mews scores...')
    # Get collapsed features
    DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH = args.dynamic_collapsed_features_folder
    
    dynamic_mews_df = pd.read_csv(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
                                                           'MewsDynamic.csv.gz'))
    
    demographics_df = pd.read_csv(os.path.join(args.static_data_dict_dir, 'demographics_before_icu.csv.gz'))

    
    # get data dicts of collapsed features
    demographics_dd = load_data_dict_json(os.path.join(args.static_data_dict_dir,
                                                      'Spec-Demographics.json'))
    outcomes_dd = load_data_dict_json(os.path.join(args.static_data_dict_dir, 'Spec-Outcomes_TransferToICU.json'))
    
    
    # merge vitals, labs and medications
    id_cols = parse_id_cols(demographics_dd)

    
    print('Merging demographics...')
    # merge demographics
    dynamic_mews_df = pd.merge(dynamic_mews_df, demographics_df, on=id_cols, how='left')

    # Set the dynamic outputs to be same as the vitals dynamic outputs because all stays contain atleast 1 vital
    dynamic_outputs_df = pd.read_csv(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                                  'OutputsDynamicMews.csv.gz'))
    
    # add admission timestamp as a column for outputs for creating train-test splits based on timestamps later
    dynamic_outputs_df=pd.merge(dynamic_outputs_df, dynamic_mews_df[id_cols+['admission_timestamp', 'window_start', 'window_end']], on=id_cols+['window_start', 'window_end'], how='left')
    
        
    # save to disk
    mews_features_csv = os.path.join(args.output_dir, 'mews_dynamic_features.csv.gz')
    mews_outcomes_csv = os.path.join(args.output_dir, 'mews_dynamic_outcomes.csv.gz')

    
    print('saving features and outcomes to :\n%s\n%s'%(mews_features_csv, mews_outcomes_csv))
    dynamic_mews_df.to_csv(mews_features_csv, index=False, compression='gzip')
    dynamic_outputs_df.to_csv(mews_outcomes_csv, index=False, compression='gzip')
