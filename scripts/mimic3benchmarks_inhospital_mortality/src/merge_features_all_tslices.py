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
from utils import load_data_dict_json
from feature_transformation import (parse_id_cols, remove_col_names_from_list_if_not_in_df, parse_time_cols, parse_feature_cols)

def read_csv_with_float32_dtypes(filename):
    # Sample 100 rows of data to determine dtypes.
    df_test = pd.read_csv(filename, nrows=100)

    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    df = pd.read_csv(filename, dtype=float32_cols)
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collapsed_tslice_folder', type=str, 
            help='folder where collapsed features from each tslice are stored')
    parser.add_argument('--tslice_folder', type=str, 
            help='folder where raw features and static features from each tslice are stored')
    parser.add_argument('--tslice_list', type=str, 
            help='list of all the tslices used for training the classifier')
    parser.add_argument('--preproc_data_dir', type=str,
            help='directory where data dict for demographics and outcomes')
    parser.add_argument('--output_dir',  type=str,
        help='folder to save merged features and outcomes from all tslices')

    args = parser.parse_args()

#     with open(os.path.join(args.preproc_data_dir, 'Spec_OutcomesPerSequence.json'), 'r') as f2:
#         outcomes_data_dict = json.load(f2)
    
    outcomes_data_dict = load_data_dict_json(os.path.join(args.preproc_data_dir, 'Spec_OutcomesPerSequence.json'))
    id_cols = parse_id_cols(outcomes_data_dict)
    
    # get all the collapsed features and outcomes in all the tslice folders
    print('Merging features in the tslice folders = %s into a single features table and a single outcomes table...'%args.tslice_list)
    features_df_all_slices_list = list()
    outcomes_df_all_slices_list = list()
    for tslice in args.tslice_list.split(' '):
        print('Appending tslice=%s...'%tslice)
        curr_tslice_folder = args.tslice_folder+tslice
        curr_collapsed_tslice_folder = args.collapsed_tslice_folder+tslice
        
        print('Loading collapsed features...')
        collapsed_features_df = read_csv_with_float32_dtypes(os.path.join(curr_collapsed_tslice_folder, 'CollapsedFeaturesPerSequence.csv'))

        outcomes_df = pd.read_csv(os.path.join(curr_tslice_folder, 'outcomes_filtered_%s_hours.csv'%tslice))
        feature_cols = collapsed_features_df.columns
        outcome_cols = outcomes_df.columns

        # append fearures from all tslices
        features_df_all_slices_list.append(collapsed_features_df.values)
        outcomes_df_all_slices_list.append(outcomes_df.values)

    features_df_all_slices = pd.DataFrame(np.concatenate(features_df_all_slices_list), columns=feature_cols)
    outcomes_df_all_slices = pd.DataFrame(np.concatenate(outcomes_df_all_slices_list), columns=outcome_cols)
    
    # get collapsed features dict
    print('Saving merged collapsed features dict...')
    with open(os.path.join(curr_collapsed_tslice_folder, 'Spec_CollapsedFeaturesPerSequence.json'), 'r') as f3:
        features_data_dict = json.load(f3)
    
    # save to disk
    features_csv = os.path.join(args.output_dir, 'features.csv')
    outcomes_csv = os.path.join(args.output_dir, 'outcomes.csv')
    features_json = os.path.join(args.output_dir, 'Spec_features.json')
    outcomes_json = os.path.join(args.output_dir, 'Spec_outcomes.json')
    
    print('saving features and outcomes to :\n%s\n%s'%(features_csv, outcomes_csv))
    features_df_all_slices.to_csv(features_csv, index=False)
    outcomes_df_all_slices.to_csv(outcomes_csv, index=False)

    print('saving features and outcomes dict to :\n%s\n%s'%(features_json, outcomes_json))
    with open(features_json, "w") as outfile_feats:
        json.dump(features_data_dict, outfile_feats)

    with open(outcomes_json, "w") as outfile_outcomes:
        json.dump(outcomes_data_dict, outfile_outcomes)
