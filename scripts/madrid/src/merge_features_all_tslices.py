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

def merge_data_dicts(data_dicts_list):
    # get a single consolidated data dict for all features and another for outcomes
    # combine all the labs, demographics and vitals jsons into a single json
    features_data_dict = dict()
    features_data_dict['schema']= dict()
    
    features_dict_merged = []
    for data_dict in data_dicts_list:
        features_dict_merged += data_dict['schema']['fields']  
    
    feat_names = list()
    features_data_dict['schema']['fields'] = []
    for feat_dict in features_dict_merged:
        if feat_dict['name'] not in feat_names:
            features_data_dict['schema']['fields'].append(feat_dict)
            feat_names.append(feat_dict['name'])
    return features_data_dict
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tslice_folder', type=str, 
            help='folder where collapsed features and static features from each tslice are stored')
    parser.add_argument('--tslice_list', type=str, 
            help='list of all the tslices used for training the classifier')
    parser.add_argument('--static_data_dict_dir', type=str,
            help='directory where data dict for demographics and outcomes')
    parser.add_argument('--output_dir',  type=str,
        help='folder to save merged features and outcomes from all tslices')

    args = parser.parse_args()
    
    # get all the collapsed labs, collapsed vitals, demographics and outcomes data dicts
    with open(os.path.join(args.static_data_dict_dir, 'Spec-Demographics.json'), 'r') as f1:
        demographics_data_dict = json.load(f1)
    demographics_data_dict['fields'] = demographics_data_dict['schema']['fields']

    with open(os.path.join(args.static_data_dict_dir, 'Spec-Outcomes_TransferToICU.json'), 'r') as f2:
        outcomes_data_dict = json.load(f2)
    
    id_cols = parse_id_cols(demographics_data_dict)
    # get all the collapsed labs, collapsed vitals, demographics and outcomes in all the tslice folders
    print('Merging collapsed vitals, collapsed labs, demographics and outcomes in all the tslice folders = %s into a single features table and a single outcomes table...'%args.tslice_list)
    features_df_all_slices_list = list()
    outcomes_df_all_slices_list = list()
    mews_df_all_slices_list = list()
    for tslice in args.tslice_list.split(' '):
        curr_tslice_folder = args.tslice_folder+tslice
        collapsed_labs_df = pd.read_csv(os.path.join(curr_tslice_folder, 'CollapsedLabsPerSequence.csv'))
        collapsed_vitals_df = pd.read_csv(os.path.join(curr_tslice_folder, 'CollapsedVitalsPerSequence.csv'))
        demographics_df = pd.read_csv(os.path.join(curr_tslice_folder, 'demographics_before_icu_filtered_%s_hours.csv'%tslice))
        mews_df = pd.read_csv(os.path.join(curr_tslice_folder, 'MewsScoresPerSequence.csv'))
        collapsed_vitals_labs_df = pd.merge(collapsed_vitals_df, collapsed_labs_df, on=id_cols, how='inner')

        # merge the collapsed feaatures and static features in each tslice
        features_df = pd.merge(collapsed_vitals_labs_df, demographics_df, on=id_cols, how='inner')
        outcomes_df = pd.read_csv(os.path.join(curr_tslice_folder, 'clinical_deterioration_outcomes_filtered_%s_hours.csv'%tslice))
        feature_cols = features_df.columns
        outcome_cols = outcomes_df.columns
        mews_cols = mews_df.columns
        
        # append fearures from all tslices
        features_df_all_slices_list.append(features_df.values)
        outcomes_df_all_slices_list.append(outcomes_df.values)
        mews_df_all_slices_list.append(mews_df.values)
        del collapsed_vitals_labs_df
    
    features_df_all_slices = pd.DataFrame(np.concatenate(features_df_all_slices_list), columns=feature_cols)
    outcomes_df_all_slices = pd.DataFrame(np.concatenate(outcomes_df_all_slices_list), columns=outcome_cols)
    mews_df_all_slices = pd.DataFrame(np.concatenate(mews_df_all_slices_list), columns=mews_cols)
    
    # get collapsed vitals and labs dict
    print('Merging collapsed vitals, collapsed labs, demographics and outcomes data dicts into a single features data dict and a single outcomes data dict...')
    with open(os.path.join(curr_tslice_folder, 'Spec_CollapsedLabsPerSequence.json'), 'r') as f3:
        collapsed_labs_data_dict = json.load(f3)

    with open(os.path.join(curr_tslice_folder, 'Spec_CollapsedVitalsPerSequence.json'), 'r') as f4:
        collapsed_vitals_data_dict = json.load(f4)
    
    with open(os.path.join(curr_tslice_folder, 'Spec_MewsScoresPerSequence.json'), 'r') as f5:
        mews_data_dict = json.load(f5)
        
    # get a single consolidated data dict for all features and another for outcomes
    # combine all the labs, demographics and vitals jsons into a single json
    features_data_dict = dict()
    features_data_dict['schema']= dict()
    features_dict_merged = collapsed_labs_data_dict['schema']['fields'] + collapsed_vitals_data_dict['schema']['fields'] + demographics_data_dict['schema']['fields'] 
    
    feat_names = list()
    features_data_dict['schema']['fields'] = []
    for feat_dict in features_dict_merged:
        if feat_dict['name'] not in feat_names:
            features_data_dict['schema']['fields'].append(feat_dict)
            feat_names.append(feat_dict['name'])

    # convert the features to numpy float 32 to avoid memory issues
    feature_cols = parse_feature_cols(features_data_dict['schema'])
    feature_type_dict = dict.fromkeys(feature_cols)
    for k in feature_type_dict.keys():
        feature_type_dict[k] = np.float32
    features_df_all_slices = features_df_all_slices.astype(feature_type_dict)
    
    # save to disk
    features_csv = os.path.join(args.output_dir, 'features.csv')
    outcomes_csv = os.path.join(args.output_dir, 'outcomes.csv')
    mews_csv = os.path.join(args.output_dir, 'mews.csv')
    features_json = os.path.join(args.output_dir, 'Spec_features.json')
    outcomes_json = os.path.join(args.output_dir, 'Spec_outcomes.json')
    mews_json = os.path.join(args.output_dir, 'Spec_mews.json')
    
    print('saving features and outcomes to :\n%s\n%s\n%s'%(features_csv, outcomes_csv, mews_csv))
    features_df_all_slices.to_csv(features_csv, index=False)
    outcomes_df_all_slices.to_csv(outcomes_csv, index=False)
    mews_df_all_slices.to_csv(mews_csv, index=False)

    print('saving features and outcomes dict to :\n%s\n%s\n%s'%(features_json, outcomes_json, mews_json))
    with open(features_json, "w") as outfile_feats:
        json.dump(features_data_dict, outfile_feats)

    with open(outcomes_json, "w") as outfile_outcomes:
        json.dump(outcomes_data_dict, outfile_outcomes)
    
    with open(mews_json, "w") as outfile_mews:
        json.dump(mews_data_dict, outfile_mews)
