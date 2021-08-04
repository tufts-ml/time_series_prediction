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

def read_csv_with_float32_dtypes(filename):
    # Sample 100 rows of data to determine dtypes.
    df_test = pd.read_csv(filename, nrows=100)

    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    df = pd.read_csv(filename, dtype=float32_cols)
    
    return df


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
    
    features_data_dict['fields'] = features_data_dict['schema']['fields']
    return features_data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamic_collapsed_features_folder', type=str, 
            help='folder where collapsed features from each tslice are stored')
    parser.add_argument('--static_data_dict_dir', type=str,
            help='directory where data dict for demographics and outcomes')
    parser.add_argument('--output_dir',  type=str,
        help='folder to save merged features and outcomes from all tslices')

    args = parser.parse_args()
    
    print('Loading collapsed labs, vitals, and medications...')
    # Get collapsed features
    DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH = args.dynamic_collapsed_features_folder
    
    dynamic_collapsed_vitals_df = pd.read_csv(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
                                                           'CollapsedVitalsDynamic.csv.gz'))
    dynamic_collapsed_labs_df = pd.read_csv(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                                         'CollapsedLabsDynamic.csv.gz'))
    dynamic_collapsed_medications_df = pd.read_csv(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
                                                                'CollapsedMedicationsDynamic.csv.gz'))
    demographics_df = pd.read_csv(os.path.join(args.static_data_dict_dir, 'demographics_before_icu.csv.gz'))

    
    # get data dicts of collapsed features
    vitals_dd = load_data_dict_json(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                                 'Spec_CollapsedVitalsDynamic.json'))
    labs_dd = load_data_dict_json(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                               'Spec_CollapsedLabsDynamic.json'))
    medications_dd = load_data_dict_json(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
                                                      'Spec_CollapsedMedicationsDynamic.json'))
    demographics_dd = load_data_dict_json(os.path.join(args.static_data_dict_dir,
                                                      'Spec-Demographics.json'))
    outcomes_dd = load_data_dict_json(os.path.join(args.static_data_dict_dir, 'Spec-Outcomes_TransferToICU.json'))
    
    print('Merging labs, vitals. medications into a single table of dynamic collapsed features...')
    # get dynamic outputs
    vitals_output =  pd.read_csv(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                              'OutputsDynamicVitals.csv.gz'))
    labs_output =  pd.read_csv(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                            'OutputsDynamicLabs.csv.gz'))
    medications_output =  pd.read_csv(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                                   'OutputsDynamicMedications.csv.gz'))
    
    # merge vitals, labs and medications
    id_cols = parse_id_cols(vitals_dd)
    dynamic_collapsed_feats_df = pd.merge(pd.merge(dynamic_collapsed_vitals_df, dynamic_collapsed_labs_df, 
                                          on=id_cols+['window_start', 'window_end'], how='left'), 
                                          dynamic_collapsed_medications_df, on=id_cols+['window_start', 'window_end'], 
                                          how='left')

    # since the nan values are all unobserved, set to 0
    dynamic_collapsed_feats_df[dynamic_collapsed_feats_df.isna()]=0.0
    
    print('Merging demographics...')
    # merge demographics
    dynamic_collapsed_feats_df = pd.merge(dynamic_collapsed_feats_df, demographics_df, on=id_cols, how='left')

    # Set the dynamic outputs to be same as the vitals dynamic outputs because all stays contain atleast 1 vital
    dynamic_outputs_df = vitals_output.copy()
    
    # add admission timestamp as a column for outputs for creating train-test splits based on timestamps later
    dynamic_outputs_df=pd.merge(dynamic_outputs_df, dynamic_collapsed_feats_df[id_cols+['admission_timestamp', 'window_start', 'window_end']], on=id_cols+['window_start', 'window_end'], how='left')
    
    
    # merge the dynamic collapsed labs, vitals, medications and demographics into a single features data dict
    features_dd = merge_data_dicts([vitals_dd, labs_dd, medications_dd, demographics_dd]) 
        
    # save to disk
    features_csv = os.path.join(args.output_dir, 'dynamic_features.csv.gz')
    outcomes_csv = os.path.join(args.output_dir, 'dynamic_outcomes.csv.gz')
    features_json = os.path.join(args.output_dir, 'Spec_features.json')
    outcomes_json = os.path.join(args.output_dir, 'Spec_outcomes.json')
    
    print('saving features and outcomes to :\n%s\n%s'%(features_csv, outcomes_csv))
    dynamic_collapsed_feats_df.to_csv(features_csv, index=False, compression='gzip')
    dynamic_outputs_df.to_csv(outcomes_csv, index=False, compression='gzip')

    print('saving features and outcomes dict to :\n%s\n%s'%(features_json, outcomes_json))
    with open(features_json, "w") as outfile_feats:
        json.dump(features_dd, outfile_feats)

    with open(outcomes_json, "w") as outfile_outcomes:
        json.dump(outcomes_dd, outfile_outcomes)
    

