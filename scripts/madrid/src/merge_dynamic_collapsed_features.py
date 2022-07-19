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
from featurize_single_time_series import make_summary_ops

def read_csv_with_float32_dtypes(filename, nrows=None):
    # Sample 100 rows of data to determine dtypes.
    df_test = pd.read_csv(filename, nrows=100)

    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    
    if nrows is None:
        df = pd.read_csv(filename, dtype=float32_cols)
    else:
        df = pd.read_csv(filename, dtype=float32_cols, nrows=nrows)
    
    return df


def convert_float64_cols_to_float32(my_df):
    cols = my_df.select_dtypes(include=[np.float64]).columns
    my_df[cols] = my_df[cols].astype(np.float32)
    
    return my_df

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
    parser.add_argument('--features_to_include',  type=str, default='labs_vitals_medications',
        help='vitals_only/labs_vitals/labs_vitals_medications')
    parser.add_argument('--filename_suffix',  type=str, default='',
        help='suffix of the collapse feature filenames to be merged. for eg CustomTimes_10_6')

    args = parser.parse_args()
    
    
    print('Loading collapsed features...')
    # Get collapsed features
    DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH = args.dynamic_collapsed_features_folder
    
    filename_suffix = args.filename_suffix
    evaluation_times = 'CustomTimes_10_6'
    max_n_rows=900000
    dynamic_collapsed_vitals_df = read_csv_with_float32_dtypes(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
                                                           'CollapsedVitalsDynamic%s.csv.gz'%evaluation_times), nrows=max_n_rows)
    dynamic_collapsed_labs_df = read_csv_with_float32_dtypes(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                                         'CollapsedLabsDynamic%s.csv.gz'%evaluation_times))
    
    # get data dicts of collapsed features
    vitals_dd = load_data_dict_json(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                                 'Spec_CollapsedVitalsDynamic%s.json'%evaluation_times))
    labs_dd = load_data_dict_json(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                               'Spec_CollapsedLabsDynamic%s.json'%evaluation_times))
    demographics_dd = load_data_dict_json(os.path.join(args.static_data_dict_dir,
                                                      'Spec-Demographics.json'))
    outcomes_dd = load_data_dict_json(os.path.join(args.static_data_dict_dir, 'Spec-Outcomes_TransferToICU.json'))
    
    id_cols = parse_id_cols(vitals_dd)

    if args.features_to_include=='labs_vitals_medications':
        print('Merging labs, vitals, medications into a single table of dynamic collapsed features...')
        
        dynamic_collapsed_medications_df = read_csv_with_float32_dtypes(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
                                                                    'CollapsedMedicationsDynamic%s.csv.gz'%evaluation_times))
        medications_dd = load_data_dict_json(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH,
                                                          'Spec_CollapsedMedicationsDynamic%s.json'%evaluation_times))    
        
        
        # merge vitals, labs and medications
        dynamic_collapsed_feats_df = pd.merge(pd.merge(dynamic_collapsed_vitals_df, dynamic_collapsed_labs_df, 
                                              on=id_cols+['start', 'stop'], how='left'), 
                                              dynamic_collapsed_medications_df, on=id_cols+['start', 'stop'], 
                                              how='left')
        
        # merge the dynamic collapsed labs, vitals, medications and demographics into a single features data dict
        features_dd = merge_data_dicts([vitals_dd, labs_dd, medications_dd, demographics_dd]) 
        del dynamic_collapsed_vitals_df, dynamic_collapsed_labs_df, dynamic_collapsed_medications_df
        
    elif args.features_to_include=='labs_vitals':
        print('Merging labs and vitals into a single table of dynamic collapsed features...')
        # merge vitals, labs and medications
        dynamic_collapsed_feats_df = pd.merge(dynamic_collapsed_vitals_df, dynamic_collapsed_labs_df, 
                                              on=id_cols+['start', 'stop'], how='left')
        
        # merge the dynamic collapsed labs, vitals, medications and demographics into a single features data dict
        features_dd = merge_data_dicts([vitals_dd, labs_dd, demographics_dd]) 
        del dynamic_collapsed_vitals_df, dynamic_collapsed_labs_df    
    
    elif args.features_to_include=='vitals_only':
        dynamic_collapsed_feats_df = dynamic_collapsed_vitals_df.copy()
        features_dd = vitals_dd.copy()
    
    print('Merging demographics...')
    # merge demographics
    demographics_df = read_csv_with_float32_dtypes(os.path.join(args.static_data_dict_dir, 'demographics_before_icu.csv.gz'))
    dynamic_collapsed_feats_df = pd.merge(dynamic_collapsed_feats_df, demographics_df, on=id_cols, how='left')
    
    
    # set the nan values to the fill value of the operation, since the those values are unobserved
    SUMMARY_OPERATIONS = make_summary_ops()
    for summary_op, (_, fill_val) in SUMMARY_OPERATIONS.items():
        cur_summary_op_collapsed_df_columns = [col for col in dynamic_collapsed_feats_df.columns if '_'+summary_op+'_' in col]
        if len(cur_summary_op_collapsed_df_columns)>0:
            dynamic_collapsed_feats_df.loc[:, cur_summary_op_collapsed_df_columns] = dynamic_collapsed_feats_df.loc[:,                                                                                cur_summary_op_collapsed_df_columns].fillna(fill_val)
    
    
    ids_df = dynamic_collapsed_feats_df.loc[:, id_cols+['admission_timestamp', 'start', 'stop']].copy()
    # get dynamic outputs
    vitals_output =  read_csv_with_float32_dtypes(os.path.join(DATASET_COLLAPSED_FEAT_DYNAMIC_INPUT_OUTPUT_PATH, 
                                              'OutputsDynamicVitals%s.csv.gz'%evaluation_times), nrows=max_n_rows)
    
    
    del demographics_df
    # Set the dynamic outputs to be same as the vitals dynamic outputs because all stays contain atleast 1 vital
    dynamic_outputs_df = vitals_output.copy()
    
    del vitals_output
    
    # add admission timestamp as a column for outputs for creating train-test splits based on timestamps later
    dynamic_outputs_df=pd.merge(dynamic_outputs_df, ids_df, on=id_cols+['start', 'stop'], how='left')
        
    # save to disk
    features_csv = os.path.join(args.output_dir, 'dynamic_features%s.csv.gz'%filename_suffix)
    outcomes_csv = os.path.join(args.output_dir, 'dynamic_outcomes%s.csv.gz'%filename_suffix)
    features_json = os.path.join(args.output_dir, 'Spec_features%s.json'%filename_suffix)
    outcomes_json = os.path.join(args.output_dir, 'Spec_outcomes%s.json'%filename_suffix)
    
    print('saving features and outcomes to :\n%s\n%s'%(features_csv, outcomes_csv))
    dynamic_collapsed_feats_df.to_csv(features_csv, index=False, compression='gzip')
    dynamic_outputs_df.to_csv(outcomes_csv, index=False, compression='gzip')

    print('saving features and outcomes dict to :\n%s\n%s'%(features_json, outcomes_json))
    with open(features_json, "w") as outfile_feats:
        json.dump(features_dd, outfile_feats)

    with open(outcomes_json, "w") as outfile_outcomes:
        json.dump(outcomes_dd, outfile_outcomes)
    

