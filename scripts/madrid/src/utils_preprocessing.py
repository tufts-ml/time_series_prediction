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
    return features_data_dict


def get_preprocessed_data(preproc_data_dir):
    ''' Get the labs vitals and demographics csvs and their data dicts'''
    labs_dict_path = os.path.join(preproc_data_dir, 'Spec-Labs.json')
    vitals_dict_path = os.path.join(preproc_data_dir, 'Spec-Vitals.json') 
    dem_dict_path = os.path.join(preproc_data_dir, 'Spec-Demographics.json')   
    med_dict_path = os.path.join(preproc_data_dir, 'Spec-Medications.json')
    outcomes_dict_path = os.path.join(preproc_data_dir, 'Spec-Outcomes_TransferToICU.json')

    with open(labs_dict_path, 'r') as f:
        labs_data_dict = json.load(f)
    try:
        labs_data_dict['fields'] = labs_data_dict['schema']['fields']
    except KeyError:
        pass
    
    with open(vitals_dict_path, 'r') as f:
        vitals_data_dict = json.load(f)
    try:
        vitals_data_dict['fields'] = vitals_data_dict['schema']['fields']
    except KeyError:
        pass
    
    with open(dem_dict_path, 'r') as f:
        demographics_data_dict = json.load(f)
    try:
        demographics_data_dict['fields'] = demographics_data_dict['schema']['fields']
    except KeyError:
        pass   
    
    with open(outcomes_dict_path, 'r') as f:
        outcomes_data_dict = json.load(f)
    try:
        outcomes_data_dict['fields'] = outcomes_data_dict['schema']['fields']
    except KeyError:
        pass   
    
    with open(med_dict_path, 'r') as f:
        medications_data_dict = json.load(f)
    try:
        medications_data_dict['fields'] = medications_data_dict['schema']['fields']
    except KeyError:
        pass   

    vitals_df = pd.read_csv(os.path.join(preproc_data_dir, 'vitals_before_icu.csv.gz'))
    labs_df = pd.read_csv(os.path.join(preproc_data_dir, 'labs_before_icu.csv.gz'))
    demographics_df = pd.read_csv(os.path.join(preproc_data_dir, 'demographics_before_icu.csv.gz'))
    medications_df = pd.read_csv(os.path.join(preproc_data_dir, 'medications_before_icu.csv.gz'))
    outcomes_df = pd.read_csv(os.path.join(preproc_data_dir, 'clinical_deterioration_outcomes.csv.gz'))
    
    return labs_df, labs_data_dict, vitals_df, vitals_data_dict, demographics_df, demographics_data_dict, medications_df, medications_data_dict, outcomes_df, outcomes_data_dict


def get_all_features_data(labs_df, labs_data_dict, vitals_df, vitals_data_dict, demographics_df, demographics_data_dict, medications_df, medications_data_dict, include_medications=True):
    '''Returns the merged labs, vitals and demographics features into a single table and the data dict'''

    time_col = parse_time_col(vitals_data_dict)
    id_cols = parse_id_cols(vitals_data_dict)

    # merge the labs, vitals and medications
    
    if include_medications:
        highfreq_df = pd.merge(pd.merge(vitals_df, 
                               labs_df, on=id_cols +[time_col], how='outer'),
                               medications_df, on=id_cols +[time_col], how='outer')
    
        # forward fill medications because the patient is/is not on medication on new time points created by outer join
        medication_features = parse_feature_cols(medications_data_dict)
        highfreq_df[id_cols + medication_features] = highfreq_df[id_cols + medication_features].groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()

        highfreq_df[id_cols + medication_features] = highfreq_df[id_cols + medication_features].fillna(0)
        highfreq_data_dict = merge_data_dicts([labs_data_dict, vitals_data_dict, medications_data_dict])
        
    else :
        highfreq_df = pd.merge(vitals_df, labs_df, on=id_cols +[time_col], how='outer')
        highfreq_data_dict = merge_data_dicts([labs_data_dict, vitals_data_dict])
        
        
    highfreq_data_dict['fields'] = highfreq_data_dict['schema']['fields']
    cols_to_keep = parse_id_cols(highfreq_data_dict) + [parse_time_col(highfreq_data_dict)] + parse_feature_cols(highfreq_data_dict)
    highfreq_df = highfreq_df[cols_to_keep].copy()


    # merge the highfrequency features with the static features
    features_df = pd.merge(highfreq_df, demographics_df, on=id_cols, how='inner')
    features_data_dict = merge_data_dicts([highfreq_data_dict, demographics_data_dict])
    features_data_dict['fields'] = features_data_dict['schema']['fields']

    return features_df, features_data_dict
