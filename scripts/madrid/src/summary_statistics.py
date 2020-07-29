import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import GroupShuffleSplit
import copy
from progressbar import ProgressBar
import warnings
import sys

warnings.filterwarnings("error")

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from split_dataset import (Splitter, split_dataframe_by_keys)

def parse_id_cols(data_dict):
    cols = []
    for col in data_dict['fields']:
        if 'role' in col and (col['role'] == 'id' or 
                              col['role'] == 'key'):
            cols.append(col['name'])
    return cols

def parse_output_cols(data_dict):
    cols = []
    for col in data_dict['fields']:
        if 'role' in col and (col['role'] == 'outcome' or 
                              col['role'] == 'output'):
            cols.append(col['name'])
    return cols

def parse_feature_cols(data_dict):
    non_time_cols = []
    for col in data_dict['fields']:
        if 'role' in col and col['role'] in ('feature', 'measurement', 'covariate'):
            non_time_cols.append(col['name'])
    return non_time_cols

def parse_time_col(data_dict):
    time_cols = []
    for col in data_dict['fields']:
        # TODO avoid hardcoding a column name
        if (col['name'] == 'hours' or col['role'].count('time')):
            time_cols.append(col['name'])
    return time_cols[-1]

def compute_missingness_rate_per_stay(t, x, tstep):
    tstarts = np.arange(0,max(t)+0.00001, tstep)
    n_bins = len(tstarts)
    missing_np = np.zeros([n_bins, x.shape[1]])
    for p in range(n_bins):
        t_start = tstarts[p]
        t_end = tstarts[p]+tstep
        t_idx = (t>=t_start)&(t<t_end)
        curr_x = x[t_idx,:]
        if t_idx.sum() >0:
            # if data is not missing for this patient-stay-tstep, mark as 0, else 1
            missing_np[p,:] = np.isnan(curr_x).all(axis=0)*1.0
        else:
            missing_np[p,:] = np.NaN
    # return average missingness across all tsteps for this  patient-stay
    return np.nanmean(missing_np, axis=0)
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc_data_dir')
    parser.add_argument('--data_dicts_dir')
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--group_cols')
    parser.add_argument('--output_dir')

    args = parser.parse_args()
    
    df_vitals = pd.read_csv(os.path.join(args.preproc_data_dir, 'vitals_before_icu.csv' ))

    df_labs = pd.read_csv(os.path.join(args.preproc_data_dir, 'labs_before_icu.csv'))

    df_demographics = pd.read_csv(os.path.join(args.preproc_data_dir, 'demographics_before_icu.csv'))

    df_transfer_to_icu_outcomes = pd.read_csv(os.path.join(args.preproc_data_dir,'clinical_deterioration_outcomes.csv'))
   
    vitals_data_dict_json = os.path.join(args.data_dicts_dir, 'Spec-Vitals.json')
    
    with open(vitals_data_dict_json, 'r') as f:
        vitals_data_dict = json.load(f)
        try:
            vitals_data_dict['fields'] = vitals_data_dict['schema']['fields']
        except KeyError:
            pass

    id_cols = parse_id_cols(vitals_data_dict)
    vitals = parse_feature_cols(vitals_data_dict)

    # compute missingness per stay
    vital_counts_per_stay_df = df_vitals.groupby(id_cols).count()
    print('#######################################')
    print('MISSINGNESS OVER FULL STAYS : ')
    missing_rate_entire_stay_dict = dict()
    for vital in vitals:
        missing_rate_entire_stay_dict[vital] = ((vital_counts_per_stay_df[vital]==0).sum())/vital_counts_per_stay_df.shape[0]
    missing_rate_entire_stay_series = pd.Series(missing_rate_entire_stay_dict)
    print(missing_rate_entire_stay_series)
    '''
    time_col = parse_time_col(vitals_data_dict)
    timestamp_arr = np.asarray(df_vitals[time_col].values.copy(), dtype=np.float64)
    features_arr = df_vitals[vitals].values
    # get per tstep hour missing rates
    tstep = 12
    print('reporting missingness in %s  hour bins'%tstep)
    keys_df = df_vitals[id_cols].copy()
    for col in id_cols:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    fp = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0).any(axis=1)), keys_df.shape[0]]) 
    n_stays = len(fp)-1
    missing_rates_all_stays =  np.zeros([n_stays, len(vitals)])
    pbar = ProgressBar()
    

    for stay in pbar(range(n_stays)):
        # get the data for the current fencepost
        fp_start = fp[stay]
        fp_end = fp[stay+1]
        cur_feat_arr = features_arr[fp_start:fp_end,:].copy()
        cur_timestamp_arr = timestamp_arr[fp_start:fp_end]

        missing_rates_all_stays[stay, :] = compute_missingness_rate_per_stay(cur_timestamp_arr, cur_feat_arr, tstep)
    '''
    
    #from IPython import embed; embed() 
    print('#######################################')
    print('Printing summary statistics for vitals')
    vitals_summary_df = pd.DataFrame()
    for vital in vitals:
        curr_vital_series = df_vitals[vital]
        vitals_summary_df.loc[vital, 'min'] = curr_vital_series.min()
        vitals_summary_df.loc[vital, 'max'] = curr_vital_series.max()
        vitals_summary_df.loc[vital, '5%'] = np.percentile(curr_vital_series[curr_vital_series.notnull()], 5)
        vitals_summary_df.loc[vital, '95%'] = np.percentile(curr_vital_series[curr_vital_series.notnull()], 95)
        vitals_summary_df.loc[vital, 'median'] = curr_vital_series.median()
    
    vitals_summary_df.loc[:,'missing_rate'] = missing_rate_entire_stay_series
    print(vitals_summary_df[['min', '5%', 'median', '95%', 'max', 'missing_rate']])

    print('#######################################')
    print('Getting train, val, test split statistics')
    
    validation_size=0.15
    train_df, test_df = split_dataframe_by_keys(
        df_vitals, cols_to_group=args.group_cols, size=args.test_size, random_state=args.random_seed) 
    train_df, validation_df = split_dataframe_by_keys(
        train_df, cols_to_group=args.group_cols, size=validation_size, random_state=args.random_seed)

    outcomes_train_df, outcomes_test_df = split_dataframe_by_keys(
        df_transfer_to_icu_outcomes, cols_to_group=args.group_cols, size=args.test_size, random_state=args.random_seed)
    outcomes_train_df, outcomes_validation_df = split_dataframe_by_keys(
        outcomes_train_df, cols_to_group=args.group_cols, size=validation_size, random_state=args.random_seed)
    
    for split_name, df, outcome_df in [('TRAIN', train_df, outcomes_train_df), ('VALIDATION', validation_df, outcomes_validation_df), ('TEST', test_df, outcomes_test_df)]:

        print('----------------------------------------------------')
        print('%s :'%(split_name))
        print('TOTAL PATIENTS : %s, TOTAL STAYS : %s, TOTAL CLINICAL DETERIORATIONS : %s, OUTCOME FREQUENCY : %s'%(len(df.patient_id.unique()), len(df.hospital_admission_id.unique()), 
            outcome_df.clinical_deterioration_outcome.sum(), outcome_df.clinical_deterioration_outcome.sum()/len(df.hospital_admission_id.unique())))
        stay_lengths = outcome_df.stay_length
        stay_lengths_outcome_0 = stay_lengths[outcome_df.clinical_deterioration_outcome==0]
        stay_lengths_outcome_1 = stay_lengths[outcome_df.clinical_deterioration_outcome==1]
        print('lengths of stay outcome 0:  %.2f(%.5f - %.5f)'%( np.median(stay_lengths_outcome_0), np.percentile(stay_lengths_outcome_0, 5), np.percentile(stay_lengths_outcome_0, 95)))
        print('lengths of stay outcome 1:  %.2f(%.5f - %.5f)'%( np.median(stay_lengths_outcome_1), np.percentile(stay_lengths_outcome_1, 5), np.percentile(stay_lengths_outcome_1, 95)))
        print('number of outcome 1 ICU transfers : %.1f'%((outcome_df.transfer_to_ICU_outcome==1).sum()))
        print('number of outcome 1 non-ICU transfer deaths : %.1f'%((outcome_df.clinical_deterioration_outcome - outcome_df.transfer_to_ICU_outcome).sum()))
        print('----------------------------------------------------')


