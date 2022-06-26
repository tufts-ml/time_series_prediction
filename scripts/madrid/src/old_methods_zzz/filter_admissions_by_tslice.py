import os
import argparse
import pandas as pd
import numpy as np
import sys
DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
import json
from feature_transformation import (parse_id_cols, parse_output_cols, parse_feature_cols, parse_time_col)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc_data_dir', help = 'directory where the labs, vitals, demographics and outcomes are stored')
    parser.add_argument('--tslice',  type=str, default=2,
        help='''Slice of data to be extracted. If tslice is provided with a % sign (for eg. 20%), 
        then the script extracts the first tslice% data from the stay. If tslice is an int (for eg. 5),
        the the script extracts the first  tslice hrs of data. If tslice is negative (for eg. -5), then 
        the script extracts the data until tslice hours before deterioration/discharge.''')
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    labs_df, labs_data_dict, vitals_df, vitals_data_dict, \
    demographics_df, demographics_data_dict, medications_df, medications_data_dict, outcomes_df, outcomes_data_dict = get_preprocessed_data(args.preproc_data_dir)  
        
    id_cols = parse_id_cols(vitals_data_dict)
    labs_feature_cols = parse_feature_cols(labs_data_dict)
    vitals_feature_cols = parse_feature_cols(vitals_data_dict)
    medications_feature_cols = parse_feature_cols(medications_data_dict)

    # get lengths of stay for each admission
    vitals_df_with_stay_lengths = pd.merge(vitals_df, outcomes_df[id_cols + ['stay_length']], on=id_cols, how='inner')
    labs_df_with_stay_lengths = pd.merge(labs_df, outcomes_df[id_cols + ['stay_length']], on=id_cols, how='inner')
    medications_df_with_stay_lengths = pd.merge(medications_df, outcomes_df[id_cols + ['stay_length']], on=id_cols, how='inner')
    #demographics_df_with_stay_lengths = pd.merge(demographics_df, outcomes_df[id_cols + ['stay_length']], on=id_cols, how='inner')
    
    # find stays that satisfy minimum stay length
    censor_start=504
    tstops_df = outcomes_df[id_cols].copy()
    if ('%' in args.tslice):
        min_stay_length = 0
        print('Including EHR measured in first %s percent of patient stays having atleast %s hours of data'%(args.tslice, min_stay_length))
        perc = int(args.tslice[:-1])
        vitals_keep_inds = vitals_df_with_stay_lengths.hours_since_admission <= np.asarray([min(censor_start, i) for i in (perc*vitals_df_with_stay_lengths.stay_length)/100])
        labs_keep_inds = labs_df_with_stay_lengths.hours_since_admission <= np.asarray([min(censor_start, i) for i in (perc*labs_df_with_stay_lengths.stay_length)/100])
        medications_keep_inds = medications_df_with_stay_lengths.hours_since_admission <= np.asarray([min(censor_start, i) for i in (perc*medications_df_with_stay_lengths.stay_length)/100])
        tstops_df.loc[:, 'tstop'] = np.asarray([min(censor_start, i) for i in (perc*outcomes_df.stay_length)/100])
    
    elif ('-' in args.tslice) :
        min_stay_length = abs(int(args.tslice))
        print('Including EHR measured upto %s hours before clinical deterioration in patients stays having atleast %s hours of data'%(min_stay_length, min_stay_length))
        vitals_keep_inds = vitals_df_with_stay_lengths.hours_since_admission <= np.asarray([min(censor_start, i) for i in (vitals_df_with_stay_lengths.stay_length - min_stay_length)])
        labs_keep_inds = labs_df_with_stay_lengths.hours_since_admission <= np.asarray([min(censor_start, i) for i in (labs_df_with_stay_lengths.stay_length - min_stay_length)])
        medications_keep_inds = medications_df_with_stay_lengths.hours_since_admission <= np.asarray([min(censor_start, i) for i in (medications_df_with_stay_lengths.stay_length - min_stay_length)])   
        tstops_df.loc[:, 'tstop'] = np.asarray([min(censor_start, i) for i in (outcomes_df.stay_length - min_stay_length)])
    
    else :
        min_stay_length = int(args.tslice)
        print('Including EHR measured within first %s hours of admission in patient stays having atleast %s hours of data'%(min_stay_length, min_stay_length))
        vitals_keep_inds = vitals_df_with_stay_lengths.hours_since_admission <= min(censor_start, min_stay_length)
        labs_keep_inds = labs_df_with_stay_lengths.hours_since_admission <= min(censor_start, min_stay_length)
        medications_keep_inds = medications_df_with_stay_lengths.hours_since_admission <= min(censor_start, min_stay_length)
        tstops_df.loc[:, 'tstop'] = min(censor_start, min_stay_length)
    
    vitals_df_with_stay_lengths = vitals_df_with_stay_lengths.loc[vitals_keep_inds, :].copy().reset_index(drop=True)
    labs_df_with_stay_lengths = labs_df_with_stay_lengths.loc[labs_keep_inds, :].copy().reset_index(drop=True)
    vitals_df_with_stay_lengths = vitals_df_with_stay_lengths[vitals_df_with_stay_lengths.stay_length>=min_stay_length]
    labs_df_with_stay_lengths = labs_df_with_stay_lengths[labs_df_with_stay_lengths.stay_length>=min_stay_length]
    medications_df_with_stay_lengths = medications_df_with_stay_lengths[medications_df_with_stay_lengths.stay_length>=min_stay_length]
    
    labs_df_with_stay_lengths[labs_feature_cols] = labs_df_with_stay_lengths[labs_feature_cols].astype(np.float32)
    vitals_df_with_stay_lengths[vitals_feature_cols] = vitals_df_with_stay_lengths[vitals_feature_cols].astype(np.float32)
    medications_df_with_stay_lengths[medications_feature_cols] = medications_df_with_stay_lengths[medications_feature_cols].astype(np.int64)
    
    # outer join the labs vitals and medications to cover same number of stays for vitals and labs
    labs_vitals_medications_merged_df = pd.merge(pd.merge(vitals_df_with_stay_lengths, 
                                                          labs_df_with_stay_lengths, 
                                                          on=id_cols + ['hours_since_admission', 'timestamp'], 
                                                          how='outer'),
                                                 medications_df_with_stay_lengths,
                                                 on=id_cols + ['hours_since_admission', 'timestamp'],
                                                 how='outer')                    
    
    

    filtered_labs_df = labs_vitals_medications_merged_df[id_cols + ['timestamp', 'hours_since_admission'] + labs_feature_cols].copy()
    filtered_vitals_df = labs_vitals_medications_merged_df[id_cols + ['timestamp', 'hours_since_admission'] + vitals_feature_cols].copy()
    filtered_medications_df = labs_vitals_medications_merged_df[id_cols + ['timestamp', 'hours_since_admission'] + medications_feature_cols].copy()
    
    # forward fill medications at time points where medications are nan
    filtered_medications_df = filtered_medications_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()
    
    # fill remaining nan medication time points to 0 because patient isnt administered the medication 
    filtered_medications_df = filtered_medications_df.fillna(0)
        
    filtered_demographics_df = demographics_df[demographics_df.hospital_admission_id.isin(labs_vitals_medications_merged_df.hospital_admission_id.unique())].copy().reset_index(drop=True)
    
    
    filtered_outcomes_df = outcomes_df[outcomes_df.hospital_admission_id.isin(labs_vitals_medications_merged_df.hospital_admission_id.unique())].copy().reset_index(drop=True)
    filtered_tstops_df = tstops_df[tstops_df.hospital_admission_id.isin(labs_vitals_medications_merged_df.hospital_admission_id.unique())].copy().reset_index(drop=True)
    
    # sort patient-stay-slice by timestamp
    filtered_vitals_df.sort_values(by=id_cols+['timestamp'], inplace=True)
    filtered_labs_df.sort_values(by=id_cols+['timestamp'], inplace=True)
    filtered_medications_df.sort_values(by=id_cols+['timestamp'], inplace=True)

    # save to csv
    vitals_csv_path = os.path.join(args.output_dir, 'vitals_before_icu_filtered_{tslice}_hours.csv.gz'.format(
        tslice=args.tslice))
    labs_csv_path = os.path.join(args.output_dir, 'labs_before_icu_filtered_{tslice}_hours.csv.gz'.format(
        tslice=args.tslice))    
    dem_csv_path = os.path.join(args.output_dir, 'demographics_before_icu_filtered_{tslice}_hours.csv.gz'.format(
        tslice=args.tslice))   
    med_csv_path = os.path.join(args.output_dir, 'medications_before_icu_filtered_{tslice}_hours.csv.gz'.format(
        tslice=args.tslice))  
    y_csv_path = os.path.join(args.output_dir, 'clinical_deterioration_outcomes_filtered_{tslice}_hours.csv.gz'.format(
        tslice=args.tslice))
    tstops_csv_path = os.path.join(args.output_dir, 'tstops_filtered_{tslice}_hours.csv.gz'.format(
        tslice=args.tslice))
    
    filtered_vitals_df.to_csv(vitals_csv_path, index=False, compression='gzip')
    filtered_labs_df.to_csv(labs_csv_path, index=False, compression='gzip')
    filtered_demographics_df.to_csv(dem_csv_path, index=False, compression='gzip')
    filtered_medications_df.to_csv(med_csv_path, index=False, compression='gzip' )
    filtered_outcomes_df.to_csv(y_csv_path, index=False, compression='gzip')
    filtered_tstops_df.to_csv(tstops_csv_path, index=False, compression='gzip')

    print('Done! Wrote to CSV file:\n%s\n%s\n%s\n%s\n%s' % (vitals_csv_path, labs_csv_path, dem_csv_path, med_csv_path, y_csv_path))
