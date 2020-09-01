import os
import argparse
import pandas as pd
import numpy as np
from summary_statistics import parse_id_cols, parse_output_cols, parse_feature_cols, parse_time_col
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument('--vitals', type=str, help = "time-series csv containing vitals per timestamp")
    parser.add_argument('--labs', type=str, help = "time-series csv containing labs per timestamp")
    parser.add_argument('--demographics', type=str, help = "csv with demographics infomration of patient per stay")
    parser.add_argument('--outcomes', type=str, help = "csv with a clinincal outcome per stay")
    parser.add_argument('--data_dict', type=str)
    '''
    parser.add_argument('--preproc_data_dir', help = 'directory where the labs, vitals, demographics and outcomes are stored')
    parser.add_argument('--tslice',  type=str, default=2,
        help='''Slice of data to be extracted. If tslice is provided with a % sign (for eg. 20%), 
        then the script extracts the first tslice% data from the stay. If tslice is an int (for eg. 5),
        the the script extracts the first  tslice hrs of data. If tslice is negative (for eg. -5), then 
        the script extracts the data until tslice hours before deterioration/discharge.''')
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    
    labs_dict_path = os.path.join(args.preproc_data_dir, 'Spec-Labs.json')
    vitals_dict_path = os.path.join(args.preproc_data_dir, 'Spec-Vitals.json') 

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

    vitals_df = pd.read_csv(os.path.join(args.preproc_data_dir, 'vitals_before_icu.csv'))
    labs_df = pd.read_csv(os.path.join(args.preproc_data_dir, 'labs_before_icu.csv'))
    demographics_df = pd.read_csv(os.path.join(args.preproc_data_dir, 'demographics_before_icu.csv'))
    outcomes_df = pd.read_csv(os.path.join(args.preproc_data_dir, 'clinical_deterioration_outcomes.csv'))
    
    id_cols = parse_id_cols(vitals_data_dict)
    labs_feature_cols = parse_feature_cols(labs_data_dict)
    vitals_feature_cols = parse_feature_cols(vitals_data_dict)

    # get lengths of stay for each admission
    vitals_df_with_stay_lengths = pd.merge(vitals_df, outcomes_df[id_cols + ['stay_length']], on=id_cols, how='inner')
    labs_df_with_stay_lengths = pd.merge(labs_df, outcomes_df[id_cols + ['stay_length']], on=id_cols, how='inner')
    #demographics_df_with_stay_lengths = pd.merge(demographics_df, outcomes_df[id_cols + ['stay_length']], on=id_cols, how='inner')
    
    # find stays that satisfy minimum stay length
    tstops_df = outcomes_df[id_cols].copy()
    if ('%' in args.tslice):
        min_stay_length = 0
        print('Including EHR measured in first %s percent of patient stays having atleast %s hours of data'%(args.tslice, min_stay_length))
        perc = int(args.tslice[:-1])
        vitals_keep_inds = vitals_df_with_stay_lengths.hours_since_admission <= (perc*vitals_df_with_stay_lengths.stay_length)/100
        labs_keep_inds = labs_df_with_stay_lengths.hours_since_admission <= (perc*labs_df_with_stay_lengths.stay_length)/100
        tstops_df.loc[:, 'tstop'] = (perc*outcomes_df.stay_length)/100
    elif ('-' in args.tslice) :
        min_stay_length = abs(int(args.tslice))
        print('Including EHR measured upto %s hours before clinical deterioration in patients stays having atleast %s hours of data'%(min_stay_length, min_stay_length))
        vitals_keep_inds = vitals_df_with_stay_lengths.hours_since_admission <= (vitals_df_with_stay_lengths.stay_length - min_stay_length)
        labs_keep_inds = labs_df_with_stay_lengths.hours_since_admission <= (labs_df_with_stay_lengths.stay_length - min_stay_length)
        tstops_df.loc[:, 'tstop'] = (outcomes_df.stay_length - min_stay_length)
    else :
        min_stay_length = int(args.tslice)
        print('Including EHR measured within first %s hours of admission in patient stays having atleast %s hours of data'%(min_stay_length, min_stay_length))
        vitals_keep_inds = vitals_df_with_stay_lengths.hours_since_admission <= min_stay_length
        labs_keep_inds = labs_df_with_stay_lengths.hours_since_admission <= min_stay_length
        tstops_df.loc[:, 'tstop'] = min_stay_length
    
    vitals_df_with_stay_lengths = vitals_df_with_stay_lengths.loc[vitals_keep_inds, :].copy().reset_index(drop=True)
    labs_df_with_stay_lengths = labs_df_with_stay_lengths.loc[labs_keep_inds, :].copy().reset_index(drop=True)
    vitals_df_with_stay_lengths = vitals_df_with_stay_lengths[vitals_df_with_stay_lengths.stay_length>=min_stay_length]
    labs_df_with_stay_lengths = labs_df_with_stay_lengths[labs_df_with_stay_lengths.stay_length>=min_stay_length]

    # outer join the labs and vitals to cover same number of stays for vitals and labs
    labs_vitals_merged_df = pd.merge(vitals_df_with_stay_lengths, labs_df_with_stay_lengths, on = id_cols + ['hours_since_admission', 'timestamp'], how='outer')
    filtered_labs_df = labs_vitals_merged_df[id_cols + ['timestamp', 'hours_since_admission'] + labs_feature_cols].copy()
    filtered_vitals_df = labs_vitals_merged_df[id_cols + ['timestamp', 'hours_since_admission'] + vitals_feature_cols].copy()
    filtered_demographics_df = demographics_df[demographics_df.hospital_admission_id.isin(labs_vitals_merged_df.hospital_admission_id.unique())].copy().reset_index(drop=True)
    filtered_outcomes_df = outcomes_df[outcomes_df.hospital_admission_id.isin(labs_vitals_merged_df.hospital_admission_id.unique())].copy().reset_index(drop=True)
    filtered_tstops_df = tstops_df[tstops_df.hospital_admission_id.isin(labs_vitals_merged_df.hospital_admission_id.unique())].copy().reset_index(drop=True)
    
    # sort patient-stay-slice by timestamp
    filtered_vitals_df.sort_values(by=id_cols+['timestamp'], inplace=True)
    filtered_labs_df.sort_values(by=id_cols+['timestamp'], inplace=True)

    # save to csv
    vitals_csv_path = os.path.join(args.output_dir, 'vitals_before_icu_filtered_{tslice}_hours.csv'.format(
        tslice=args.tslice))
    labs_csv_path = os.path.join(args.output_dir, 'labs_before_icu_filtered_{tslice}_hours.csv'.format(
        tslice=args.tslice))    
    dem_csv_path = os.path.join(args.output_dir, 'demographics_before_icu_filtered_{tslice}_hours.csv'.format(
        tslice=args.tslice))    
    y_csv_path = os.path.join(args.output_dir, 'clinical_deterioration_outcomes_filtered_{tslice}_hours.csv'.format(
        tslice=args.tslice))
    tstops_csv_path = os.path.join(args.output_dir, 'tstops_filtered_{tslice}_hours.csv'.format(
        tslice=args.tslice))
    filtered_vitals_df.to_csv(vitals_csv_path, index=False)
    filtered_labs_df.to_csv(labs_csv_path, index=False)
    filtered_demographics_df.to_csv(dem_csv_path, index=False)
    filtered_outcomes_df.to_csv(y_csv_path, index=False)
    filtered_tstops_df.to_csv(tstops_csv_path, index=False)

    print('Done! Wrote to CSV file:\n%s\n%s\n%s\n%s' % (vitals_csv_path, labs_csv_path, dem_csv_path, y_csv_path))
