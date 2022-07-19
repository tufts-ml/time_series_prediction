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
from feature_transformation import (parse_id_cols, parse_output_cols, parse_feature_cols, parse_time_cols)
from utils import load_data_dict_json



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--preproc_data_dir', help = 'directory where the features are stored')
    parser.add_argument('--tslice',  type=str, default=2,
        help='''Slice of data to be extracted. If tslice is provided with a % sign (for eg. 20%), 
        then the script extracts the first tslice% data from the stay. If tslice is an int (for eg. 5),
        the the script extracts the first  tslice hrs of data. If tslice is negative (for eg. -5), then 
        the script extracts the data until tslice hours before deterioration/discharge.''')
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    features_df = pd.read_csv(os.path.join(args.preproc_data_dir, 'features_per_tstep.csv'))
    features_data_dict = load_data_dict_json(os.path.join(args.preproc_data_dir, 'Spec_FeaturesPerTimestep.json'))
    outcomes_df = pd.read_csv(os.path.join(args.preproc_data_dir, 'outcomes_per_seq.csv'))
        
    
    id_cols = parse_id_cols(features_data_dict)
    feature_cols = parse_feature_cols(features_data_dict)
    time_col = 'hours_in' # TODO figure out why the parsing of the time column isnt working here
    
    # get lengths of stay for each admission
    stay_lengths_df = features_df[id_cols + [time_col]].groupby(id_cols, as_index=False).max().rename(columns={time_col:'stay_length'})
    features_df_with_stay_lengths = pd.merge(features_df, stay_lengths_df, on=id_cols, how='left')
    outcomes_df = pd.merge(outcomes_df, stay_lengths_df, on=id_cols, how='left')
    
    # find stays that satisfy minimum stay length
    censor_start=504
    tstops_df = outcomes_df[id_cols].copy()
    if ('%' in args.tslice):
        min_stay_length = 0
        print('Including EHR measured in first %s percent of patient stays having atleast %s hours of data'%(args.tslice, min_stay_length))
        perc = int(args.tslice[:-1])
        features_keep_inds = features_df_with_stay_lengths[time_col] <= np.asarray([min(censor_start, i) for i in (perc*features_df_with_stay_lengths.stay_length)/100])

        tstops_df.loc[:, 'tstop'] = np.asarray([min(censor_start, i) for i in (perc*outcomes_df.stay_length)/100])
    
    elif ('-' in args.tslice) :
        min_stay_length = abs(int(args.tslice))
        print('Including EHR measured upto %s hours death in patients stays having atleast %s hours of data'%(min_stay_length, min_stay_length))
        features_keep_inds = features_df_with_stay_lengths[time_col] <= np.asarray([min(censor_start, i) for i in (features_df_with_stay_lengths.stay_length - min_stay_length)])
        tstops_df.loc[:, 'tstop'] = np.asarray([min(censor_start, i) for i in (outcomes_df.stay_length - min_stay_length)])
    
    
    else :
        min_stay_length = int(args.tslice)
        print('Including EHR measured within first %s hours of admission in patient stays having atleast %s hours of data'%(min_stay_length, min_stay_length))
        features_keep_inds = features_df_with_stay_lengths[time_col] <= min(censor_start, min_stay_length)
        tstops_df.loc[:, 'tstop'] = min(censor_start, min_stay_length)
    
    
    features_df_with_stay_lengths = features_df_with_stay_lengths.loc[features_keep_inds, :].copy().reset_index(drop=True)
    features_df_with_stay_lengths = features_df_with_stay_lengths[features_df_with_stay_lengths.stay_length>=min_stay_length]
    
    features_df_with_stay_lengths[feature_cols] = features_df_with_stay_lengths[feature_cols].astype(np.float32)
    
    
    # remove the stay length column
    filtered_features_df = features_df_with_stay_lengths[id_cols + [time_col] + feature_cols].copy()
    
    # keep only the outcomes for the patients who satisfy min stay length
    filtered_outcomes_df = outcomes_df[outcomes_df.hadm_id.isin(filtered_features_df.hadm_id.unique())].copy().reset_index(drop=True)
    filtered_tstops_df = tstops_df[tstops_df.hadm_id.isin(filtered_features_df.hadm_id.unique())].copy().reset_index(drop=True)
    
    # sort patient-stay-slice by timestamp
    filtered_features_df.sort_values(by=id_cols+[time_col], inplace=True)
    filtered_outcomes_df.sort_values(by=id_cols, inplace=True)
    
    # save to csv
    features_csv_path = os.path.join(args.output_dir, 'features_before_death_filtered_{tslice}_hours.csv'.format(
        tslice=args.tslice))
    y_csv_path = os.path.join(args.output_dir, 'outcomes_filtered_{tslice}_hours.csv'.format(
        tslice=args.tslice))
    tstops_csv_path = os.path.join(args.output_dir, 'tstops_filtered_{tslice}_hours.csv'.format(
        tslice=args.tslice))
    filtered_features_df.to_csv(features_csv_path, index=False)
    filtered_outcomes_df.to_csv(y_csv_path, index=False)
    filtered_tstops_df.to_csv(tstops_csv_path, index=False)

    print('Done! Wrote to CSV file:\n%s\n%s' % (features_csv_path, y_csv_path))