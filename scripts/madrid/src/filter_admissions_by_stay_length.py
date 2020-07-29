import os
import argparse
import pandas as pd
import numpy as np
from summary_statistics import parse_id_cols, parse_output_cols, parse_feature_cols, parse_time_col
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--outcomes', type=str)
    parser.add_argument('--data_dict', type=str)
    parser.add_argument('--stay_length',  type=int, default=2,
        help='Minimmum length of stay required to be considered as part of cohort')
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    args.data_dict_path = args.data_dict
    with open(args.data_dict_path, 'r') as f:
        args.data_dict = json.load(f)
    try:
        args.data_dict['fields'] = args.data_dict['schema']['fields']
    except KeyError:
        pass

    print('done parsing...')
    print('reading csv...')
    ts_df = pd.read_csv(args.input)
    print('done reading csv...')
    data_dict = None
    outcomes_df = pd.read_csv(args.outcomes)
    
    id_cols = parse_id_cols(args.data_dict)
    feature_cols = parse_feature_cols(args.data_dict)
    time_col = parse_time_col(args.data_dict)
    
    print('Including patient stays lasting atleast %s hours'%args.stay_length)
    # get lengths of stay for each admission
    #stay_lengths = ts_df.groupby(id_cols)['hours_since_admission'].apply(lambda x : max(x)-min(x))
    ts_df_with_stay_lengths = pd.merge(ts_df, outcomes_df[id_cols + ['stay_length']], on=id_cols, how='inner')
    stay_lengths = ts_df_with_stay_lengths.groupby(id_cols)['stay_length'].apply(lambda x : float(x.unique()))
    
    # find stays that satisfy minimum stay length
    filtered_by_stay_length_ids = stay_lengths[stay_lengths.values>=args.stay_length].reset_index()[id_cols]
    # get the admissions that satisfy minimum stay length
    filtered_ts_df = pd.merge(ts_df, filtered_by_stay_length_ids, on=id_cols, how='inner')
    filtered_outcomes_df = pd.merge(outcomes_df, filtered_by_stay_length_ids, on=id_cols, how='inner')
    
    # save to csv
    x_csv_path = os.path.join(args.output_dir, 'vitals_before_icu_filtered_{stay_length}_hours.csv'.format(stay_length=args.stay_length))
    y_csv_path = os.path.join(args.output_dir, 'clinical_deterioration_outcomes_filtered_{stay_length}_hours.csv'.format(stay_length=args.stay_length))
    filtered_ts_df.to_csv(x_csv_path, index=False)
    filtered_outcomes_df.to_csv(y_csv_path, index=False)
    print('Done! Wrote to CSV file:\n%s\n%s' % (x_csv_path, y_csv_path))
