'''
Script to merge the vitals, labs and demographics train test splits into single train and test sets
'''
import os
import argparse
import pandas as pd
import numpy as np
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test_split_dir', type=str)
    parser.add_argument('--train_csv_filename', type=str)
    parser.add_argument('--test_csv_filename', type=str)
    parser.add_argument('--output_data_dict_filename', type=str)
    args = parser.parse_args()

    print('Merging dempgraphics, labs and vitals into single train and test sets...')
    # combine all the labs, demographics and vitals into a single train and single test csv
    features = ['demographics', 'vitals', 'labs']
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for feature in features:
        feature_train_df = pd.read_csv(os.path.join(args.train_test_split_dir, feature+'_train.csv'))
        feature_test_df = pd.read_csv(os.path.join(args.train_test_split_dir, feature+'_test.csv'))
        if train_df.empty:
            train_df = feature_train_df.copy()
            test_df = feature_test_df.copy()
        else:
            train_df = pd.merge(train_df, feature_train_df, on=['patient_id', 'hospital_admission_id', 'facility_code'])
            test_df = pd.merge(test_df, feature_test_df, on=['patient_id', 'hospital_admission_id', 'facility_code'])

   # combine all the labs, demographics and vitals jsons into a single json
    json_list = []
    for feature in features:
        feature_json = os.path.join(args.train_test_split_dir, feature+'_dict.json')
        with open(feature_json, "rb") as f:
            json_list.append(json.load(f))
    
    # merge the schemas of labs demographics and vitals
    final_dict = json_list[0].copy()
    final_dict['schema']['fields'] = json_list[0]['schema']['fields'] + json_list[1]['schema']['fields'] + json_list[2]['schema']['fields']


    # save to disk
    train_df.to_csv(args.train_csv_filename, index=False)
    test_df.to_csv(args.test_csv_filename, index=False)
    print('saved train file to %s'%args.train_csv_filename)
    print('saved test file to %s'%args.test_csv_filename)

    output_data_dict_filename = args.output_data_dict_filename
    with open(output_data_dict_filename, "w") as outfile:
        json.dump(final_dict, outfile)

    print('saved features dict to %s'%output_data_dict_filename)
