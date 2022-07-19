'''
Produces supervised time-series dataset for in-hospital mortality prediction task

Preconditions
-------------
EICU in-hospital mortality codes extracted on disk

Post-conditions
---------------
Will produce folder with 3 files:
* Time-varying features: features_per_tstep.csv
* Outcomes per sequence : outcomes_per_seq.csv

'''


import argparse
import numpy as np
import os
import pandas as pd
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_raw_path',
        default=None,
        help='Path to the top folder of mimic3benchmarks in-hospital mortality dataset')
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Path to preprocessed csv files for tidy time-series of ICU bedside sensors data, features_per_tstep.csv and outcomes.csv')
    args = parser.parse_args()
    
    
    
    raw_data_dir = args.dataset_raw_path
    raw_file_csv = os.path.join(raw_data_dir, 'eicu_extract.hdf')
    
    # Read the outcomes and static features
    data_df = pd.read_hdf(raw_file_csv, key='data_df')
    data_df = data_df.reset_index()
    
    # Read the raw vitals, labs, measurements data
    lvm_df = pd.read_hdf(raw_file_csv, key='labs_vitals_treatments')
    lvm_df = lvm_df.reset_index()
    
    # define id columns and time columns
    id_cols = ['subject_id', 'hadm_id', 'icustay_id']
    time_col = ['hours_in']
    
    # remove multiindex from eicu dataframe
    lvm_df.columns = id_cols + time_col + ['_'.join(col) for col in lvm_df.columns.values if ((col[0] not in id_cols)&(col[0] not in time_col))]

    meaurement_cols = []
    for i in lvm_df.columns:
        if (i not in id_cols)&(i!=time_col):
            meaurement_cols.append(i)
    
    
    # exclude some measurements (not sure what they mean, and they're hard to parse) 
    exclude_measurements_list = ['treatment_treatment_list', 'treatment_treatment_area', 
                             'treatment_specific_treatment', 'noninvasive_mean_mean',
                            'noninvasive_mean_count', 'noninvasive_mean_std']
    final_measurements_cols = []
    for col in meaurement_cols:
        if (('_mean' in col)&(col not in exclude_measurements_list)):
            final_measurements_cols.append(col)
    
    features_df = lvm_df[id_cols + time_col + final_measurements_cols]
    
    # add gender and age to the features
    features_df = pd.merge(features_df, data_df[id_cols + ['age', 'gender']], on=id_cols, how='left')  
    
    # parse gender as is_male or is_unknown
    features_df['gender_is_male']=(features_df['gender'].values==1)*1
    features_df['gender_is_unknown']=np.isnan(features_df['gender'].values)*1
    features_df.drop(columns={'gender'}, inplace=True)
    
    # get the outcomes dataframe
    outcomes_df = data_df[id_cols + ['mort_hosp', 'mort_icu']]
    
    # assume that patients with nan outcomes don't die
    outcomes_df = outcomes_df.fillna(0.0)
    
    # remove '_mean' for measurement names
    old_cols = features_df.columns
    new_cols = [col.replace('_mean', '') for col in old_cols] 
    rename_cols_dict = dict(zip(old_cols, new_cols))
    features_df.rename(columns=rename_cols_dict, inplace=True)
    
    
    '''
    # create spec-sheet
    features_specs_df = pd.DataFrame(columns=['ColumnName', 'Role', 'Type', 'Minimum', 
                                              'Maximum', 'Units', 'Description', 'Required'])
    features_specs_df.loc[:, 'ColumnName']=features_df.columns 
    
    outcome_specs_df = pd.DataFrame(columns=['ColumnName', 'Role', 'Type', 'Minimum', 
                                             'Maximum', 'Units', 'Description', 'Required']) 
    outcome_specs_df.loc[:, 'ColumnName']=outcomes_df.columns
    
    features_specs_df.to_csv('feature_specs.csv', index=False)
    
    outcome_specs_df.to_csv('outcome_specs.csv', index=False) 
    '''
    
    # save the files to csv
    features_csv = os.path.join(args.output_dir, 'features_per_tstep.csv')
    features_df.to_csv(features_csv, index=False)
    print("Wrote features per timestep to CSV file: %s"%features_csv)
    
    outcomes_csv = os.path.join(args.output_dir, 'outcomes_per_seq.csv')
    outcomes_df.to_csv(outcomes_csv, index=False)
    print("Wrote outcomes per sequence to CSV file: %s"%outcomes_csv)
    
    
    '''
    # plot some example sequences with y=0 and y=1
    
    #y=0 examples
    features_y0_df = features_df[features_df.subject_id==3][plot_features+['hours_in']] 
    f, axs = plt.subplots(1,1, figsize=(15,5)) 
    features_y0_df.plot(x='hours_in', y=plot_features, ax=axs, marker='.') 
    axs.set_ylim([0, 200]) 
    plt.legend(loc='upper right') 
    f.savefig('y0_subject_mimic_0.png')                                                                                               
    features_y0_df = features_df[features_df.subject_id==13][plot_features+['hours_in']] 
    f, axs = plt.subplots(1,1, figsize=(15,5)) 
    features_y0_df.plot(x='hours_in', y=plot_features, ax=axs, marker='.') 
    axs.set_ylim([0, 200]) 
    plt.legend(loc='upper right') 
    f.savefig('y0_subject_mimic_1.png')    
    
    # y=1 examples
    features_y1_df = features_df[features_df.subject_id==12][plot_features+['hours_in']] 
    f, axs = plt.subplots(1,1, figsize=(15,5)) 
    features_y1_df.plot(x='hours_in', y=plot_features, ax=axs, marker='.') 
    axs.set_ylim([0, 200]) 
    plt.legend(loc='upper right') 
    f.savefig('y1_subject_mimic_0.png') 
    
    features_y1_df = features_df[features_df.subject_id==31][plot_features+['hours_in']] 
    f, axs = plt.subplots(1,1, figsize=(15,5)) 
    features_y1_df.plot(x='hours_in', y=plot_features, ax=axs, marker='.') 
    axs.set_ylim([0, 200]) 
    plt.legend(loc='upper right') 
    f.savefig('y1_subject_mimic_0.png') 
    
    
    '''
    