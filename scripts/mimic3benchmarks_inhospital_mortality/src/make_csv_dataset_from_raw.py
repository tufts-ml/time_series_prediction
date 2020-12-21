'''
Produces supervised time-series dataset for in-hospital mortality prediction task

Preconditions
-------------
mimic3-benchmarks in-hospital mortality codes extracted on disk

Post-conditions
---------------
Will produce folder with 3 files:
* Time-varying features: data_per_tstamp.csv
* Per-sequence features: data_per_seq.csv

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
    

    # read the per hour mean aggregated labs and vitals
    highfreq_df = pd.read_hdf(os.path.join(args.dataset_raw_path, 'all_hourly_data.h5'), 'vitals_labs_mean')
    
    #remove multi-index in columns
    feature_cols = list(highfreq_df.columns.droplevel(1))
    id_time_cols = list(highfreq_df.index.names)
    
    labs_vitals_df = pd.DataFrame(highfreq_df.reset_index().values, columns=id_time_cols+feature_cols) 
    labs_vitals_df[id_time_cols] = labs_vitals_df[id_time_cols].astype(int)
    
    
    # get the static df
    static_df = pd.read_hdf(os.path.join(args.dataset_raw_path, 'all_hourly_data.h5'), 'patients') 
    id_cols = list(static_df.index.names)
    
    # keep only required columns
    static_features = ['gender', 'age']
    outcome_col = ['mort_hosp']
    static_cols_to_keep = id_cols + static_features
    outcome_cols_to_keep = id_cols + outcome_col
    static_df.reset_index(inplace=True)
    static_features_df = static_df[static_cols_to_keep].copy()
    outcomes_df = static_df[outcome_cols_to_keep]
    
    
    # parse gender
    is_gender_male = np.asarray(static_features_df.gender=='M')*1.0
    static_features_df['is_gender_male'] = is_gender_male
    static_features_df.drop(columns='gender', inplace=True) 
     
    # merge the static and high-frequency features 
    features_df = pd.merge(labs_vitals_df, static_features_df, on=id_cols, how='inner')  
    
    
    from IPython import embed; embed()
    '''
    # create spec-sheet
    features_specs_df = pd.DataFrame(columns=['ColumnName', 'Role', 'Type', 'Minimum', 
                                              'Maximum', 'Units', 'Description', 'Required'])
    features_specs_df.loc[:, 'ColumnName']=features_df.columns 
    
    outcome_specs_df = pd.DataFrame(columns=['ColumnName', 'Role', 'Type', 'Minimum', 
                                             'Maximum', 'Units', 'Description', 'Required']) 
    outcome_specs_df.loc[:, 'ColumnName']=outcomes_df.columns
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
    