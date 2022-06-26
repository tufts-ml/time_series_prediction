'''
Produces supervised time-series dataset for in-hospital mortality prediction task

Preconditions
-------------
mimic-iv in-icu features extracted on disk

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
from scipy.interpolate import interp1d   
from progressbar import ProgressBar

def get_hours_from_adm(example_time): 
    days = example_time.split(' ')[0] 
    hrs, minutes, seconds = example_time.split('days ')[-1].split(':')  
    hours_from_admission = int(days)*24 + int(hrs) + int(minutes)/60 + float(seconds)/3600 
    return hours_from_admission 


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
    
    
    # get the raw data
    print('Loading raw data...')
    root_dir = '/cluster/tufts/hugheslab/prath01/datasets/MIMIC-IV/physionet.org/files/mimiciv/MIMIC-IV-Data-Pipeline'
    data_dir = os.path.join(root_dir, 'data', 'features')
    data_df = pd.read_csv(os.path.join(data_dir, 'preproc_chart.csv.gz'))
    
    
    # merge with the items file to get the labs and vitals names
    print('Getting chartevent names...')
    items_csv = os.path.join(root_dir, 'mimic-iv-2.0', 'icu', 'd_items.csv.gz')
    items_df = pd.read_csv(items_csv)
    data_df = pd.merge(items_df[['itemid', 'label']], data_df, on='itemid')
    
    print('Converting the chartevent times to hours from admission...')
    data_df['hours_from_admission'] = data_df['event_time_from_admit'].apply(get_hours_from_adm)
    
    # keep only some vitals and lab measurements
    keep_columns = ['Heart Rate', 'Respiratory Rate', 'O2 saturation pulseoxymetry',
       'Non Invasive Blood Pressure systolic',
       'Non Invasive Blood Pressure diastolic',
       'Non Invasive Blood Pressure mean', 
       #'Arterial Blood Pressure mean',
       #'Arterial Blood Pressure systolic',
       #'Arterial Blood Pressure diastolic', 
        'Temperature Fahrenheit',
       #'Central Venous Pressure', 'Tidal Volume (observed)',
       #'Mean Airway Pressure', 'Peak Insp. Pressure', 'O2 Flow',
       #'Respiratory Rate (Total)', 
       'Potassium (serum)',
       'Sodium (serum)', 'Chloride (serum)', 'Hematocrit (serum)',
       'Hemoglobin', 'Creatinine (serum)', 
       #'HCO3 (serum)', 
        'BUN',
       #'Anion gap', 
        'Glucose (serum)', 'Magnesium', 
        #'Inspired Gas Temp.',
       'Phosphorous', 'Platelet Count', 
        #'Calcium non-ionized', 
       'WBC', 
        #'Inspiratory Time', 
       'PH (Arterial)',
       #'Pulmonary Artery Pressure diastolic',
       #'Pulmonary Artery Pressure systolic', 'Arterial O2 pressure',
       #'Arterial CO2 Pressure', 'TCO2 (calc) Arterial',
       #'Pulmonary Artery Pressure mean', 'PTT', 
        'Prothrombin time',
       'Lactic Acid', 
        #'Ionized Calcium', 
        'Daily Weight', 
       #'Blood Flow (ml/min)',
       'Glucose (whole blood)', 
        'Potassium (whole blood)', 
        'Total Bilirubin', 
        'ALT', 'AST', 
        #'Albumin', 
       'Troponin-T', 
        'Fibrinogen']
    
    # keep only the vitals and labs of interest and keep only first 24 hours of data
    total_hrs = 24
    print('Keeping only columns with low missingness ratio and restricting data to first %s hours...'%total_hrs)
    keep_inds = (data_df['label'].isin(keep_columns))&(data_df['hours_from_admission']<=total_hrs)

    data_df = data_df.loc[keep_inds] 
    data_df = data_df.drop(columns={'event_time_from_admit', 'itemid'}) 
    
    # keep only measurements taken after admission
    keep_inds = data_df.hours_from_admission>=0  
    data_df = data_df.loc[keep_inds]
    
    id_cols = ['stay_id', 'label'] 
    keys_df = data_df[id_cols].copy()
    for col in id_cols:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    fps = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0).any(axis=1)), keys_df.shape[0]]) 
    
    nrows = len(fps)-1
    dt = 1 # hourly buckets 
    labels_list = [] 
    vals_list = [] 
    t_list = [] 
    stay_id_list = [] 
    pbar = ProgressBar()

    print('Transforming data into %s hour buckets'%dt)
    for ii in pbar(range(nrows)): 
        curr_t = data_df.iloc[fps[ii]:fps[ii+1]]['hours_from_admission'].values 
        curr_vals = data_df.iloc[fps[ii]:fps[ii+1]]['valuenum'].values 
        t_end = np.ceil(curr_t.max())
        if t_end==0:
            t_end=1e-5
        Tnew = np.arange(0, t_end, dt)  
        if len(curr_vals)==1: 
            Xnew = np.nan*np.ones_like(Tnew) 
            Xnew[-1] = curr_vals 
        else: 
            F = interp1d(curr_t,curr_vals,kind='previous',fill_value="extrapolate")   
            Xnew = F(Tnew) 
        labels_new = [data_df.iloc[fps[ii]]['label']]*len(Tnew) 
        stay_id_new = [data_df.iloc[fps[ii]]['stay_id']]*len(Tnew) 
        vals_list.append(Xnew) 
        t_list.append(Tnew) 
        labels_list.append(labels_new) 
        stay_id_list.append(stay_id_new) 

    new_data_df = pd.DataFrame({'stay_id' : np.hstack(stay_id_list), 'hours_from_admission':np.hstack(t_list), 'label':np.hstack(labels_list), 'value' : np.hstack(vals_list)})

    keep_inds = ~np.isinf(new_data_df['value'])
    new_data_df = new_data_df.loc[keep_inds]
    del data_df
    
    # transform the dataframe where we have a measurement for every time point
    unique_labs_vitals = new_data_df['label'].unique()
    
    
    for ii, lv in enumerate(unique_labs_vitals): 
        curr_df = new_data_df.loc[new_data_df.label==lv, ['stay_id', 'hours_from_admission', 'value']].rename(columns={'value' : lv}) 
        if ii==0: 
            final_df = curr_df.copy() 
        else: 
            final_df = pd.merge(final_df, curr_df, on=['stay_id', 'hours_from_admission'], how='outer') 
            
    
    final_df = final_df.sort_values(by=['stay_id', 'hours_from_admission']).reset_index(drop=True)  
    
    # get the outcomes
    print('Loading the patient stay outcomes from admissions file...')
    outcomes_df = pd.read_csv('/cluster/tufts/hugheslab/prath01/datasets/MIMIC-IV/physionet.org/files/mimiciv/MIMIC-IV-Data-Pipeline/data/cohort/cohort_icu_mortality.csv.gz')
    
    # calculate length of stay
    print('Calculating length of stay for all patients...')
    td = pd.to_datetime(outcomes_df['outtime']) - pd.to_datetime(outcomes_df['intime'])  
    outcomes_df['length_of_stay_in_hours'] = [ii.total_seconds()/3600 for ii in td] 
    
    # minor pre-processing on outcomes file
    outcomes_df['is_gender_male']=(outcomes_df['gender']=='M')*1
    outcomes_df.rename(columns={'label':'in_icu_mortality'}, inplace=True)
    final_df = pd.merge(final_df, outcomes_df[['hadm_id', 'subject_id', 'stay_id', 'Age', 'is_gender_male', 'in_icu_mortality', 'length_of_stay_in_hours']], on=['stay_id'])
    
    id_cols = ['subject_id', 'hadm_id', 'stay_id']
    outcome_cols = ['in_icu_mortality', 'length_of_stay_in_hours']
    time_col = ['hours_from_admission']
    feature_cols = [] 
    for col in final_df.columns: 
        if ((col not in id_cols)&(col not in outcome_cols)&(col not in time_col)): 
            feature_cols.append(col) 
    
    print('Creating tidy features_per_tstep and outcomes_per_seq tables')
    features_df = final_df[id_cols+time_col+feature_cols].copy()
    outcomes_df = final_df[id_cols+outcome_cols].copy()
    outcomes_df = outcomes_df.drop_duplicates(subset=id_cols)
    
    save_dir = '/cluster/tufts/hugheslab/datasets/MIMIC-IV/'
    features_csv = os.path.join(save_dir, 'features_per_tstep.csv.gz')
    outcomes_csv = os.path.join(save_dir, 'outcomes_per_seq.csv.gz')
    
    print('Saving features per timestep to :\n%s'%features_csv)
    print('Saving outcomes per sequence to :\n%s'%outcomes_csv)
    features_df.to_csv(features_csv, index=False, compression='gzip')
    outcomes_df.to_csv(outcomes_csv, index=False, compression='gzip')
    
    
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
    