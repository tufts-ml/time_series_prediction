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

def to_fixed_width_str(uval, max_len=10, extra='...'):
    '''

    Examples
    --------
    >>> to_fixed_width_str('a')
    'a            '
    >>> to_fixed_width_str('supercalifragilistic')
    'supercalif...'
    '''
    s = str(uval)
    if len(s) <= max_len:
       return s + ' ' * ((max_len) - len(s) + len(extra))
    else:
       return s[:max_len] + extra



def summarize_columns(df):
    ''' Summarize for each column what dtype it is, how often its missing, etc.

    '''
    for col in df.columns:
        print("")
        print("%s" % col)
        frac_missing = np.mean(df[col].isna())
        if frac_missing < 0.001 and frac_missing > 0:
            print("frac_missing %.3e" % (frac_missing))
        else:
            print("frac_missing %.3f" % (frac_missing))
        present_df = df[col].dropna()
        dtype = type(present_df.values[0])
        is_numeric = np.issubdtype(dtype, np.number)
        ct_df = present_df.value_counts()
        uvals_U = ct_df.index.values
        counts_U = ct_df.values 
        if is_numeric and uvals_U.size > 20:
            min_val = np.abs(present_df.values.min())
            max_val = np.abs(present_df.values.max())
            if min_val < 0.0001 or max_val > 100000:
                fmt = '%6.2f-th percentile % 12.3g'
            else:
                fmt = '%6.2f-th percentile % 12.4f'
            for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
                print(fmt % (p, np.percentile(present_df.values, p, interpolation='lower')))
        else:
            sortids = np.argsort(-1 * counts_U)
            for rankid, ss in enumerate(sortids[:10]):
                print("rank %5d/%d  %-10s  appears %8d times" % (rankid, uvals_U.size, to_fixed_width_str(uvals_U[ss]), counts_U[ss]))
            sortids = sortids[10:]
            M = sortids.size
            L = np.minimum(10, M)
            for rankid, ss in enumerate(sortids[M-L:M]):
                print("rank %5d/%d  %-10s  appears %8d times" % (M - L + rankid, uvals_U.size, to_fixed_width_str(uvals_U[ss]), counts_U[ss]))

def sanitize_column_names(df, inplace=False):
    ''' Sanitize column names to be all lowercase and contain no spaces

    Examples
    --------
    >>> df = pd.DataFrame(np.eye(2), columns=['a name with spaces', 'some Capital LetTerS'])
    >>> san_df = sanitize_column_names(df)
    >>> san_df.columns[0]
    'a_name_with_spaces'
    >>> san_df.columns[1]
    'some_capital_letters'
    '''
    if not inplace:
        df = df.copy()
    df.columns = map(str.lower, df.columns)
    remove_spaces = lambda s: s.replace('  ', ' ').replace(' ', '_')
    df.columns = map(remove_spaces, df.columns)
    return df

def extract_metadata_from_csv_path(csv_path):
    ''' Extract metadata about sequence from filename
    
    Examples
    --------
    >>> csv_path = '/path/to/my/dataset/3390_episode1_timeseries.csv'
    >>> md_dict = extract_metadata_from_csv_path(csv_path)
    >>> md_dict['episode_id']
    1
    >>> md_dict['subject_id']
    3390
    '''
    _, basename = os.path.split(csv_path)
    basename, _ = os.path.splitext(basename)
    assert basename.endswith("_timeseries")
    subject_id, ep_id_str, _ = basename.split("_")
    md_dict = dict(subject_id=int(subject_id), episode_id=int(ep_id_str.replace('episode','')))
    return md_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        default=None,
        help='Path to the top folder of mimic3benchmarks in-hospital mortality dataset')
    parser.add_argument(
        '--data_per_subject_path',
        default=None,
        help='Path to the top folder of the mimic3benchmarks per-subject path')
    parser.add_argument(
        '--n_sequences_to_read_per_split',
        type=int,
        default=None,
        help='Total sequences to read from each of the train/ and test/ folders. If None, will read all.')
    parser.add_argument(
        '--output_vitals_ts_csv_path',
        default=None,
        help='Path to csv file for tidy time-series of ICU bedside sensors data')
    parser.add_argument(
        '--output_metadata_per_seq_csv_path',
        default=None,
        help='Path to csv file for per-sequence metadata (did die in hospital, train/test assignment in original data, etc)')
    args = parser.parse_args()
    locals().update(args.__dict__)

    ## Verify correct dataset path
    if dataset_path is None or not os.path.exists(dataset_path):
        raise ValueError("Bad path to raw data:\n" + dataset_path)

    ## Each stays CSV file has this header:
    ## SUBJECT_ID,HADM_ID,ICUSTAY_ID,LAST_CAREUNIT,DBSOURCE,INTIME,OUTTIME,LOS,ADMITTIME,DISCHTIME,DEATHTIME,ETHNICITY,DIAGNOSIS,GENDER,DOB,DOD,AGE,MORTALITY_INUNIT,MORTALITY,MORTALITY_INHOSPITAL
    stays_csv_cols_to_read = [
        'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'ETHNICITY', 'DIAGNOSIS', 'GENDER', 'AGE']
    all_stays_df = pd.read_csv(os.path.join(data_per_subject_path, 'all_stays.csv'), usecols=stays_csv_cols_to_read)
    all_stays_df = sanitize_column_names(all_stays_df)
    assert 'subject_id' in all_stays_df.columns # sanitized col names should be lowercase

    episode_csv_cols_to_read = [
        'Icustay', 'Height', 'Weight']
    
    ## todo include diagnosis? would that be something known at admission???
    metadata_csv_cols_to_write = [
        'subject_id', 'episode_id', 'hadm_id', 'icustay_id', 'age', 'height', 'weight',
        'ethnicity', 'gender', 'last_careunit', 'dbsource', 'is_testset', 'inhospital_mortality']

    ## Each timeseries CSV file has this header:
    ## Hours,Capillary refill rate,Diastolic blood pressure,Fraction inspired oxygen,Glascow coma scale eye opening,Glascow coma scale motor response,Glascow coma scale total,Glascow coma scale verbal response,Glucose,Heart Rate,Height,Mean blood pressure,Oxygen saturation,Respiratory rate,Systolic blood pressure,Temperature,Weight,pH

    ts_csv_cols_to_read = ['Hours','Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glucose','Heart Rate','Height','Mean blood pressure','Oxygen saturation','Respiratory rate','Systolic blood pressure','Temperature','Weight','pH']

    ts_csv_cols_to_write = ['subject_id', 'episode_id', 'hours', 'weight', 'heart_rate', 'temperature',
        'mean_blood_pressure', 'systolic_blood_pressure', 'diastolic_blood_pressure', 
        'respiratory_rate', 'fraction_inspired_oxygen', 'oxygen_saturation',
        'capillary_refill_rate', 'ph', 'glucose']

    ts_data_df_list = list()
    metadata_dict_list = list()
    ## Load the per-subject time series from csv files
    for split in ['train', 'test']:
        csv_files_per_subj = glob.glob(os.path.join(dataset_path, split, '*timeseries.csv'))
        csv_files_per_subj = [p for p in sorted(csv_files_per_subj)]
        if n_sequences_to_read_per_split is not None:
            csv_files_per_subj = csv_files_per_subj[:n_sequences_to_read_per_split]
        for csv_file in csv_files_per_subj:
            # Track the icustay_id for each sequence
            md_dict = extract_metadata_from_csv_path(csv_file)
            ep_csv_path = os.path.join(
                data_per_subject_path, split, str(md_dict['subject_id']),
                'episode%d.csv' % md_dict['episode_id'])
            ep_df = pd.read_csv(ep_csv_path, usecols=episode_csv_cols_to_read)
            assert ep_df.shape[0] == 1
            ep_df = sanitize_column_names(ep_df)
            ep_df['icustay_id'] = ep_df['icustay']
            del ep_df['icustay']
            for col in ep_df.columns:
                if col not in md_dict:
                    md_dict[col] = ep_df[col].values[0]
            metadata_dict_list.append(md_dict)

            ts_data_df = pd.read_csv(csv_file, usecols=ts_csv_cols_to_read)
            ts_data_df['subject_id'] = md_dict['subject_id']
            ts_data_df['episode_id'] = md_dict['episode_id']
            ts_data_df_list.append(ts_data_df)

    # Manage icustay as a dataframe
    icustay_df = pd.DataFrame(metadata_dict_list)

    x_df = pd.concat(ts_data_df_list)
    x_df = sanitize_column_names(x_df, inplace=True)
    x_df.sort_values(['subject_id', 'episode_id', 'hours'], inplace=True)
    x_df.to_csv(
        output_vitals_ts_csv_path,
        columns=ts_csv_cols_to_write,
        index=False)
    print("Wrote vitals tidy time-series to CSV file:")
    print(output_vitals_ts_csv_path)

    subj_uval_set = set(x_df['subject_id'].unique())
    subj_episode_uval_set = set([str(s) + '_' + str(e)
        for (s,e) in zip(x_df['subject_id'], x_df['episode_id'])])
    all_md_list = list()
    for split in ['train', 'test']:
        metadata_df = pd.read_csv(os.path.join(dataset_path, split, 'listfile.csv'))
        for row_id in range(metadata_df.shape[0]):
            csv_path = metadata_df['stay'].values[row_id]
            md_dict = extract_metadata_from_csv_path(csv_path)
            md_dict['inhospital_mortality'] = metadata_df['y_true'].values[row_id]
            md_dict['is_testset'] = split.count('test')

            ## Decide if we keep this one or not
            key_str = '%s_%s' % (md_dict['subject_id'], md_dict['episode_id'])
            if key_str not in subj_episode_uval_set:
                continue
            all_md_list.append(md_dict)
    metadata_df = pd.DataFrame(all_md_list)
    metadata_df.sort_values(['subject_id', 'episode_id'], inplace=True)
    metadata_df = pd.merge(
        metadata_df, icustay_df,
        left_on=['subject_id', 'episode_id'],
        right_on=['subject_id', 'episode_id'],
        how='left', validate='one_to_one').copy()
    assert 'icustay_id' in metadata_df
    ## Merge in the stays.csv info
    cols_to_add = list()
    for col in all_stays_df.columns:
        if col in metadata_df:
            continue
        col_values = all_stays_df[col].notnull()
        if col_values.shape[0] < 1:
            continue
        cols_to_add.append(col)
        dtype = type(col_values.values[0])
        is_numeric_col = np.issubdtype(dtype, np.number)
        if is_numeric_col:
            metadata_df[col] = np.nan
        else:
            metadata_df[col] = ''

    for row_id in range(metadata_df.shape[0]):
        subj_id = metadata_df['subject_id'].values[row_id]
        stay_id = metadata_df['icustay_id'].values[row_id]
        q_df = all_stays_df.query("subject_id == %d and icustay_id == %d" % (subj_id, stay_id))
        if q_df.shape[0] == 0:
            raise ValueError("Stay info not found: subj %d stay_id %d" % (subj_id, stay_id))
        assert q_df.shape[0] == 1
        for col in cols_to_add:
            metadata_df[col].values[row_id] = q_df[col].values[0]
    metadata_df['hadm_id'] = metadata_df['hadm_id'].astype(np.int32)
    metadata_df.to_csv(
        output_metadata_per_seq_csv_path,
        columns=metadata_csv_cols_to_write,
        index=False)
    print("Wrote to CSV file:")
    print(output_metadata_per_seq_csv_path)


    summarize_columns(metadata_df[metadata_csv_cols_to_write])

 

