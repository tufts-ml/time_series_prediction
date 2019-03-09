import argparse
import numpy as np
import os
import pandas as pd
import glob

def sanitize_column_names(df, inplace=False):
    ''' Sanitize column names to be all lowercase and contain no spaces

    Examples
    --------
    >>> df = pd.DataFrame(np.eye(3), columns=['a name with spaces', 'some Capital LetTerS'])
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
    >>> md_dict['subj_id']
    3390
    '''
    _, basename = os.path.split(csv_path)
    basename, _ = os.path.splitext(basename)
    assert basename.endswith("_timeseries")
    subj_id, ep_id_str, _ = basename.split("_")
    md_dict = dict(subj_id=int(subj_id), episode_id=int(ep_id_str.replace('episode','')))
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
    parser.add_argument(
        '--output_metadata_per_subj_csv_path',
        default=None,
        help='Path to csv file for per-subject metadata (age, weight, etc)')
    args = parser.parse_args()
    locals().update(args.__dict__)

    ## Verify correct dataset path
    if dataset_path is None or not os.path.exists(dataset_path):
        raise ValueError("Bad path to raw data:\n" + dataset_path)

    ## Each CSV file has this header:
    ## Hours,Capillary refill rate,Diastolic blood pressure,Fraction inspired oxygen,Glascow coma scale eye opening,Glascow coma scale motor response,Glascow coma scale total,Glascow coma scale verbal response,Glucose,Heart Rate,Height,Mean blood pressure,Oxygen saturation,Respiratory rate,Systolic blood pressure,Temperature,Weight,pH

    cols_to_read = ['Hours','Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glucose','Heart Rate','Height','Mean blood pressure','Oxygen saturation','Respiratory rate','Systolic blood pressure','Temperature','Weight','pH']

    cols_to_write = ['subj_id', 'episode_id', 'hours', 'height', 'weight', 'heart_rate', 'temperature',
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
            md_dict = extract_metadata_from_csv_path(csv_file)
            md_dict['is_testset'] = split.count('test')
            metadata_dict_list.append(md_dict)

            ts_data_df = pd.read_csv(csv_file, usecols=cols_to_read)
            ts_data_df['subj_id'] = md_dict['subj_id']
            ts_data_df['episode_id'] = md_dict['episode_id']
            ts_data_df_list.append(ts_data_df)

    x_df = pd.concat(ts_data_df_list)
    x_df = sanitize_column_names(x_df, inplace=True)
    x_df.sort_values(['subj_id', 'episode_id', 'hours'], inplace=True)
 
    x_df.to_csv(
        output_vitals_ts_csv_path,
        columns=cols_to_write,
        index=False)
    print("Wrote vitals tidy time-series to CSV file:")
    print(output_vitals_ts_csv_path)

    subj_uval_set = set(x_df['subj_id'].unique())
    subj_episode_uval_set = set([str(s) + '_' + str(e)
        for (s,e) in zip(x_df['subj_id'], x_df['episode_id'])])

    all_md_list = list()
    for split in ['train', 'test']:
        metadata_df = pd.read_csv(os.path.join(dataset_path, split, 'listfile.csv'))
        for row_id in range(metadata_df.shape[0]):
            csv_path = metadata_df['stay'].values[row_id]
            md_dict = extract_metadata_from_csv_path(csv_path)
            md_dict['inhospital_mortality'] = metadata_df['y_true'].values[row_id]
            md_dict['is_testset'] = split.count('test')

            ## Decide if we keep this one or not
            key_str = '%s_%s' % (md_dict['subj_id'], md_dict['episode_id'])
            if key_str not in subj_episode_uval_set:
                continue
            all_md_list.append(md_dict)
    metadata_df = pd.DataFrame(all_md_list)
    metadata_df.sort_values(['subj_id', 'episode_id'], inplace=True)
    metadata_df.to_csv(
        output_metadata_per_seq_csv_path,
        columns=['subj_id', 'episode_id', 'inhospital_mortality', 'is_testset'],
        index=False)
    print("Wrote to CSV file:")
    print(output_metadata_per_seq_csv_path)

 
