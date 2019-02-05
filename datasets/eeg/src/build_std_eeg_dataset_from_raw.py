import argparse
import numpy as np
import pandas as pd
import scipy.stats

def standardize_eeg_data(input_csv_path='raw/data.csv'):
    ''' Read raw csv file and create tidy dataframe

    Returns
    -------
    tidy_df : Pandas DataFrame
        subj_id, chunk_id, seq_num, eeg_signal, seizure_label
    '''
    data = pd.read_csv(input_csv_path)
    data['subj_id'] = data['id'].str.extract(r'X\d*\.(.*)', expand=False)
    data['chunk_id'] = data['id'].str.extract(r'X(\d*)\..*',
                                              expand=False).astype(int)
    data['category_label'] = np.asarray(data['y'], dtype=np.int32)
    data['seizure_binary_label'] = np.asarray(data['y'] == 1, dtype=np.int32)
    data = data.drop(['id', 'y'], axis=1)

    ## Convert one-col-per-timestep to one-row-per-timestep
    tidy_df = pd.wide_to_long(data, stubnames='X', i=['subj_id', 'chunk_id'],
                              j='seq_num')
    tidy_df['eeg_signal'] = np.asarray(tidy_df['X'], dtype=np.float64)
    del tidy_df['X']

    ## Reorder columns and avoid weird multi-level indexing
    tidy_df = pd.DataFrame(tidy_df.to_records())
    tidy_df = tidy_df[['subj_id', 'chunk_id', 'seq_num', 'eeg_signal',
                       'seizure_binary_label', 'category_label']].copy()
    tidy_df.sort_values(['subj_id', 'chunk_id', 'seq_num'], inplace=True)
    tidy_df.reset_index(drop=True, inplace=True)
    return tidy_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_csv_path',
        default='eeg_signal_data.csv')
    args = parser.parse_args()
    locals().update(args.__dict__)

    tidy_df = standardize_eeg_data()

    # Write to csv file
    tidy_df.to_csv(output_csv_path, index=False, header=True)
