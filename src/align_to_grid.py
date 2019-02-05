# align_to_grid.py

# Input:  a time-series file, specified as the first command line parameter,
#         with one or more columns ending in '_id', and one of:
#           - a 'time' column containing irregular timesteps
#           - a 'seq_num' column containing the numbers 1, ..., n for each group
# Output: aligned.csv, a time-series file with regular steps grouped by all
#         columns ending in '_id'. The steps are determined by the second
#         command line parameter. If there is a
#           - 'time' column: the steps correspond to the syntax in
#             http://pandas.pydata.org/pandas-docs/stable
#             /timeseries.html#offset-aliases
#           - 'seq_num' column: the steps are every n entries in the sequence,
#             where n is the parameter
#         Values represent the mean for the given group within each step. Steps
#         are closed and labeled on the right: e.g., a step labeled 2:00:00
#         might cover data from 1:00:01 to 2:00:00.

# TODO: summary stat other than mean?

import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
group_cols = list(df.columns[df.columns.str.endswith('_id')])

if 'time' in df.columns:
    df['_time'] = pd.to_datetime(df['time'])
    df = df.drop('time', axis='columns')
    df = df.rename({'_time': 'time'}, axis='columns')
    aligned = df.groupby(group_cols).resample(sys.argv[2], on='time', label='right',
                                            closed='right').mean()
    aligned = aligned.drop(group_cols, axis='columns')
elif 'seq_num' in df.columns:
    length = int(sys.argv[2])
    df['seq_num'] = df['seq_num'].apply(lambda x: length*(1+int((x-1)/length)))
    aligned = df.groupby(group_cols + ['seq_num']).mean()
else:
    raise Exception('File must contain a "time" or "seq_num" column')

aligned.to_csv('aligned.csv')
