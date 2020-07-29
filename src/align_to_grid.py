# align_to_grid.py

# Input:  --input: (required) a time-series file. One or more columns must
#             have role 'id', and there must be exactly one of the following:
#               - a column of role 'time' containing (likely irregular)
#                 timesteps that can be parsed by pandas.to_datetime()
#               - a column of role 'sequence' containing the numbers 1, ..., n
#                 for each group, or time values that can be parsed as floats
#                 (e.g. fractional hours).
#         --data_dict: (required) data dictionary for that file
#         --step_size: (required) time-series step size; see below
#         --output: (required) output file path
# Output: a time-series file with regular steps grouped by all columns in the
#         data of role 'id' in the data dictionary. If there is a
#           - column of role 'time': the steps correspond to the syntax in
#             http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
#           - column of role 'sequence': the steps are every n entries in the
#             sequence, where n is the argument, or are every n time units if
#             the column values are interpreted as fractional time units.
#         Values represent the mean for the given group within each step. Steps
#         are closed and labeled on the right: e.g., a step labeled 2:00:00
#         might cover data from 1:00:01 to 2:00:00.

# TODO: summary stat other than mean?
import sys
import numpy as np
import pandas as pd
import argparse
import json

def diff_from_start(timestamp_df):
     tstart = timestamp_df.iloc[0,0]
     timestamp_df['hours'] = [(pd.to_datetime(i)-pd.to_datetime(tstart)).total_seconds()/3600 for i in timestamp_df.iloc[:,0]]
     return timestamp_df

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_ts_csv_path', required=True)
parser.add_argument('--data_dict', required=True)
parser.add_argument('--step_size', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

# Import data
df = pd.read_csv(args.input_ts_csv_path)
with open(args.data_dict, 'r') as f:
    data_dict = json.load(f)

if 'fields' not in data_dict.keys():
    data_dict = data_dict['schema']

id_cols = [c['name'] for c in data_dict['fields']
           if c['role'] == 'id' and c['name'] in df.columns]
time_cols = [c['name'] for c in data_dict['fields'] if c['role'] in ('time', 'timestamp','timestamp_relative')]
seq_cols = [c['name'] for c in data_dict['fields'] if c['role'] == 'sequence' ]
relative_time_cols = [c['name'] for c in data_dict['fields'] if c['role'] == 'timestamp_relative']

# Align data to grid
if len(id_cols) < 1:
    raise Exception('File must contain at least one id column')

if len(time_cols) == 1:
    time_col = time_cols[0]
    df['_time'] = pd.to_datetime(df[time_col])
    df = df.drop(time_col, axis='columns')
    df = df.rename({'_time': time_col}, axis='columns')
    grouped = df.groupby(id_cols)
    aligned = grouped.resample(args.step_size, on=time_col, label='right',
                               closed='right').mean()
    aligned = aligned.drop(id_cols+relative_time_cols, axis='columns')
    aligned.reset_index(inplace=True)
    aligned.loc[:,time_col] = aligned.reset_index()[id_cols + time_cols].groupby(id_cols)[time_cols].apply(diff_from_start)['hours']
    aligned = aligned.set_index(id_cols+time_cols)

elif len(time_cols)>1:
    time_col = time_cols[-1]
    aligned = df.drop(columns = time_cols[:-1])

    #TODO handle timestamps in datetime format throughout the pipeline
elif len(seq_cols) == 1:
    seq_col = seq_cols[0]
    length = float(args.step_size)
    df[seq_col] = df[seq_col].apply(lambda x: length*(np.floor(x/length)))
    aligned = df.groupby(id_cols + [seq_col]).mean()

# Export data
print('saving aligned data')
aligned.to_csv(args.output)
