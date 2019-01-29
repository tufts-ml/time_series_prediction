# align_to_grid.py

# Input:  a time-series file with irregular timesteps given in the 'time'
#         column, specified as the first command line parameter, and one or more
#         columns ending in '_id'
# Output: aligned.csv, a time-series file with regular timesteps corresponding
#         to the second command line parameter, grouped by all columns ending
#         in '_id' and with values representing the mean for the given group
#         within the given time window. For timestep syntax see
#     http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

# TODO: summary stat other than mean?

import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])

df['time2'] = pd.to_datetime(df['time'])
df = df.drop('time', axis='columns')
df = df.rename({'time2': 'time'}, axis='columns')

group_cols = list(df.columns[df.columns.str.endswith('_id')])

aligned = df.groupby(group_cols).resample(sys.argv[2], on='time', label='right',
                                          closed='left').mean()
aligned = aligned.drop(group_cols, axis='columns')

aligned.to_csv('aligned.csv')