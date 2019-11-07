# Splits the cleaned EEG dataset into a time-series and a static file.
# For testing the data transformation workflow.

import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
ts = df[['subj_id', 'chunk_id', 'seq_num', 'eeg_signal']]
static = pd.pivot_table(df[['subj_id', 'seizure_binary_label', 
                            'category_label']],
                        index='subj_id')
static.to_csv('eeg_static.csv')
ts.to_csv('eeg_ts.csv', index=False)
