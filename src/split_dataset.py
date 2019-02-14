# split_dataset.py

# Input:  Argument 1: a time-series file with one or more columns of role 'id'
#         Argument 2: data dictionary for that file
#         Argument 3: fractional size of the validation set, expressed as a
#                     number between 0 and 1
#         Argument 4: fractional size of the test set
#         Additionally, a seed used for randomization is hard-coded.
# Output: train.csv, test.csv, and valid.csv, where grouping is by all columns
#         of role 'id'.

import argparse
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('data_dict')
parser.add_argument('valid_size', type=float)
parser.add_argument('test_size', type=float)
args = parser.parse_args()

SEED = 20190206

# Import data
df = pd.read_csv(args.input_file)
data_dict = json.load(open(args.data_dict))

# Split dataset
train_test = None
train = None
test = None
valid = None
gss1 = GroupShuffleSplit(n_splits=1, random_state=SEED, 
                         test_size=args.valid_size)
#grp = df[df.columns[df.columns.str.endswith('_id')]]
id_cols = [c['name'] for c in data_dict['fields'] if c['role'] == 'id']
grp = df[id_cols]
grp = [' '.join(row) for row in grp.astype(str).values]
for a, b in gss1.split(df, groups=grp):
    train_test = df.iloc[a]
    valid = df.iloc[b]
gss2 = GroupShuffleSplit(n_splits=1, random_state=SEED,
                         test_size=args.test_size/(1-args.valid_size))
grp = train_test[id_cols]
grp = [' '.join(row) for row in grp.astype(str).values]
for a, b in gss2.split(train_test, groups=grp):
    train = train_test.iloc[a]
    test = train_test.iloc[b]

# Output data
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
valid.to_csv('valid.csv', index=False)
