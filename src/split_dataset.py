# split_dataset.py

# Input:  --input: (required) a time-series file with one or more columns of
#              role 'id'
#         --data_dict: (required) data dictionary for that file
#         --test_size: (required) fractional size of the test set, expressed as
#              a number between 0 and 1
#         --output_dir: (required) directory where output files are saved
#         Additionally, a seed used for randomization is hard-coded.
# Output: train.csv and test.csv, where grouping is by all columns of role 'id'.

import argparse
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--data_dict', required=True)
parser.add_argument('--test_size', required=True, type=float)
parser.add_argument('--output_dir', required=True)
args = parser.parse_args()

SEED = 20190206

# Import data
df = pd.read_csv(args.input)
data_dict = json.load(open(args.data_dict))

# Split dataset
train_test = None
train = None
test = None
valid = None
gss1 = GroupShuffleSplit(n_splits=1, random_state=SEED, 
                         test_size=args.test_size)
id_cols = [c['name'] for c in data_dict['fields'] if c['role'] == 'id']
grp = df[id_cols]
grp = [' '.join(row) for row in grp.astype(str).values]
for a, b in gss1.split(df, groups=grp):
    train = df.iloc[a]
    test = df.iloc[b]

# Output data
train.to_csv(args.output_dir + '/train.csv', index=False)
test.to_csv(args.output_dir + '/test.csv', index=False)
