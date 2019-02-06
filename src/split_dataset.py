# split_dataset.py

# Splits a dataset into training, validation, and test sets, grouping on all
# columns ending in '_id'.
# Input:  a file with one or more columns ending in '_id', specified as the
#         first command line parameter; the fractional size of the validation
#         and test sets as the second and third parameters.
# Output: train.csv, test.csv, and valid.csv, where grouping is by all columns
#         ending in '_id'.

# Note: the Harutyunyan et al. workflow also has a script to do this.

import sys
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

VALID_SIZE = float(sys.argv[2])
TEST_SIZE = float(sys.argv[3])
SEED = 20190206

df = pd.read_csv(sys.argv[1])
train_test = None
train = None
test = None
valid = None
gss1 = GroupShuffleSplit(n_splits=1, test_size=VALID_SIZE, random_state=SEED)
grp = df[df.columns[df.columns.str.endswith('_id')]]
grp = [' '.join(row) for row in grp.astype(str).values]
for a, b in gss1.split(df, groups=grp):
    train_test = df.iloc[a]
    valid = df.iloc[b]
gss2 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE/(1-VALID_SIZE),
                         random_state=SEED)
grp = train_test[train_test.columns[train_test.columns.str.endswith('_id')]]
grp = [' '.join(row) for row in grp.astype(str).values]
for a, b in gss2.split(train_test, groups=grp):
    train = train_test.iloc[a]
    test = train_test.iloc[b]

train.to_csv('train.csv')
test.to_csv('test.csv')
valid.to_csv('valid.csv')
