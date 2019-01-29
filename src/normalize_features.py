# normalized_features.py

# Input:  a file with unnormalized feature values and a data dictionary for
#         that file, specified as command line parameters
# Output: normalized.csv, with normalized feature values. Booleans (as specified
#         by the data dictionary) and columns ending in '_id' are left alone.
#         Scaling is based on each column's IQR for robustness to outliers.

import sys
import json
import pandas as pd
from sklearn.preprocessing import robust_scale

df = pd.read_csv(sys.argv[1])
data_dict = json.load(open(sys.argv[2]))

for x in data_dict['fields']:
    col = x['name']
    if col[-3:] != '_id' and x['type'] != 'boolean':
        df[col] = robust_scale(df[col])

df.to_csv('normalized.csv', index=False)