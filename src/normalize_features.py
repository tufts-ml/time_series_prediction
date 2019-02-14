# normalized_features.py

# Input:  Argument 1: a time-series file with unnormalized feature values
#         Argument 2: data dictionary for that flie
#         Argument 3: output file path
# Output: A time-series file with normalized feature values. Only fields of 
#         role 'feature' and type 'integer' or 'number' (as specified in the
#         data dictionary) are normalized. Scaling is based on each column's
#         IQR for robustness to outliers.

import json
import pandas as pd
import argparse
from sklearn.preprocessing import robust_scale

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('data_dict')
parser.add_argument('output_file')
args = parser.parse_args()

# Import data
df = pd.read_csv(args.input_file)
data_dict = json.load(open(args.data_dict))

# Normalize features
for c in data_dict['fields']:
    col = c['name']
    if (c['role'] == 'feature' and c['type'] in ('integer', 'number')):
        df[col] = robust_scale(df[col])

# Export data
df.to_csv(args.output_file, index=False)
