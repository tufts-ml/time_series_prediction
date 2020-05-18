# normalized_features.py

# Input:  --input: (required) a time-series file with unnormalized feature
#             values
#         --data_dict: (required) data dictionary for that flie
#         --output: (required) output file path
# Output: A time-series file with normalized feature values. Only fields of 
#         role 'feature' and type 'number' (as specified in the
#         data dictionary) are normalized. Scaling is based on each column's
#         IQR for robustness to outliers.

import json
import pandas as pd
import argparse
from sklearn.preprocessing import robust_scale

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--data_dict', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Import data-dictionary
    df = pd.read_csv(args.input)
    with open(args.data_dict) as f:
        data_dict = json.load(f)

    # Normalize features
    try:
        fields = data_dict['fields']
    except KeyError:
        fields = data_dict['schema']['fields']
    for c in fields:
        col = c['name']
        if (c['role'] in ('feature', 'covariate', 'measurement')
                and c['type'] in ('number', 'numeric')
                and col in df.columns):
            df[col] = robust_scale(df[col])

    # Export data
    df.to_csv(args.output, index=False)
