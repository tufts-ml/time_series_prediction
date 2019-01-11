# ts.py
#
# Converts raw MIMIC time-series data to the standardized format
# Input: the MIMIC-III database in CSV format (pointed to by INPUT_PATH)
# Output: ts.csv

# TODO: cohort selection
# TODO: lab measurements and urine output

import pandas as pd

# TODO: change to actual (non-demo) data path
INPUT_PATH = 'datasets/mimic_demo/'
OUTPUT_PATH = 'datasets/mimic_demo_output/'

# Dictionary where keys are Carevue and Metavision names for desired
# measurements and values are our abbreviations
# (note there are two source patient monitoring systems)
MEASUREMENTS = {
    'Heart Rate': 'hr',
    'Respiratory Rate': 'rr',
    'Arterial Blood Pressure mean': 'bp', 'Arterial BP Mean': 'bp',
    'Temperature Fahrenheit': 'temp', 'Temperature F': 'temp',
    'O2 saturation pulseoxymetry': 'spo2', 'SpO2': 'spo2',
    'Inspired O2 Fraction': 'fio2', 'FiO2 Set': 'fio2'}


chartevents = pd.read_csv(INPUT_PATH + 'CHARTEVENTS.csv')
d_items     = pd.read_csv(INPUT_PATH + 'D_ITEMS.csv')

# Merge data

merged = chartevents.merge(d_items, how='inner', on='ITEMID')
assert len(merged) == len(chartevents)
merged.columns = merged.columns.str.lower()
merged['charttime_dt'] = pd.to_datetime(merged['charttime'])

# Keep only desired measurement types and needed columns

merged = merged.loc[merged['label'].isin(MEASUREMENTS),
                    ['subject_id', 'label', 'valuenum', 'charttime_dt']]

# Standardize column names and value formats across Carevue and Metavision
merged.loc[merged['label'] == 'FiO2 Set', 'valuenum'] *= 100
merged['label'] = merged['label'].map(MEASUREMENTS)

# Reshape data to wide format

ts = merged.pivot_table(values='valuenum', columns='label',
                          index=['subject_id', 'charttime_dt'])

ts.to_csv(OUTPUT_PATH + 'ts.csv')