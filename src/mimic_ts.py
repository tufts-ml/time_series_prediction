# ts.py
#
# Converts raw MIMIC time-series data to the standardized format
# Input: the MIMIC-III database in CSV format (pointed to by INPUT_PATH)
# Output: ts.csv

# TODO: cohort selection
# TODO: urine output

import pandas as pd

# TODO: change to actual (non-demo) data path
INPUT_PATH = 'datasets/mimic_demo/'
OUTPUT_PATH = 'datasets/mimic_demo_output/'

# Dictionary where keys are Carevue and Metavision names for desired chart
# variables and values are the column names for the output dataset
# (note there are two source patient monitoring systems)
CHART_VARS = {
    'Heart Rate': 'hr',
    'Respiratory Rate': 'rr',
    'Arterial Blood Pressure mean': 'bp', 'Arterial BP Mean': 'bp',
    'Temperature Fahrenheit': 'temp', 'Temperature F': 'temp',
    'O2 saturation pulseoxymetry': 'spo2', 'SpO2': 'spo2',
    'Inspired O2 Fraction': 'fio2', 'FiO2 Set': 'fio2'}
    
# Dictionary where keys are MIMIC names for desired lab variables and values
# are the column names for the output dataset
LAB_VARS = {
    'Urea Nitrogen': 'bun',
    'Creatinine': 'creatinine',
    'Glucose': 'glucose',
    'Bicarbonate': 'bicarbonate',
    'Hematocrit': 'hct',
    'Lactate': 'lactate',
    'Magnesium': 'magnesium',
    'Platelet Count': 'platelet',
    'Potassium': 'potassium',
    'Sodium': 'sodium',
    'White Blood Cells': 'wbc'}

chartevents = pd.read_csv(INPUT_PATH + 'CHARTEVENTS.csv')
labevents   = pd.read_csv(INPUT_PATH + 'LABEVENTS.csv')
d_items     = pd.read_csv(INPUT_PATH + 'D_ITEMS.csv')
d_labitems  = pd.read_csv(INPUT_PATH + 'D_LABITEMS.csv')

# Merge chart data, keeping only desired variables
chart_merge = chartevents.merge(d_items, how='inner', on='ITEMID')
assert len(chart_merge) == len(chartevents)
chart_merge.columns = chart_merge.columns.str.lower()
chart_merge['time'] = pd.to_datetime(chart_merge['charttime'])
chart_merge = chart_merge.loc[chart_merge['label'].isin(CHART_VARS),
                              ['subject_id', 'label', 'valuenum', 'time']]

# Standardize column names and value formats across Carevue and Metavision
chart_merge.loc[chart_merge['label'] == 'FiO2 Set', 'valuenum'] *= 100
chart_merge['label'] = chart_merge['label'].map(CHART_VARS)

# Same for lab data
lab_merge = labevents.merge(d_labitems, how='inner', on='ITEMID')
assert len(lab_merge) == len(labevents)
lab_merge.columns = lab_merge.columns.str.lower()
lab_merge['time'] = pd.to_datetime(lab_merge['charttime'])
lab_merge = lab_merge.loc[lab_merge['label'].isin(LAB_VARS),
                          ['subject_id', 'label', 'valuenum', 'time']]
lab_merge['label'] = lab_merge['label'].map(LAB_VARS)

# Reshape data to wide format

ts = chart_merge.append(lab_merge)
ts = ts.pivot_table(values='valuenum', columns='label',
                    index=['subject_id', 'time'])
ts.to_csv(OUTPUT_PATH + 'ts.csv')
