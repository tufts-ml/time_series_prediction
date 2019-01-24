# mimic_static.py

# NOTE: will not be used

# Converts raw MIMIC static data to the standardized format
# Input: the MIMIC-III database in CSV format (pointed to by INPUT_PATH)
# Output: static.csv

# TODO: cohort selection

import pandas as pd

# TODO: change to actual (non-demo) data path
INPUT_PATH = 'datasets/mimic_demo/'
OUTPUT_PATH = 'datasets/mimic_demo_output/'

icustays          = pd.read_csv(INPUT_PATH + 'ICUSTAYS.csv')
admissions        = pd.read_csv(INPUT_PATH + 'ADMISSIONS.csv')
patients          = pd.read_csv(INPUT_PATH + 'PATIENTS.csv')
# services        = pd.read_csv(INPUT_PATH + 'SERVICES.csv')
# diagnoses_icd   = pd.read_csv(INPUT_PATH + 'DIAGNOSES_ICD.csv')
# d_icd_diagnoses = pd.read_csv(INPUT_PATH + 'D_ICD_DIAGNOSES.csv')
# drgcodes        = pd.read_csv(INPUT_PATH + 'DRGCODES.csv')

# Merge all data on the ICU stay level

merged = icustays.merge(admissions, how='inner', on=['HADM_ID', 'SUBJECT_ID'])
merged = merged.merge(patients, how='inner', on='SUBJECT_ID')
assert len(merged) == len(icustays)
merged.columns = merged.columns.str.lower()

# Restrict to one row per patient representing the earliest ICU stay

merged['intime_dt'] = pd.to_datetime(merged['intime'])
merged['dob_dt'] = pd.to_datetime(merged['dob'])
merged = merged.iloc[merged.groupby('subject_id')['intime_dt'].idxmin()]
assert len(merged) == len(patients)

# Calculate age, converting censored ages to missing in accordance with
# https://mimic.physionet.org/mimictables/patients/:
#   "Patients who are older than 89 years old at any time in the database have
#   had their date of birth shifted to obscure their age and comply with HIPAA.
#   The shift process was as follows: the patient’s age at their first admission
#   was determined. The date of birth was then set to exactly 300 years before
#   their first admission."

merged['age_dt'] = merged['intime_dt'] - merged['dob_dt']
merged.loc[merged['age_dt'].apply(lambda x: not (0 < x.days < 365*200)),
           'age_dt'] = pd.NaT
merged['age_days'] = merged['age_dt'].apply(lambda x: x.days)

# Keep needed variables and export (where are height and weight?)

static = merged[['subject_id', 'age_days', 'gender']]
static.to_csv(OUTPUT_PATH + 'static.csv', index=False)


