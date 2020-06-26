import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import GroupShuffleSplit
import copy

class Splitter:
    def __init__(self, n_splits=1, size=0, random_state=0, cols_to_group=None):
        self.n_splits = n_splits
        self.size = size
        self.cols_to_group = cols_to_group
        if hasattr(random_state, 'rand'):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(int(random_state))

    def make_groups_from_df(self, data_df):
        grp = data_df[self.cols_to_group]
        grp = [' '.join(row) for row in grp.astype(str).values]
        return grp

    def split(self, X, y=None, groups=None):
        gss1 = GroupShuffleSplit(random_state=copy.deepcopy(self.random_state), test_size=self.size, n_splits=self.n_splits)
        for tr_inds, te_inds in gss1.split(X, y=y, groups=groups):
            yield tr_inds, te_inds

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits

def split_dataframe_by_keys(data_df=None, size=0, random_state=0, cols_to_group=None):
    gss1 = Splitter(n_splits=1, size=size, random_state=random_state, cols_to_group=cols_to_group)
    for a, b in gss1.split(data_df, groups=gss1.make_groups_from_df(data_df)):
        train_df = data_df.iloc[a].copy()
        test_df = data_df.iloc[b].copy()
    return train_df, test_df

def parse_id_cols(data_dict):
    cols = []
    for col in data_dict['fields']:
        if 'role' in col and (col['role'] == 'id' or 
                              col['role'] == 'key'):
            cols.append(col['name'])
    return cols

def parse_output_cols(data_dict):
    cols = []
    for col in data_dict['fields']:
        if 'role' in col and (col['role'] == 'outcome' or 
                              col['role'] == 'output'):
            cols.append(col['name'])
    return cols

def parse_feature_cols(data_dict):
    non_time_cols = []
    for col in data_dict['fields']:
        if 'role' in col and col['role'] in ('feature', 'measurement', 'covariate'):
            non_time_cols.append(col['name'])
    return non_time_cols

def parse_time_col(data_dict):
    time_cols = []
    for col in data_dict['fields']:
        # TODO avoid hardcoding a column name
        if (col['name'] == 'hours' or col['role'].count('time')):
            time_cols.append(col['name'])
    return time_cols[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc_data_dir')
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--group_cols')
    parser.add_argument('--output_dir')

    args = parser.parse_args()
    
    df_vitals = pd.read_csv(os.path.join(args.preproc_data_dir, 'vitals_before_icu.csv' ))

    df_labs = pd.read_csv(os.path.join(args.preproc_data_dir, 'labs_before_icu.csv'))

    df_demographics = pd.read_csv(os.path.join(args.preproc_data_dir, 'demographics_before_icu.csv'))

    df_transfer_to_icu_outcomes = pd.read_csv(os.path.join(args.preproc_data_dir,'icu_transfer_outcomes.csv'))
   

    vitals_data_dict_json = os.path.join(args.preproc_data_dir, 'Spec-Vitals.json')

    with open(vitals_data_dict_json, 'r') as f:
        vitals_data_dict = json.load(f)
        try:
            vitals_data_dict['fields'] = vitals_data_dict['schema']['fields']
        except KeyError:
            pass

    id_cols = parse_id_cols(vitals_data_dict)
    vitals = parse_feature_cols(vitals_data_dict)

    # compute missingness per stay
    vital_counts_per_stay_df = df_vitals.groupby(id_cols).count()
    print('#######################################')
    print('reporting missingness over full stays..')
    for vital in vitals:
        print('----------------------------------------------------')
        print('%s has a missing rate of %.4f percent over all stays'%(vital, ((vital_counts_per_stay_df[vital]==0).sum())/vital_counts_per_stay_df.shape[0]))
        print('----------------------------------------------------')
    
    print('#######################################')
    print('Printing summary statistics for vitals')
    vitals_summary_df = pd.DataFrame()
    for vital in vitals:
        curr_vital_series = df_vitals[vital]
        vitals_summary_df.loc[vital, 'min'] = curr_vital_series.min()
        vitals_summary_df.loc[vital, 'max'] = curr_vital_series.max()
        vitals_summary_df.loc[vital, '5%'] = np.percentile(curr_vital_series[curr_vital_series.notnull()], 5)
        vitals_summary_df.loc[vital, '95%'] = np.percentile(curr_vital_series[curr_vital_series.notnull()], 95)
        vitals_summary_df.loc[vital, 'median'] = curr_vital_series.median()
    
    print(vitals_summary_df)

    print('#######################################')
    print('Getting train test split statistics')

    train_df, test_df = split_dataframe_by_keys(
        df_vitals, cols_to_group=args.group_cols, size=args.test_size, random_state=args.random_seed) 
    outcomes_train_df, outcomes_test_df = split_dataframe_by_keys(
        df_transfer_to_icu_outcomes, cols_to_group=args.group_cols, size=args.test_size, random_state=args.random_seed)
    print('----------------------------------------------------')
    print('training set has %s stays for %s patients and %s ICU transfers'%(len(train_df.hospital_admission_id.unique()), len(train_df.patient_id.unique()),
        outcomes_train_df.transfer_to_ICU_outcome.sum()))
    print('----------------------------------------------------')
    print('test set has %s stays for %s patients and %s ICU transfers'%(len(test_df.hospital_admission_id.unique()), len(test_df.patient_id.unique()),
        outcomes_test_df.transfer_to_ICU_outcome.sum()))
    print('----------------------------------------------------')

    print('Lengths of stay statistics :')
    training_set_stay_lengths = train_df.groupby(id_cols)['hours_since_admission'].apply(lambda x : max(x)-min(x)).values
    test_set_stay_lengths = test_df.groupby(id_cols)['hours_since_admission'].apply(lambda x : max(x)-min(x)).values
    training_set_stay_lengths_outcome_0 = training_set_stay_lengths[outcomes_train_df.transfer_to_ICU_outcome==0]
    test_set_stay_lengths_outcome_0 = test_set_stay_lengths[outcomes_test_df.transfer_to_ICU_outcome==0]
    training_set_stay_lengths_outcome_1 = training_set_stay_lengths[outcomes_train_df.transfer_to_ICU_outcome==1]
    test_set_stay_lengths_outcome_1 = test_set_stay_lengths[outcomes_test_df.transfer_to_ICU_outcome==1]
    print('----------------------------------------------------')
    print('training set lengths of stay outcome 0:  %.2f(%.2f - %.2f)'%(np.median(training_set_stay_lengths_outcome_0), np.percentile(training_set_stay_lengths_outcome_0, 5), np.percentile(training_set_stay_lengths_outcome_0, 95)))
    print('----------------------------------------------------')
    print('training set lengths of stay outcome 1:  %.2f(%.2f - %.2f)'%(np.median(training_set_stay_lengths_outcome_1), np.percentile(training_set_stay_lengths_outcome_1, 5), np.percentile(training_set_stay_lengths_outcome_1, 95)))
    print('----------------------------------------------------')
    print('test set lengths of stay outcome 0:  %.2f(%.2f - %.2f)'%(np.median(test_set_stay_lengths_outcome_0), np.percentile(test_set_stay_lengths_outcome_0, 5), np.percentile(test_set_stay_lengths_outcome_0, 95)))
    print('----------------------------------------------------')
    print('test set lengths of stay outcome 1:  %.2f(%.2f - %.2f)'%(np.median(test_set_stay_lengths_outcome_1), np.percentile(test_set_stay_lengths_outcome_1, 5), np.percentile(test_set_stay_lengths_outcome_1, 95)))
    


    from IPython import embed; embed()
