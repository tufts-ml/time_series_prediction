# dynamic_feature_transformation.py

# Input: Requires a dataframe, specification of whether to collapse
#        into summary statistics and which statistics, or whether to
#        add a transformation of a column and the operation with which
#        to transform

#        Requires a json data dictionary to accompany the dataframe,
#        data dictionary columns must all have a defined "role" field

# Output: Puts transformed dataframe into ts_transformed.csv and 
#         updated data dictionary into transformed.json


import sys
import pandas as pd
import argparse
import json
import numpy as np
from scipy import stats
import ast
import time
import copy
import os

from progressbar import ProgressBar

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from feature_transformation import (get_fenceposts, parse_time_cols, parse_id_cols, parse_feature_cols, 
                                    update_data_dict_collapse, remove_col_names_from_list_if_not_in_df) 
from utils import load_data_dict_json
from featurize_single_time_series import featurize_ts

def main():
    parser = argparse.ArgumentParser(description="Script for collapsing"
                                                 "time features or adding"
                                                 "new features.")
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to csv dataframe of readings')
    parser.add_argument('--ts_data_dict', type=str, required=False,
                        help='Path to json data dictionary file')
    parser.add_argument('--outcomes', type=str, required=False, 
                        help='Path to csv dataframe of outcomes')
    parser.add_argument('--outcomes_data_dict', type=str, required=False,
                        help='Path to json data dictionary file for outcomes')
    parser.add_argument('--dynamic_collapsed_features_csv', type=str, required=False, default=None)
    parser.add_argument('--dynamic_collapsed_features_data_dict', type=str, required=False, default=None)
    parser.add_argument('--dynamic_outcomes_csv', type=str, required=False, default=None)    

    parser.add_argument('--features_to_summarize',
        type=str, required=False,
        default='slope std', 
        help="Enclose options with 's, choose "
                             "from mean, std, min, max, "
                             "median, slope, count, present, hours_since_measured")
    parser.add_argument('--percentile_ranges_to_summarize', type=str, required=False,
                        default='[(0, 10), (0, 25), (0, 50), (50, 100), (75, 100), (90, 100), (0, 100)]',
                        help="Enclose pairs list with 's and [], list all desired ranges in "
                             "parentheses like this: '[(0, 50), (25, 75), (50, 100)]'")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        is_fake = True
        ts_df, ts_data_dict, outcomes_df, outcomes_data_dict = make_fake_input_data(
            n_seqs=10, n_features=100, min_duration=24.0, max_duration=240.0)
    else:
        is_fake = False
        print('reading features...')
        ts_df = pd.read_csv(args.input)
        ts_data_dict = load_data_dict_json(args.ts_data_dict)
        print('done reading features...')
    
        print('reading outcomes...')
        outcomes_df = pd.read_csv(args.outcomes)
        outcomes_data_dict = load_data_dict_json(args.outcomes_data_dict)
        print('done reading outcomes...')    

    # transform data
    t1 = time.time()
    dynamic_collapsed_df, dynamic_outcomes_df = featurize_stack_of_many_time_series(
        ts_df=ts_df, ts_data_dict=ts_data_dict,                                           
        outcomes_df=outcomes_df, 
        outcomes_data_dict=outcomes_data_dict,
        summary_ops=args.features_to_summarize, 
        percentile_slices_to_featurize=args.percentile_ranges_to_summarize,
        outcome_col='mort_hosp'
        )
    t2 = time.time()
    print('done collapsing data..')
    print('time taken to collapse data : {} seconds'.format(t2-t1))

    if is_fake:
        sys.exit()

    # save data to file
    dynamic_collapsed_df.to_csv(args.dynamic_collapsed_features_csv, index=False, compression='gzip')
    print('Saved dynamic collapsed features to :\n%s'%args.dynamic_collapsed_features_csv)
    
    dynamic_outcomes_df.to_csv(args.dynamic_outcomes_csv, index=False, compression='gzip')
    print('Saved dynamic outcomes to :\n%s'%args.dynamic_outcomes_csv)
    
    # save data dictionary to file
    dynamic_collapsed_features_data_dict = update_data_dict_collapse(
        ts_data_dict, args.features_to_summarize, args.percentile_ranges_to_summarize)
    with open(args.dynamic_collapsed_features_data_dict, 'w') as f:
        json.dump(dynamic_collapsed_features_data_dict, f, indent=4)
    print('Saved dynamic collapsed features dict to :\n%s' % (
        args.dynamic_collapsed_features_data_dict))


def featurize_stack_of_many_time_series(
        ts_df=None,
        ts_data_dict=None,
        outcomes_df=None,
        outcomes_data_dict=None,
        summary_ops=['mean', 'min', 'max'],
        percentile_slices_to_featurize=[(0,100)],
        outcome_col='mort_hosp',
        outcome_seq_duration_col='stay_length',
        start_time_of_each_sequence=-24.0,
        max_time_of_each_sequence=504,
        start_time_of_endpoints=-12.0,
        time_between_endpoints=12,
        prediction_horizon=24,
        verbose=True,
        ):
    ''' Featurize many patient stays slices and extract outcome for each slice
    Args
    ----
    ts_df : pandas DataFrame
        Each row provides all measurements at one time of a single patient-stay
        Must contain one column already converted to numerical time
    ts_data_dict : dict
        Provides specification for every column of ts_df
    outcomes_df : pandas DataFrame
        Each row provides outcome of a single patient stay
    outcomes_data_dict : dict
        Provides specification for each column of outcomes_df
    summary_ops : list of strings
        Identifies the summary functions we wish to apply to each variable's ts
    percentile_slices_to_featurize : list of tuples
        Indicates percentile range of all subwindows we will featurize
        Example: [(0, 100), (0, 50)])
    Returns
    -------
    all_feat_df : DataFrame
        One row per featurized window of any patient-stay slice
        Key columns: ids + ['start', 'stop']
        Value columns: one per extracted feature
    all_outcomes_df : DataFrame
        One row per featurized window of any patient-stay slice
        Key columns: ids + ['start', 'stop']
        Value columns: just one, the outcome column
    Examples
    --------
    >>> args = make_fake_input_data(n_seqs=25, n_features=10, max_duration=50.0)
    >>> feat_df, outcome_df = featurize_stack_of_many_time_series(*args,
    ...     summary_ops=['mean', 'slope'],
    ...     start_time_of_each_sequence=0,
    ...     start_time_of_endpoints=0.0,
    ...     time_between_endpoints=12.0,
    ...     verbose=False,
    ...     );
    >>> feat_df.shape
    (95, 24)
    '''
    # Parse desired slices to featurize at each window
    # This allows command-line specification of ranges as a string
    if isinstance(percentile_slices_to_featurize, str):
        percentile_slices_to_featurize = ast.literal_eval(
            percentile_slices_to_featurize)
    if isinstance(summary_ops, str):
        summary_ops = summary_ops.split(' ')

    # Parse provided data dictionary
    # Recover specific columns for each of the different roles:
    id_cols = parse_id_cols(ts_data_dict)
    id_cols = remove_col_names_from_list_if_not_in_df(id_cols, ts_df)
    feature_cols = parse_feature_cols(ts_data_dict)
    feature_cols = remove_col_names_from_list_if_not_in_df(feature_cols, ts_df)
    time_cols = parse_time_cols(ts_data_dict)
    time_cols = remove_col_names_from_list_if_not_in_df(time_cols, ts_df)
    if len(time_cols) == 0:
        raise ValueError("Expected at least one variable with role='time'")
    elif len(time_cols) > 1:
        print("More than one time variable found. Choosing %s" % time_cols[-1])
    time_col = time_cols[-1]

    # Obtain fenceposts delineating each individual sequence within big stack
    # We assume that sequences changeover when *any* key differs
    # We convert all keys to a numerical datatype to make this possible
    keys_df = ts_df[id_cols].copy()
    for col in id_cols:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    middle_fence_posts = 1 + np.flatnonzero(
        np.diff(keys_df.values, axis=0).any(axis=1))
    fp = np.hstack([0, middle_fence_posts, keys_df.shape[0]])
    
    
    feat_arr_per_seq = list()
    windows_per_seq = list()
    outcomes_per_seq = list()
    durations_per_seq = list()
    missingness_density_per_seq = list()
    ids_per_seq = list()
    
    # Total number of features we'll compute in each feature vector
    F = len(percentile_slices_to_featurize) * len(feature_cols) * len(summary_ops)

    # Loop over each sequence in the tall tidy-format dataset
    start_time_sec = time.time()
    n_seqs = len(fp) - 1
    pbar = ProgressBar()
    for p in pbar(range(n_seqs)):
        
        # Get features and times for the current fencepost
        fp_start = fp[p]
        fp_end = fp[p+1]

        # Get the current stay keys
        cur_id_df = ts_df[id_cols].iloc[fp_start:fp_end].drop_duplicates(
            subset=id_cols)

        if outcomes_df is not None:
            # Get the total duration of the current sequence
            cur_outcomes_df = pd.merge(
                outcomes_df, cur_id_df, on=id_cols, how='inner')

            # Get the current sequence's finale outcome
            cur_final_outcome = int(cur_outcomes_df[outcome_col].values[0])
            try:
                cur_seq_duration = float(cur_outcomes_df[outcome_seq_duration_col].values[0])
            except KeyError:
                cur_seq_duration = float(ts_df[time_col].values[fp_end-1])
        else:
            cur_seq_duration = float(ts_df[time_col].iloc[fp_start:fp_end].values[-1])

        # Create windows at desired spacing
        stop_time_of_cur_sequence = min(
            cur_seq_duration, max_time_of_each_sequence)
        window_ends = np.arange(
            start_time_of_endpoints,
            stop_time_of_cur_sequence + 0.01 * time_between_endpoints,
            time_between_endpoints)
        
        # Create a dictionary of times and values for each feature
        time_arr_by_var = dict()
        val_arr_by_var = dict()
        times_U = ts_df[time_col].values[fp_start:fp_end]
        for feature_col in feature_cols:
            vals_U = ts_df[feature_col].values[fp_start:fp_end]
            keep_mask_U = np.isfinite(vals_U)
            if np.sum(keep_mask_U) > 0:
                time_arr_by_var[feature_col] = times_U[keep_mask_U].astype(np.float32)
                val_arr_by_var[feature_col] = vals_U[keep_mask_U]

        cur_seq_missing_density = (
            1.0 - len(val_arr_by_var.keys()) / float(len(feature_cols)))

        W = len(window_ends)
        window_features_WF = np.zeros([W, F], dtype=np.float32)
        window_starts_stops_W2 = np.zeros([W, 2], dtype=np.float32)
        if outcomes_df is not None:
            window_outcomes_W1 = np.zeros([W, 1], dtype=np.int64)
        
        for ww, window_end in enumerate(window_ends):
            window_starts_stops_W2[ww, 0] = start_time_of_each_sequence
            window_starts_stops_W2[ww, 1] = window_end
            
            window_features_WF[ww, :], feat_names = featurize_ts(
                time_arr_by_var, val_arr_by_var, 
                var_cols=feature_cols,
                var_spec_dict=ts_data_dict,
                start_numerictime=start_time_of_each_sequence,
                stop_numerictime=window_end,
                summary_ops=summary_ops,
                percentile_slices_to_featurize=percentile_slices_to_featurize)
            
            if outcomes_df is not None:
                # Determine the outcome for this window
                # Set outcome as final outcome if within the provided horizon
                # Otherwise, set to zero
                if window_end >= cur_seq_duration - prediction_horizon:
                    window_outcomes_W1[ww] = cur_final_outcome
                else:
                    window_outcomes_W1[ww] = 0
        
        # Append all windows from this sequence to the big lists
        feat_arr_per_seq.append(window_features_WF)
        windows_per_seq.append(window_starts_stops_W2)
        ids_per_seq.append(np.tile(cur_id_df.values[0], (W, 1)))

        durations_per_seq.append(np.tile(cur_seq_duration, (W, 1)))
        missingness_density_per_seq.append(cur_seq_missing_density)
        if outcomes_df is not None:
            outcomes_per_seq.append(window_outcomes_W1)
    
    # Produce final data frames
    features_df = pd.DataFrame(np.vstack(feat_arr_per_seq), columns=feat_names)    
    ids_df = pd.DataFrame(np.vstack(ids_per_seq), columns=id_cols) 
    windows_df = pd.DataFrame(np.vstack(windows_per_seq), columns=['start', 'stop'])
    all_features_df = pd.concat([ids_df, windows_df, features_df], axis=1)
    
    if outcomes_df is not None:
        durations_df = pd.DataFrame(np.vstack(durations_per_seq), columns=[outcome_seq_duration_col])
        outcomes_df = pd.DataFrame(np.vstack(outcomes_per_seq), columns=[outcome_col])
        all_outcomes_df = pd.concat([ids_df, windows_df, durations_df, outcomes_df], axis=1)
    else:
        durations_df = pd.DataFrame(np.vstack(durations_per_seq), columns=[outcome_seq_duration_col])
        all_outcomes_df = pd.concat([ids_df, windows_df, durations_df], axis=1)

    seq_lengths = np.vstack([a[0] for a in durations_per_seq])
    elapsed_time_sec = time.time() - start_time_sec
    
    if verbose:
        print('-----------------------------------------')
        print('Processed %d sequences of duration %.1f-%.1f in %.1f sec' % (
            n_seqs,
            np.percentile(seq_lengths, 5),
            np.percentile(seq_lengths, 95),
            elapsed_time_sec,
            ))
        print('    Total number of measured features: %d' % len(feature_cols))
        print('    Fraction of possible features NEVER seen in a seq. : %.2f-%.2f ' % (
            np.percentile(missingness_density_per_seq, 5),
            np.percentile(missingness_density_per_seq, 95),
            ))
        print('-----------------------------------------')   
    
    return all_features_df, all_outcomes_df


def make_fake_input_data(
        n_seqs=10, n_features=10,
        min_duration=24,
        max_duration=24*10,
        random_state=42,
        proba_deterioration=0.1):
    ''' Create example input data for debugging
    Returns
    -------
    ts_df
    ts_data_dict
    outcomes_df
    outcomes_data_dict
    '''

    signal_var = 'BODY_TEMPERATURE'
    distractor_vars = 'HEIGHT;WEIGHT;BMI;RESPIRATORY_RATE'.split(';')
    all_other_vars = ['IRRELEVANT_LAB_%03d' % d for d in range(n_features-5)]
    prng = np.random.RandomState(random_state)

    ts_df_per_seq = list()
    outcome_df_per_seq = list()

    for seq_id in range(n_seqs):
        patient_id = prng.choice(np.arange(100, 999))
        admission_id = 100000 * patient_id + prng.choice(np.arange(100, 999))

        start_numerictime = 0.0
        stop_numerictime = prng.uniform(min_duration, max_duration)
        v_by_var = dict()
        t_by_var = dict(__START_TIME__=start_numerictime, __PREDICTION_TIME__=stop_numerictime)

        n_signals_observed = prng.choice(
            np.arange(2, int(np.ceil(stop_numerictime))//5))
        t_L = np.sort(np.asarray([
            prng.uniform(start_numerictime, stop_numerictime)
            for _ in range(n_signals_observed)]))
        t_by_var[signal_var] = t_L

        # Determine the outcome (which is tied to the signal var)
        if seq_id == 0:
            did_deteriorate = 1 # always make sure at least one deteriorates
        elif seq_id == 1:
            did_deteriorate = 0
        else:
            did_deteriorate = int(prng.rand() < proba_deterioration)
        if did_deteriorate:
            # Temperature is abnormal, taking a nosedive from fever to too cold
            stdt_L = (t_L - start_numerictime) / (stop_numerictime - start_numerictime)
            tempF_L = (
                100.0 - 2.0 * stdt_L + 0.05 * prng.randn(t_L.size))
        else:
            # Temperature is normal, around 98.6
            tempF_L = 98.6 + 0.3 * prng.randn(t_L.size)
        v_by_var[signal_var] = (tempF_L - 32) * 5.0/9.0

        # Fill in many other variables
        for var_name in sorted(distractor_vars):
            n_distractors_observed = prng.choice(np.arange(5, int(np.ceil(stop_numerictime))/3))
            t_L = np.sort(np.asarray([
                prng.uniform(start_numerictime, stop_numerictime)
                for _ in range(n_signals_observed)]))
            t_by_var[var_name] = t_L
            v_by_var[var_name] = prng.randn(t_L.size)

        for var_name in sorted(all_other_vars):
            if var_name in t_by_var:
                continue
            do_observe_rare = int(prng.rand() < 0.05)
            if do_observe_rare > 0:
                t_L = np.sort(np.asarray([
                    prng.uniform(start_numerictime, stop_numerictime)
                    for _ in range(1)]))
                t_by_var[var_name] = t_L
                v_by_var[var_name] = prng.randn(t_L.size)

        # Convert to ts_df format
        ts_T = np.unique(np.hstack([arr for arr in t_by_var.values()]))
        T = ts_T.size
        all_vars = np.sort(np.hstack([signal_var] + distractor_vars + all_other_vars))
        V = len(all_vars)
        vals_TV = np.nan * np.ones((T, V), dtype=np.float32)
        for vv, var in enumerate(all_vars):
            if var not in t_by_var:
                continue
            insert_ids = np.searchsorted(ts_T, t_by_var[var])
            vals_TV[insert_ids, vv] = v_by_var[var]
        ts_df = pd.DataFrame(vals_TV, columns=all_vars)
        ts_df.insert(0, 'time', ts_T)
        ts_df.insert(0, 'admission_id', admission_id)
        ts_df.insert(0, 'patient_id', patient_id)

        outcome_df = pd.DataFrame([dict(
            patient_id=patient_id,
            admission_id=admission_id,
            clinical_deterioration_outcome=did_deteriorate,
            stay_length=stop_numerictime)])

        ts_df_per_seq.append(ts_df)
        outcome_df_per_seq.append(outcome_df)

    tall_ts_df = pd.concat(ts_df_per_seq, axis=0)
    tall_outcome_df = pd.concat(outcome_df_per_seq, axis=0)

    def get_role(col_name):
        if col_name.endswith('id'):
            return 'id'
        elif col_name.endswith('time'):
            return 'time'
        else:
            return 'measurement'

    # Make data dicts
    field_list = list()
    for col in ts_df.columns:
        info_dict = dict(
            name=col,
            role=get_role(col),
        )
        field_list.append(info_dict)
    ts_data_dict = {'fields':field_list}
    outcomes_data_dict = {'fields':[]}

    return tall_ts_df, ts_data_dict, tall_outcome_df, outcomes_data_dict

def update_data_dict_collapse(data_dict, collapse_range_features, range_pairs): 

    id_cols = parse_id_cols(data_dict)
    feature_cols = parse_feature_cols(data_dict)

    new_fields = []
    for name in id_cols:
        for col in data_dict['fields']:
            if col['name'] == name: 
                new_fields.append(col)
                
    for op in collapse_range_features.split(' '):
        for low, high in ast.literal_eval(range_pairs):
            for name in feature_cols:
                for col in data_dict['fields']:
                    if col['name'] == name: 
                        new_dict = dict(col)
                        new_dict['name'] = '{}_{}_{}-{}'.format(name, op, low, high)
                        new_fields.append(new_dict)

    new_data_dict = copy.deepcopy(data_dict)
    if 'schema' in new_data_dict:
        new_data_dict['schema']['fields'] = new_fields
        del new_data_dict['fields']
    else:
        new_data_dict['fields'] = new_fields
        
    return new_data_dict

if __name__ == '__main__':
    main()