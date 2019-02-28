# fill_missing_values.py

# Input: a dataframe that may contain missing values, strategy for 
#		 filling those values, meta data for that df's columns

# Output: a dataframe that fills in missing values 

import sys
import pandas as pd
import argparse
import json

EXEMPT_COLS = []

def main():
	parser = argparse.ArgumentParser(description="Script for filling in "
												 "missing values in a "
												 "dataset according to a "
												 "specified strategy.")

	parser.add_argument('--data', type=str, required=True, 
						help='Path to csv dataframe of readings')
	parser.add_argument('--static', type=str, required=False, 
						help='Path to csv dataframe of static values')
	parser.add_argument('--data_dict', type=str, required=True,
						help='JSON dictionary describing data schema')

	parser.add_argument('--output', type=str, required=False, default=None)

	parser.add_argument('--strategy', type=str, required=True,
						choices=['pop_mean', 'carry_forward', 
								 'similar_subject_mean', 'GRU_simple',
								 'GRU_complex', 'nulls', 'None'])
	parser.add_argument('--multiple_strategies', default=False, 
						help='Set to False to execute only one strategy')
	parser.add_argument('--second_strategy', type=str, required=False,
						default='carry_forward', 
						choices=['pop_mean', 'carry_forward', 
								 'similar_subject_mean', 'GRU_simple',
								 'GRU_complex', 'nulls', 'None'])
	parser.add_argument('--third_strategy', type=str, required=False,
						default='pop_mean', 
						choices=['pop_mean', 'carry_forward', 
								 'similar_subject_mean', 'GRU_simple',
								 'GRU_complex', 'nulls', 'None'])

	args = parser.parse_args()

	# TODO: reorganize this control flow to better accomodate the different
	# 		combinations of arguments needed. 
	ts_df = pd.read_csv(args.data)
	get_exempt_cols(args.data_dict)
	static_df = None

	if ts_df.isnull().values.any():
		ts_df = apply_strategy(ts_df, static_df, args.strategy)
	
	if args.multiple_strategies:	
		if ts_df.isnull().values.any():
			ts_df = apply_strategy(ts_df, static_df, args.second_strategy)
		if ts_df.isnull().values.any():
			ts_df = apply_strategy(ts_df, static_df, args.third_strategy)
		if ts_df.isnull().values.any():
			ts_df = apply_strategy(ts_df, static_df, 'nulls')

	# save data to file
	if args.output is None:
		file_name = args.data.split('/')[-1].split('.')[0]
		data_output = '{}_filled.csv'.format(file_name)
	elif args.output[-4:] == '.csv':
		data_output = args.output
	else:
		data_output = '{}.csv'.format(args.output)
	ts_df.to_csv(data_output, index=False)

def apply_strategy(ts_df, static_df, strategy): 
	if strategy == 'pop_mean':
		return pop_mean(ts_df)
	elif strategy == 'carry_forward':
		return carry_forward(ts_df)
	elif strategy == 'similar_subject_mean':
		try:
			static_df = pd.read_csv(args.static)
		except:
			print('Could not open static file')
		return similar_subject_mean(ts_df, static_df)
	elif strategy == 'GRU_simple':
		return GRU_simple(ts_df)
	elif strategy == 'GRU_complex':
		return GRU_complex(ts_df)
	elif strategy == 'nulls':
		return nulls(ts_df)
	else:
		return ts_df

def get_exempt_cols(data_dict_file):
	with open(data_dict_file, 'r') as f:
		data_dict = json.load(f)

	for col in data_dict['fields']:
		if 'role' in col and (col['role'] == 'id' or 
		    				  col['role'] == 'sequence' or
		    				  col['role'] == 'time'):
			EXEMPT_COLS.append(col['name'])

def pop_mean(ts_df):
	data_cols = [c for c in ts_df.columns.values if c not in EXEMPT_COLS]

	for c in data_cols: 
		ts_df[c].fillna(ts_df[c].mean(), inplace=True)

	return ts_df

def carry_forward(ts_df): 
	ts_df = ts_df.groupby(['subject_id']).apply(lambda x: x.fillna(method='pad'))
	return ts_df

# Currently does not work due to changes necessary for more common cases. Will
# return to fix soon. 
def similar_subject_mean(ts_df, static_df):
	return ts_df

	data_cols = [c for c in ts_df.columns.values if c not in EXEMPT_COLS]
	subject_groups = []

	# TODO: Restructure this so we can add any permutation of categories, 
	#		ideally from an exterior source, use pandas query() to do so
	for i in range(len(AGE_CUTOFFS) - 1):
		for g in GENDERS:
			subject_groups.append(
				list(static_df.loc[(static_df['age_days'] > AGE_CUTOFFS[i]) & 
								   (static_df['age_days'] <= AGE_CUTOFFS[i+1]) &
								   (static_df['gender'] == g)]['subject_id'].values))

	# Find excluded subjects because of missing attributes and add them as 
	# their own individual groups. 
	missing_subjects = list(set(static_df.subject_id.unique()) - 
						    set([s for g in subject_groups for s in g]))
	for subject in missing_subjects:
		subject_groups.append([subject])

	new_ts_df = pd.DataFrame(columns=ts_df.columns)
	for group in subject_groups:
		subs_df = ts_df.loc[ts_df['subject_id'].isin(group)].copy()
		for c in data_cols:
			subs_df[c].fillna(subs_df[c].mean(), inplace=True)
		new_ts_df = new_ts_df.append(subs_df)

	return new_ts_df.sort_values(by=['subject_id', 'time'])

def GRU_simple(ts_df):
	pass

def GRU_complex(ts_df):
	pass

def nulls(ts_df):
	ts_df.fillna(0, inplace=True)
	return ts_df

if __name__ == '__main__':
    main()