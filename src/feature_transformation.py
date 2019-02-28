# feature_transformation.py

# Input: Requires a dataframe, specification of whether to collapse
#		 into summary statistics and which statistics, or whether to
# 		 add a transformation of a column and the operation with which
#		 to transform

#		 Requires a json data dictionary to accompany the dataframe,
#		 data dictionary columns must all have a defined "role" field

# Output: Puts transformed dataframe into ts_transformed.csv and 
#		  updated data dictionary into transformed.json

# Warning: With multiple ID columns and a large dataset, this runs
# 		   quite slowly

import sys
import pandas as pd
import argparse
import json
import numpy as np
from scipy import stats

def main(): 
	parser = argparse.ArgumentParser(description="Script for collapsing"
												 "time features or adding"
												 "new features.")

	parser.add_argument('--input', type=str, required=True, 
						help='Path to csv dataframe of readings')
	parser.add_argument('--data_dict', type=str, required=True,
						help='Path to json data dictionary file')
	parser.add_argument('--output', type=str, required=False, default=None)
	parser.add_argument('--data_dict_output', type=str, required=False, 
						default=None)
	
	parser.add_argument('--collapse', default=False, action='store_true')
	parser.add_argument('--collapse_features', type=str, required=False,
						default='mean median std_dev min_val max_val', 
						help="Enclose options with 's, choose"
							 "from mean, std_dev, min_val, max_val"
							 "median")
	
	# TODO: Add arithmetic opertions (ie column1 * column2 / column3)
	parser.add_argument('--add_feature', default=False, action='store_true')
	parser.add_argument('--add_from', type=str, required=False)
	parser.add_argument('--new_feature', type=str, required=False,
						default='z-score', 
						choices=['square', 'floor', 'int', 'sqrt', 
								 'abs', 'z-score', 'ceiling', 'float'])
	
	args = parser.parse_args()

	ts_df = pd.read_csv(args.input)
	data_dict = None
	
	# transform data
	if args.collapse:
		ts_df = collapse(ts_df, args)
		data_dict = update_data_dict_collapse(args)
	elif args.add_feature:
		ts_df = add_new_feature(ts_df, args)
		data_dict = update_data_dict_add_feature(args)

	# save data to file
	if args.output is None:
		file_name = args.input.split('/')[-1].split('.')[0]
		data_output = '{}_transformed.csv'.format(file_name)
	elif args.output[-4:] == '.csv':
		data_output = args.output
	else:
		data_output = '{}.csv'.format(args.output)
	ts_df.to_csv(data_output)
	print("Wrote to output CSV:\n%s" % (data_output))

	# save data dictionary to file
	if args.data_dict_output is None:
		file_name = args.data_dict.split('/')[-1].split('.')[0]
		dict_output = '{}_transformed.json'.format(file_name)
	elif args.data_dict_output[-5:] == '.json':
		dict_output = args.data_dict_output
	else:
		dict_output = '{}.json'.format(args.data_dict_output)
	with open(dict_output, 'w') as f:
		json.dump(data_dict, f)
	f.close()


# COLLAPSE TIME SERIES

# TODO: Possibly rearrange this to be simpler, ie put most of collapse func
#		in the recursive steps
def collapse(ts_df, args): 
	id_and_output_cols = parse_id_and_output_cols(args.data_dict)
	id_and_output_cols = remove_col_names_from_list_if_not_in_df(id_and_output_cols, ts_df)

	feature_cols = parse_feature_cols(args.data_dict)
	feature_cols = remove_col_names_from_list_if_not_in_df(feature_cols, ts_df)

	operations = []
	for op in args.collapse_features.split(' '):
		operations.append(get_summary_stat_func(op))

	new_df = pd.pivot_table(ts_df, values=feature_cols, index=id_and_output_cols, aggfunc=operations)

	# format columns in a clean fashion
	new_df.columns = ['_'.join(str(s).strip() for s in col if s) for col in new_df.columns]
	return new_df

def all_id_combinations(cols, df, combos, ids=[]):
	if len(cols) == 0:
		combos.append(ids)
		return

	for i in df[cols[0]].unique():
		ids_copy = list(ids)
		ids_copy.append(i)
		all_id_combinations(cols[1:], df.loc[df[cols[0]] == i], 
							combos, ids_copy)

def get_summary_stat_func(op):
	if op == 'mean':
		return np.mean
	elif op == 'std_dev':
		return np.std
	elif op == 'median':
		return np.median
	elif op == 'min_val':
		return np.amin
	elif op == 'max_val':
		return np.amax

# ADD NEW FEATURE COLUMN

def add_new_feature(ts_df, args):
	new_col_name = '{}_{}'.format(args.new_feature, args.add_from) 
	original_values = ts_df[args.add_from].tolist()
	new_values = None 

	if args.new_feature == 'z-score':
		new_values = stats.zscore(original_values)
	elif args.new_feature == 'square':
		new_values = np.square(original_values)
	elif args.new_feature == 'sqrt':
		new_values = np.sqrt(original_values)
	elif args.new_feature == 'floor':
		new_values = np.floor(original_values)
	elif args.new_feature == 'ceiling':
		new_values = np.ceil(original_values)
	elif args.new_feature == 'float':
		new_values = np.array(original_values).astype(float)
	elif args.new_feature == 'int':
		new_values = np.array(original_values).astype(int)
	elif args.new_feature == 'abs':
		new_values = np.absolute(original_values)
	
	ts_df[new_col_name] = new_values
	return ts_df


# DATA DICTIONARY STUFF: PARSING FUNCTIONS AND DICT UPDATING

def update_data_dict_collapse(args): 
	with open(args.data_dict, 'r') as f:
		data_dict = json.load(f)
	f.close()

	id_and_output_cols = parse_id_and_output_cols(args.data_dict)
	feature_cols = parse_feature_cols(args.data_dict)

	operations = []
	for op in args.collapse_features.split(' '):
		operations.append(get_summary_stat_func(op))

	new_fields = []
	for name in id_and_output_cols:
		for col in data_dict['fields']:
			if col['name'] == name: 
				new_fields.append(col)
	for name in feature_cols:
		for col in data_dict['fields']:
			if col['name'] == name: 
				for op in operations:
					new_dict = dict(col)
					new_dict['name'] = '{}_{}'.format(op, name)
					new_fields.append(new_dict)

	new_data_dict = dict()
	new_data_dict['fields'] = new_fields
	return new_data_dict

def update_data_dict_add_feature(args): 
	with open(args.data_dict, 'r') as f:
		data_dict = json.load(f)
	f.close()

	new_fields = []
	for col in data_dict['fields']:
		if 'name' in col and col['name'] == args.add_from:
			new_dict = dict(col)
			new_dict['name'] = '{}_{}'.format(args.new_feature, col['name'])
			new_fields.append(new_dict)
		else: 
			new_fields.append(col)

	new_data_dict = dict()
	new_data_dict['fields'] = new_fields
	return new_data_dict

def parse_id_and_output_cols(data_dict_file):	
	cols = []
	with open(data_dict_file, 'r') as f:
		data_dict = json.load(f)
	f.close()

	for col in data_dict['fields']:
		if 'role' in col and (col['role'] == 'id' or 
							  col['role'] == 'outcome' or 
							  col['role'] == 'other'):
			cols.append(col['name'])
	return cols

def parse_feature_cols(data_dict_file):
	non_time_cols = []
	with open(data_dict_file, 'r') as f:
		data_dict = json.load(f)
	f.close()

	for col in data_dict['fields']:
		if 'role' in col and col['role'] == 'feature':
			non_time_cols.append(col['name'])
	return non_time_cols

def remove_col_names_from_list_if_not_in_df(col_list, df):
	''' Remove column names from provided list if not in dataframe

	Examples
	--------
	>>> df = pd.DataFrame(np.eye(3), columns=['a', 'b', 'c'])
	>>> remove_col_names_from_list_if_not_in_df(['q', 'c', 'a', 'e', 'f'], df)
	['c', 'a']
	'''
	assert isinstance(col_list, list)
	for cc in range(len(col_list))[::-1]:
		col = col_list[cc]
		if col not in df.columns:
			col_list.remove(col)
	return col_list

if __name__ == '__main__':
    main()