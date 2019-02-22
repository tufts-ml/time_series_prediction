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

	parser.add_argument('--data', type=str, required=True, 
						help='Path to csv dataframe of readings')
	parser.add_argument('--data_dict', type=str, required=True,
						help='Path to json data dictionary file')
	
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

	ts_df = pd.read_csv(args.data)
	data_dict = None
	
	if args.collapse:
		ts_df = collapse(ts_df, args)
		data_dict = update_data_dict_collapse(args)
	elif args.add_feature:
		ts_df = add_new_feature(ts_df, args)
		data_dict = update_data_dict_add_feature(args)

	ts_df.to_csv('ts_transformed.csv', index=False)
	with open('transformed.json', 'w') as f:
		json.dump(data_dict, f)
	f.close()


# COLLAPSE TIME SERIES

# TODO: Possibly rearrange this to be simpler, ie put most of collapse func
#		in the recursive steps
def collapse(ts_df, args): 
	id_cols = parse_id_cols(args.data_dict)
	non_time_cols = parse_non_time_cols(args.data_dict)
	time_cols = parse_time_cols(args.data_dict)

	# define new columns, apply every summary statistic to every
	# value column
	new_cols = []
	for col in ts_df.columns.values:
		if col in non_time_cols:
			new_cols.append(col)
		if col not in non_time_cols and col not in time_cols:
			for operation in args.collapse_features.split(' '):
				new_cols.append('{}_{}'.format(col, operation))

	rows = []
	# break down summary groups by id columns 
	# (ie [[name1, chunk1], [name1, chunk2], ...])
	id_groups = []
	all_id_combinations(id_cols, ts_df, id_groups)

	for id_group in id_groups: 
		query_string = ''
		for id_col, id_value, i in zip(id_cols, id_group, range(len(id_group))):
			query_string += '{} == "{}"'.format(id_col, id_value) 
			if i + 1 < len(id_group):
				query_string += ' and '
		group_df = ts_df.query(query_string)

		# add summary row 
		new_row = []
		for col in ts_df.columns.values:
			if col in non_time_cols:
				new_row.append(group_df[col].iloc[0])
			if col not in non_time_cols and col not in time_cols:
				for operation in args.collapse_features.split(' '):
					new_row.append(convert_to_summary_stat(group_df, col, 
												   	   	   operation))
		rows.append(new_row)

	new_ts_df = pd.DataFrame(data=rows, columns=new_cols)
	return new_ts_df

def all_id_combinations(cols, df, combos, ids=[]):
	if len(cols) == 0:
		combos.append(ids)
		return

	for i in df[cols[0]].unique():
		ids_copy = list(ids)
		ids_copy.append(i)
		all_id_combinations(cols[1:], df.loc[df[cols[0]] == i], 
							combos, ids_copy)

def convert_to_summary_stat(df, col, op):
	if op == 'mean':
		return np.mean(df[col].tolist())
	elif op == 'std_dev':
		return np.std(df[col].tolist())
	elif op == 'median':
		return np.median(df[col].tolist())
	elif op == 'min_val':
		return np.amin(df[col].tolist())
	elif op == 'max_val':
		return np.amax(df[col].tolist())


# ADD NEW FEATURE COLUMN

def add_new_feature(ts_df, args):
	new_col_name = '{}_{}'.format(args.add_from, args.new_feature) 
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

	new_fields = []
	for col in data_dict['fields']:
		# cuts out time columns
		if 'role' in col and (col['role'] == 'id' or
							  col['role'] == 'sequence' or
							  col['role'] == 'outcowme' or 
							  col['role'] == 'other'):
			new_fields.append(col)
		if 'role' in col and col['role'] == 'feature':
			for operation in args.collapse_features.split(' '):
				new_dict = dict(col)
				new_dict['name'] = '{}_{}'.format(col['name'], operation)
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
			new_dict['name'] = '{}_{}'.format(col['name'], args.new_feature)
			new_fields.append(new_dict)
		else: 
			new_fields.append(col)

	new_data_dict = dict()
	new_data_dict['fields'] = new_fields
	return new_data_dict

def parse_id_cols(data_dict_file):	
	cols = []
	with open(data_dict_file, 'r') as f:
		data_dict = json.load(f)
	f.close()

	for col in data_dict['fields']:
		if 'role' in col and col['role'] == 'id':
			cols.append(col['name'])
	return cols

def parse_non_time_cols(data_dict_file):
	non_time_cols = []
	with open(data_dict_file, 'r') as f:
		data_dict = json.load(f)
	f.close()

	for col in data_dict['fields']:
		# We don't care about time, we want to remove it
		if 'role' in col and (col['role'] == 'id' or 
		    				  col['role'] == 'sequence' or
							  col['role'] == 'outcome' or 
							  col['role'] == 'other'):
			non_time_cols.append(col['name'])
	return non_time_cols

def parse_time_cols(data_dict_file):
	time_cols = []
	with open(data_dict_file, 'r') as f:
		data_dict = json.load(f)
	f.close()

	for col in data_dict['fields']:
		if 'role' in col and col['role'] == 'time':
			time_cols.append(col['name'])
	return time_cols


if __name__ == '__main__':
    main()