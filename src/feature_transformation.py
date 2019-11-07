# feature_transformation.py

# Input: Requires a dataframe, specification of whether to collapse
#		 into summary statistics and which statistics, or whether to
# 		 add a transformation of a column and the operation with which
#		 to transform

#		 Requires a json data dictionary to accompany the dataframe,
#		 data dictionary columns must all have a defined "role" field

# Output: Puts transformed dataframe into ts_transformed.csv and 
#		  updated data dictionary into transformed.json


import sys
import pandas as pd
import argparse
import json
import numpy as np
from scipy import stats
import ast

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
						default='count mean median std min max', 
						help="Enclose options with 's, choose "
							 "from mean, std, min, max, "
							 "median, slope, count, present")
	parser.add_argument('--collapse_range_features', type=str, required=False,
						default='slope std', 
						help="Enclose options with 's, choose "
							 "from mean, std, min, max, "
							 "median, slope, count, present")
	parser.add_argument('--range_pairs', type=str, required=False,
						default='[(0, 10), (0, 25), (0, 50), (50, 100), (75, 100), (90, 100)]',
						help="Enclose pairs list with 's and [], list all desired ranges in "
							 "parentheses like this: '[(0, 50), (25, 75), (50, 100)]'")
	parser.add_argument('--max_time_step', type=int, required=False,
						default=-1, help="Specify the maximum number of time "
										 "steps to collapse on, for example, "
										 "input 48 for 48 hours at 1 hour time steps. "
										 "Set to -1 for no limit.")

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
		json.dump(data_dict, f, indent=4)
	f.close()


# COLLAPSE TIME SERIES
def collapse(ts_df, args): 
	id_and_output_cols = parse_id_and_output_cols(args.data_dict)
	id_and_output_cols = remove_col_names_from_list_if_not_in_df(id_and_output_cols, ts_df)
	feature_cols = parse_feature_cols(args.data_dict)
	feature_cols = remove_col_names_from_list_if_not_in_df(feature_cols, ts_df)
	operations = []
	for op in args.collapse_features.split(' '):
		operations.append(get_summary_stat_func(op, max_time_step=args.max_time_step))
	# without this check it still iterates once for some reason
	if len(args.collapse_range_features) > 0:
		
		for op in args.collapse_range_features.split(' '):
			for low, high in ast.literal_eval(args.range_pairs):
				operations.append(get_summary_stat_func(op, low_perc=low, 
														high_perc=high,
														max_time_step=args.max_time_step))

	# TODO: Retest when pandas updates and fixes their dropna=False bug. See
	#       pandas-dev/pandas#25738

	new_df = pd.pivot_table(ts_df, values=feature_cols, index=id_and_output_cols, 
							aggfunc=operations, dropna=True)

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

def get_summary_stat_func(op, low_perc=0, high_perc=100, max_time_step=-1):	
	return bind_func_for_range(op, low_perc, high_perc, max_time_step)


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

	new_fields = []
	for name in id_and_output_cols:
		for col in data_dict['fields']:
			if col['name'] == name: 
				new_fields.append(col)
	for op in args.collapse_features.split(' '):
		for name in feature_cols:
			for col in data_dict['fields']:
				if col['name'] == name: 
					new_dict = dict(col)
					new_dict['name'] = '{}_{}'.format(op, name)
					new_fields.append(new_dict)
	if len(args.collapse_range_features) > 0: 
		for op in args.collapse_range_features.split(' '):
			for low, high in ast.literal_eval(args.range_pairs):
				for name in feature_cols:
					for col in data_dict['fields']:
						if col['name'] == name: 
							new_dict = dict(col)
							new_dict['name'] = '{}_{}_to_{}_{}'.format(op, low, high, name)
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


# COLLAPSE FUNCTIONS AND DICT

def slope(data):
	if len(data.dropna()) > 1:
		slope, _, _, _, _ = stats.linregress(x=range(len(data.dropna())), y=data.dropna())
		return slope
	else:
		return 0

def standard_dev(data): 
	if len(data.dropna()) > 1:
		return np.std(data)
	else: 
		return 0

def count(data):
	return len(data.dropna())

def present(data): 
	return 1 if len(data.dropna() > 0) else 0

def bind_func_for_range(op, start_percentile, end_percentile, max_time_step):
	if max_time_step < -1 or max_time_step == 0:
		raise Exception('set_max_time_steps value must be -1 (for unlimited) '
						'or a positive number.')
	def closure_func(data):
		f = COLLAPSE_FUNCTIONS[op]
		# Treat the last data point as 100 percentile.
		if max_time_step == -1: 
			lower_bound = (len(data)*start_percentile)//100
			upper_bound = (len(data)*end_percentile)//100
		# Treat the max time step as 100 percentile.
		elif max_time_step < len(data):
			lower_bound = (max_time_step*start_percentile)//100
			upper_bound = (max_time_step*end_percentile)//100
		# Add nan values to list until it is the length of the max time step. 
		# Treat that as 100 percentile.
		else: 
			data = data.append(pd.Series([np.nan for i in range(max_time_step - len(data))]),
							   ignore_index=True)
			lower_bound = (len(data)*start_percentile)//100
			upper_bound = (len(data)*end_percentile)//100
		
		return f(data[lower_bound:upper_bound])
	
	if start_percentile == 0 and end_percentile == 100:
		closure_func.__name__ = op
	else:
		closure_func.__name__ = '{}_{}_to_{}'.format(op, start_percentile, end_percentile)

	return closure_func

COLLAPSE_FUNCTIONS = {
	"mean": np.mean,
	"std":  standard_dev,
	"median": np.nanmedian,
	"min": np.amin,
	"max": np.amax,
	"slope": slope, 
	"count": count,
	"present": present
}

if __name__ == '__main__':
    main()