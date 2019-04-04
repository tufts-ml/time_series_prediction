# feature_transformation.py

# Input: Requires a dataframe, specification of whether to collapse
#		 into summary statistics and which statistics, or whether to
# 		 add a transformation of a column and the operation with which
#		 to transform

#		 Requires a json data dictionary to accompany the dataframe,
#		 data dictionary columns must all have a defined "role" field

# Output: Puts transformed dataframe into ts_transformed.csv and 
#		  updated data dictionary into transformed.json

# Warning: Calculating slope on a column or half of a column requires at
#		   least two data points, otherwise the outputed data file and dict
#		   will be wrong. 

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
						default='mean median std min max', 
						help="Enclose options with 's, choose "
							 "from mean, std, min, max, "
							 "median, slope")
	parser.add_argument('--collapse_half_features', type=str, required=False,
						default='slope std', 
						help="Enclose options with 's, choose "
							 "from mean, std, min, max, "
							 "median, slope")
	
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
		operations.append(get_summary_stat_func(op))
	# without this check it still iterates once for some reason
	if len(args.collapse_half_features) > 0: 
		for op in args.collapse_half_features.split(' '):
			operations.append(get_summary_stat_func(op, 'first'))
			operations.append(get_summary_stat_func(op, 'second'))

	# TODO: Retest when pandas updates and fixes their dropna=False bug. See
	#       pandas-dev/pandas#25738 
	new_df = pd.pivot_table(ts_df, values=feature_cols, index=id_and_output_cols, 
							aggfunc=operations, dropna=False)

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

def get_summary_stat_func(op, eval_range='all'):
	return COLLAPSE_FUNCTIONS[eval_range][op]

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
	if len(args.collapse_half_features) > 0: 
		for op in args.collapse_half_features.split(' '):
			for name in feature_cols:
				for col in data_dict['fields']:
					if col['name'] == name: 
						new_dict = dict(col)
						new_dict['name'] = '{}_first_half_{}'.format(op, name)
						new_fields.append(new_dict)
			for name in feature_cols:
				for col in data_dict['fields']:
					if col['name'] == name:
						new_dict = dict(col)
						new_dict['name'] = '{}_second_half_{}'.format(op, name)
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
# Pivot table requires that functions passed to aggfunc have names so
# it can turn those names into column names in the returned dataframe.
# Because of this, the aggregate functions cannot be made more specific 
# by using lambda to wrap them. Instead, every variation must be defined
# explicitly, which is done below. 
def slope(data):
	if not data.dropna().empty:
		slope, _, _, _, _ = stats.linregress(x=range(len(data)), y=data)
		return slope
	else:
		return np.nan

def mean_first_half(d):
	return np.mean(d[:(len(d)//2)])
def mean_second_half(d): 
	return np.mean(d[(len(d)//2):])
def mean_middle_half(d):
	return np.mean(d[(len(d)//4):(3*len(d)//4)])
def std_first_half(d):
	return np.std(d[:(len(d)//2)])
def std_second_half(d): 
	return np.std(d[(len(d)//2):])
def std_middle_half(d):
	return np.std(d[(len(d)//4):(3*len(d)//4)])
def median_first_half(d):
	return np.median(d[:(len(d)//2)])
def median_second_half(d): 
	return np.median(d[(len(d)//2):])
def median_middle_half(d):
	return np.median(d[(len(d)//4):(3*len(d)//4)])
def min_first_half(d):
	return np.amin(d[:(len(d)//2)])
def min_second_half(d): 
	return np.amin(d[(len(d)//2):])
def min_middle_half(d):
	return np.amin(d[(len(d)//4):(3*len(d)//4)])
def max_first_half(d):
	return np.amax(d[:(len(d)//2)])
def max_second_half(d): 
	return np.amax(d[(len(d)//2):])
def max_middle_half(d):
	return np.amax(d[(len(d)//4):(3*len(d)//4)])
def slope_first_half(d):
	return slope(d[:(len(d)//2)])
def slope_second_half(d): 
	return slope(d[(len(d)//2):])
def slope_middle_half(d):
	return slope(d[(len(d)//4):(3*len(d)//4)])

COLLAPSE_FUNCTIONS = {
	"all": {
		"mean": np.mean,
		"std":  np.std,
		"median": np.median,
		"min": np.amin,
		"max": np.amax,
		"slope": slope
	},
	"first": {
		"mean": mean_first_half,
		"std":  std_first_half,
		"median": median_first_half,
		"min": min_first_half,
		"max": max_first_half,
		"slope": slope_first_half
	},
	"second": {
		"mean": mean_second_half,
		"std":  std_second_half,
		"median": median_second_half,
		"min": min_second_half,
		"max": max_second_half,
		"slope": slope_second_half
	}, 
	"middle": {
		"mean": mean_middle_half,
		"std":  std_middle_half,
		"median": median_middle_half,
		"min": min_middle_half,
		"max": max_middle_half,
		"slope": slope_middle_half
	}
}

if __name__ == '__main__':
    main()