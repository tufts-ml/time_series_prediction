import sys
import pandas as pd
import argparse
import random
import json

def main():
	parser = argparse.ArgumentParser(
		description="Take a pre-existing EEG data file and remove a "
					"specified percent of values to generate a test file "
					"with missing data.")

	parser.add_argument('--data', type=str, required=True, 
						help='Path to csv dataframe of readings.')
	parser.add_argument('--data_dict', type=str, required=True,
						help='Path to dataframe\'s corresponding data dict.')
	parser.add_argument('--output', type=str, required=False,
						help='Path to csv dataframe of output.')
	parser.add_argument('--seed', type=int, required=False, default=None,
						help='Seed random generation for repeat experiments.')
	parser.add_argument('--percent_removed', type=float, required=False,
						default=0.1, help='Percentage of data values removed.')
	parser.add_argument('--columns', type=str, required=False, default=None,
						help='Specify columns to remove data from (cannot be ID cols).')

	args = parser.parse_args()
	ts_df = pd.read_csv(args.data)

	random.seed(args.seed)
	
	# generate columns to remove data points from from
	columns = parse_feature_cols(args.data_dict)
	if args.columns is not None:
		parsed_cols = args.columns.split(' ')
		columns = [col for col in columns if col in parsed_cols]
	
	# remove data points
	for col in columns:
		index_sample = random.sample(population=ts_df.index.tolist(), 
									 k=int(args.percent_removed * len(ts_df)))
		for i in index_sample:
			ts_df.at[i, col] = None

	# save data to file
	if args.output is None:
		file_name = args.input.split('/')[-1].split('.')[0]
		data_output = '{}_missing_data.csv'.format(file_name)
	elif args.output[-4:] == '.csv':
		data_output = args.output 
	else:
		data_output = '{}.csv'.format(args.output)
	ts_df.to_csv(data_output, index=False)
	print("Wrote to output CSV:\n%s" % (data_output))

def parse_feature_cols(data_dict_file):
	non_time_cols = []
	with open(data_dict_file, 'r') as f:
		data_dict = json.load(f)
	f.close()

	for col in data_dict['fields']:
		if 'role' in col and col['role'] == 'feature':
			non_time_cols.append(col['name'])
	return non_time_cols

if __name__ == '__main__':
    main()