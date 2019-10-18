import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser(description='Choose Y label')

parser.add_argument('--id', type=str, required=True,
						help='Location of vitals data for training')
parser.add_argument('--label', type=str, required=True,
						help='Location of vitals data for testing')
data_loc = "../../../datasets/unimib_shar_activities/v20190307"
test_loc = data_loc+"/test.csv"
train_loc = data_loc+"/train.csv"
args = parser.parse_args()
separator = args.id+"_id"
labels_loc = data_loc+"/metadata_per_"+args.id+".csv"
labels = pd.read_csv(labels_loc)
test_X = pd.read_csv(test_loc)
train_X = pd.read_csv(train_loc)
id_array_train_P = train_X[separator]
id_array_test_Q = test_X[separator]
label_array_train_P = labels[args.label][id_array_train_P]
label_array_test_Q = labels[args.label][id_array_test_Q]
label_array_train_P.to_csv(data_loc+"/train_Y.csv")
label_array_test_Q.to_csv(data_loc+"/test_Y.csv")
train_X.to_csv(data_loc+"/train_X.csv")
test_X.to_csv(data_loc+"/test_X.csv")