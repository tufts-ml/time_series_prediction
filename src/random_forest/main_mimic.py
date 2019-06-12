import sys, os
import argparse
import numpy as np 
import pandas as pd
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from yattag import Doc
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser(description='sklearn forestRegression')
	
	parser.add_argument('--train_vitals_csv', type=str, required=True,
						help='Location of vitals data for training')
	parser.add_argument('--test_vitals_csv', type=str, required=True,
						help='Location of vitals data for testing')
	parser.add_argument('--metadata_csv', type=str, required=True,
						help='Location of metadata for testing and training')
	parser.add_argument('--data_dict', type=str, required=True)
	parser.add_argument('--seed', type=int, default=1111,
						help='random seed')
	parser.add_argument('--save', type=str,  default='LRmodel.pt',
						help='path to save the final model')
	parser.add_argument('--report_dir', type=str, default='results',
						help='dir in which to save results report')
	args = parser.parse_args()

	# extract data
	train_vitals = pd.read_csv(args.train_vitals_csv)
	test_vitals = pd.read_csv(args.test_vitals_csv)
	metadata = pd.read_csv(args.metadata_csv)

	X_train, y_train = extract_labels(train_vitals, metadata, args.data_dict)
	X_test, y_test = extract_labels(test_vitals, metadata, args.data_dict)

	# hyperparameter space
	n_estimators = [10, 100, 500, 1000]
	max_depth = [20, 40, 60, 80, 100]
	hyperparameters = dict(n_estimators=n_estimators, max_depth=max_depth)

	# grid search
	forest = RandomForestClassifier()
	classifier = GridSearchCV(forest, hyperparameters, cv=5, verbose=1)

	best_forest = classifier.fit(X_train, y_train)

	# View best hyperparameters
	best_n_estimators = best_forest.best_estimator_.get_params()['n_estimators']
	best_max_depth = best_forest.best_estimator_.get_params()['max_depth']

	y_pred = best_forest.predict(X_test)
	y_pred_proba = best_forest.predict_proba(X_test)

	# Brief Summary
	print('Best n_estimators:', best_forest.best_estimator_.get_params()['n_estimators'])
	print('Best max_depth:', best_forest.best_estimator_.get_params()['max_depth'])
	print('Accuracy:', accuracy_score(y_test, y_pred))
	print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_pred))
	print('Log Loss:', log_loss(y_test, y_pred_proba))
	conf_matrix = confusion_matrix(y_test, y_pred)
	true_pos = conf_matrix[0][0]
	true_neg = conf_matrix[1][1]
	false_pos = conf_matrix[1][0]
	false_neg = conf_matrix[0][1]
	print('True Positive Rate:', float(true_pos) / (true_pos + false_neg))
	print('True Negative Rate:', float(true_neg) / (true_neg + false_pos))
	print('Positive Predictive Value:', float(true_pos) / (true_pos + false_pos))
	print('Negative Predictive Value', float(true_neg) / (true_neg + false_pos))

	create_html_report(args.report_dir, y_test, y_pred, y_pred_proba, 
					   hyperparameters, best_penalty, best_C)

def create_html_report(report_dir, y_test, y_pred, y_pred_proba, hyperparameters, best_n_estimators, best_max_depth):
	try:
		os.mkdir(report_dir)
	except OSError:
		pass

	# Set up HTML report
	doc, tag, text = Doc().tagtext()

	# Metadata
	with tag('h2'):
		text('Random Forest Classifier Results')
	with tag('h3'):
		text('Hyperparameters searched:')
	with tag('p'):
		text('Number of estimators: ', str(hyperparameters['n_estimators']))
	with tag('p'):
		text('Max depth: ', str(hyperparameters['max_depth']))

	# Model
	with tag('h3'):
		text('Parameters of best model:')
	with tag('p'):
		text('Number of estimators: ', best_n_estimators)
	with tag('p'):
		text('Max depth: ', best_max_depth)
	
	# Performance
	with tag('h3'):
		text('Performance Metrics:')
	with tag('p'):
		text('Accuracy: ', accuracy_score(y_test, y_pred))
	with tag('p'):
		text('Balanced Accuracy: ', balanced_accuracy_score(y_test, y_pred))
	with tag('p'):
		text('Log Loss: ', log_loss(y_test, y_pred_proba))
	
	conf_matrix = confusion_matrix(y_test, y_pred)
	conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

	true_pos = conf_matrix[0][0]
	true_neg = conf_matrix[1][1]
	false_pos = conf_matrix[1][0]
	false_neg = conf_matrix[0][1]
	with tag('p'):
		text('True Positive Rate: ', float(true_pos) / (true_pos + false_neg))
	with tag('p'):
		text('True Negative Rate: ', float(true_neg) / (true_neg + false_pos))
	with tag('p'):
		text('Positive Predictive Value: ', float(true_pos) / (true_pos + false_pos))
	with tag('p'):
		text('Negative Predictive Value: ', float(true_neg) / (true_neg + false_pos))
	
	# Confusion Matrix
	columns = ['Predicted 1', 'Predicted 0']
	rows = ['Actual 1', 'Actual 0']
	cell_text = []
	for cm_row, cm_norm_row in zip(conf_matrix, conf_matrix_norm):
		row_text = []
		for i, i_norm in zip(cm_row, cm_norm_row):
			row_text.append('{} ({})'.format(i, i_norm))
		cell_text.append(row_text)

	ax = plt.subplot(111, frame_on=False) 
	ax.xaxis.set_visible(False) 
	ax.yaxis.set_visible(False)

	confusion_table = ax.table(cellText=cell_text,
							   rowLabels=rows,
							   colLabels=columns,
							   loc='center')
	plt.savefig(report_dir + '/confusion_matrix.png')
	plt.close()

	with tag('p'):
		text('Confusion Matrix:')
	doc.stag('img', src=('confusion_matrix.png'))

	# ROC curve/area
	y_pred_proba_pos, y_pred_proba_neg = zip(*y_pred_proba)
	fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_pos)
	roc_area = roc_auc_score(y_test, y_pred_proba_pos)
	plt.plot(fpr, tpr)
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.savefig(report_dir + '/roc_curve.png')
	plt.close()

	with tag('p'):
		text('ROC Curve:')
	doc.stag('img', src=('roc_curve.png'))	
	with tag('p'):
		text('ROC Area: ', roc_area)

	with open(report_dir + '/report.html', 'w') as f:
		f.write(doc.getvalue())

def extract_labels(vitals, metadata, data_dict):
	id_cols = parse_id_cols(data_dict)
	outcome = parse_outcome_col(data_dict)
	print(id_cols)

	df = pd.merge(vitals, metadata, on=id_cols, how='left')
	y = list(df[outcome])

	if len(vitals) != len(y):
		raise Exception('Number of sequences did not match number of labels.')

	return vitals, y

def parse_id_cols(data_dict_file):   
	cols = []
	with open(data_dict_file, 'r') as f:
		data_dict = json.load(f)
	f.close()

	for col in data_dict['fields']:
		if 'role' in col and col['role'] == 'id':
			cols.append(col['name'])
	return cols

def parse_outcome_col(data_dict_file):   
	cols = []
	with open(data_dict_file, 'r') as f:
		data_dict = json.load(f)
	f.close()

	for col in data_dict['fields']:
		if 'role' in col and col['role'] == 'outcome':
			return col['name']
	return ''

if __name__ == '__main__':
	main()

