import sys, os
import argparse
import numpy as np 
import pandas as pd
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
							balanced_accuracy_score, confusion_matrix, multilabel_confusion_matrix,
							roc_auc_score)

from yattag import Doc
import matplotlib.pyplot as plt


def main():
	parser = argparse.ArgumentParser(description='sklearn LogisticRegression')
	
	parser.add_argument('--train_activity_csv', type=str, required=True,
						help='Location of activities data for training')
	parser.add_argument('--test_activity_csv', type=str, required=True,
						help='Location of activities data for testing')
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
	train_activity = pd.read_csv(args.train_activity_csv)
	test_activity = pd.read_csv(args.test_activity_csv)
	metadata = pd.read_csv(args.metadata_csv)

	X_train, y_train = extract_labels(train_activity, metadata, args.data_dict)
	X_test, y_test = extract_labels(test_activity, metadata, args.data_dict)

	# hyperparameter space
	penalty = ['l1', 'l2']
	C = [ 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3, 1e4]
	hyperparameters = dict(C=C, penalty=penalty)

	# grid search
	#configure multiclass classification option based on checking metadata?

	logistic = LogisticRegression(solver='liblinear', max_iter=10000, multi_class="auto")
	classifier = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1)

	best_logistic = classifier.fit(X_train, y_train)

	# View best hyperparameters
	best_penalty = best_logistic.best_estimator_.get_params()['penalty']
	best_C = best_logistic.best_estimator_.get_params()['C']

	y_pred = best_logistic.predict(X_test)
	y_pred_proba = best_logistic.predict_proba(X_test)

	# Brief Summary
	print('Best Penalty:', best_penalty)
	print('Best C:', best_C)
	print('Accuracy:', accuracy_score(y_test, y_pred))
	print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_pred))
	print('Log Loss:', log_loss(y_test, y_pred_proba))
	conf_matrix = multilabel_confusion_matrix(y_test, y_pred)
	true_neg = conf_matrix[0][0]
	true_pos = conf_matrix[1][1]
	false_neg = conf_matrix[1][0]
	false_pos = conf_matrix[0][1]
	print(conf_matrix)
	print('True Positive Rate:', float(true_pos) / (true_pos + false_neg))
	print('True Negative Rate:', float(true_neg) / (true_neg + false_pos))
	print('Positive Predictive Value:', float(true_pos) / (true_pos + false_pos))
	print('Negative Predictive Value', float(true_neg) / (true_neg + false_pos))


	create_html_report(args.report_dir, y_test, y_pred, y_pred_proba, 
					   hyperparameters, best_penalty, best_C)

def create_html_report(report_dir, y_test, y_pred, y_pred_proba, hyperparameters, best_penalty, best_C):
	try:
		os.mkdir(report_dir)
	except OSError:
		pass

	# Set up HTML report
	doc, tag, text = Doc().tagtext()

	# Metadata
	with tag('h2'):
		text('Logistic Classifier Results')
	with tag('h3'):
		text('Hyperparameters searched:')
	with tag('p'):
		text('Penalty: ', str(hyperparameters['penalty']))
	with tag('p'):
		text('C: ', str(hyperparameters['C']))

	# Model
	with tag('h3'):
		text('Parameters of best model:')
	with tag('p'):
		text('Penalty: ', best_penalty)
	with tag('p'):
		text('C: ', best_C)
	
	# Performance
	with tag('h3'):
		text('Performance Metrics:')
	with tag('p'):
		text('Accuracy: ', accuracy_score(y_test, y_pred))
	with tag('p'):
		text('Balanced Accuracy: ', balanced_accuracy_score(y_test, y_pred))
	with tag('p'):
		text('Log Loss: ', log_loss(y_test, y_pred_proba))
	
	conf_matrix = multilabel_confusion_matrix(y_test, y_pred)
	print(conf_matrix)
	conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

	true_neg = conf_matrix[0][0]
	true_pos = conf_matrix[1][1]
	false_neg = conf_matrix[1][0]
	false_pos = conf_matrix[0][1]
	#change way of calculating metrics and confusion matrix
	with tag('p'):
		text('True Positive Rate: ', float(true_pos) / (true_pos + false_neg))
	with tag('p'):
		text('True Negative Rate: ', float(true_neg) / (true_neg + false_pos))
	with tag('p'):
		text('Positive Predictive Value: ', float(true_pos) / (true_pos + false_pos))
	with tag('p'):
		text('Negative Predictive Value: ', float(true_neg) / (true_neg + false_pos))
	
	# Confusion Matrix
	columns = ['Predicted 0', 'Predicted 1']
	rows = ['Actual 0', 'Actual 1']
	cell_text = []
	for cm_row, cm_norm_row in zip(conf_matrix, conf_matrix_norm):
		row_text = []
		for i, i_norm in zip(cm_row, cm_norm_row):
			row_text.append('{} ({})'.format(i, i_norm))
		cell_text.append(row_text)

	ax = plt.subplot(111, frame_on=False) 
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	print(rows)
	print(columns)
	print(cell_text)
	
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
	y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
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
	print(outcome)
	print(id_cols)
	df = pd.merge(vitals, metadata, on=id_cols, how='left')
	print(df)
	y = list(df[outcome])
	if len(vitals) != len(y):
		raise Exception('Number of sequences did not match number of labels.')
	print("extract")
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

