import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                            balanced_accuracy_score, confusion_matrix, 
                            roc_auc_score, make_scorer)

from yattag import Doc
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='sklearn LogisticRegression')

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
    
    # remove subject_id and episode_id from the train and test features
    X_train = X_train.iloc[:,2:]
    X_test = X_test.iloc[:,2:]
#------------------------------------------- TRAIN ----------------------------#
    # hyperparameter space
    penalty = ['l1','l2']
    C = [1e-5, 1e-4, \
         1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3, 1e4]
    hyperparameters = dict(C=C, penalty=penalty)

    # define a auc scorer function
    roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                                 needs_threshold=True)
    
#     from IPython import embed; embed()
    # grid search
    logistic = LogisticRegression(solver='liblinear', max_iter=10000,\
                                  class_weight = 'balanced',\
                                  random_state = 42,\
                                  tol = 1e-3) 
    classifier = GridSearchCV(logistic, hyperparameters, cv=5, verbose=10, scoring = roc_auc_scorer)  
    best_logistic = classifier.fit(X_train, y_train)

#------------------------------------------- REPORT ----------------------------#
    # View best hyperparameters
    best_penalty = best_logistic.best_estimator_.get_params()['penalty']
    best_C = best_logistic.best_estimator_.get_params()['C']

    y_pred = best_logistic.predict(X_test)
    y_pred_proba = best_logistic.predict_proba(X_test)
    
    # check performance on training data to check for overfitting 
    y_train_pred = best_logistic.predict(X_train)
    y_train_pred_proba = best_logistic.predict_proba(X_train)
    
    # Brief Summary
    print('Best Penalty:', best_penalty)
    print('Best C:', best_C)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_pred))
    print('Log Loss:', log_loss(y_test, y_pred_proba))
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)# to check for overfitting
    true_neg = conf_matrix[0][0]
    true_pos = conf_matrix[1][1]
    false_neg = conf_matrix[1][0]
    false_pos = conf_matrix[0][1]
    true_neg_train = conf_matrix_train[0][0]
    true_pos_train = conf_matrix_train[1][1]
    false_neg_train = conf_matrix_train[1][0]
    false_pos_train = conf_matrix_train[0][1]
    
    print('True Positive Rate:', float(true_pos) / (true_pos + false_neg))
    print('True Negative Rate:', float(true_neg) / (true_neg + false_pos))
    print('Positive Predictive Value:', float(true_pos) / (true_pos + false_pos))
    print('Negative Predictive Value', float(true_neg) / (true_neg + false_pos))
    
    print('True Positive Rate on training data:', float(true_pos_train) / (true_pos_train + false_neg_train))
    print('True Negative Rate on training data:', float(true_neg_train) / (true_neg_train + false_pos_train))
    print('Positive Predictive Value on training data:', float(true_pos_train) / (true_pos_train + false_pos_train))
    print('Negative Predictive Value on training data:', float(true_neg_train) / (true_neg_train + false_pos_train))
    t2 = time.time()
    print('time taken to run classifier : {} seconds'.format(t2-t1))
    create_html_report(args.report_dir, y_test, y_pred, y_pred_proba, 
                       y_train, y_train_pred, y_train_pred_proba, hyperparameters, best_penalty, best_C)

def create_html_report(report_dir, y_test, y_pred, y_pred_proba, 
                       y_train, y_train_pred, y_train_pred_proba, hyperparameters, best_penalty, best_C):
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
        text('Accuracy on training data: ', accuracy_score(y_train, y_train_pred))        
    with tag('p'):
        text('Balanced Accuracy: ', balanced_accuracy_score(y_test, y_pred))
    with tag('p'):
        text('Log Loss: ', log_loss(y_test, y_pred_proba))

    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    conf_matrix_train = confusion_matrix(y_train, y_train_pred)# to check for overfitting
    conf_matrix_norm_train = conf_matrix_train.astype('float') / conf_matrix_train.sum(axis=1)[:, np.newaxis]



    true_neg = conf_matrix[0][0]
    true_pos = conf_matrix[1][1]
    false_neg = conf_matrix[1][0]
    false_pos = conf_matrix[0][1]
    true_neg_train = conf_matrix_train[0][0]
    true_pos_train = conf_matrix_train[1][1]
    false_neg_train = conf_matrix_train[1][0]
    false_pos_train = conf_matrix_train[0][1]
    
    with tag('p'):
        text('True Positive Rate: ', float(true_pos) / (true_pos + false_neg))
    with tag('p'):
        text('True Positive Rate on training data: ', float(true_pos_train) / (true_pos_train + false_neg_train))        
    with tag('p'):
        text('True Negative Rate: ', float(true_neg) / (true_neg + false_pos))
    with tag('p'):
        text('True Negative Rate on training data: ', float(true_neg_train) / (true_neg_train + false_pos_train))        
    with tag('p'):
        text('Positive Predictive Value: ', float(true_pos) / (true_pos + false_pos))
    with tag('p'):
        text('Positive Predictive Value on training data: ', float(true_pos_train) / (true_pos_train + false_pos_train))        
    with tag('p'):
        text('Negative Predictive Value: ', float(true_neg) / (true_neg + false_pos))
    with tag('p'):
        text('Negative Predictive Value on training data: ', float(true_neg_train) / (true_neg_train + false_pos_train))

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

    confusion_table = ax.table(cellText=cell_text,
                               rowLabels=rows,
                               colLabels=columns,
                               loc='center')
    plt.savefig(report_dir + '/confusion_matrix_test.png')
    
    plt.close()
    # save confusion matrix for training data
    cell_text_train = []
    for cm_row, cm_norm_row in zip(conf_matrix_train, conf_matrix_norm_train):
        row_text = []
        for i, i_norm in zip(cm_row, cm_norm_row):
            row_text.append('{} ({})'.format(i, i_norm))
        cell_text_train.append(row_text)

    ax = plt.subplot(111, frame_on=False) 
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    confusion_table = ax.table(cellText=cell_text_train,
                               rowLabels=rows,
                               colLabels=columns,
                               loc='center')
    plt.savefig(report_dir + '/confusion_matrix_train.png')    
    
    
    plt.close()

    with tag('p'):
        text('Confusion Matrix on test data:')
    doc.stag('img', src=('confusion_matrix_test.png'))

    with tag('p'):
        text('Confusion Matrix on training data:')
    doc.stag('img', src=('confusion_matrix_train.png'))
    
    
    # ROC curve/area
    y_pred_proba_neg, y_pred_proba_pos = zip(*y_pred_proba)
    y_train_pred_proba_neg, y_train_pred_proba_pos = zip(*y_train_pred_proba)    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_pos)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred_proba_pos)    
    roc_area = roc_auc_score(y_test, y_pred_proba_pos)
    roc_area_train = roc_auc_score(y_train, y_train_pred_proba_pos)
    print('train ROC : {}'.format(roc_area_train))
    print('test ROC : {}'.format(roc_area))
    plt.plot(fpr, tpr)
    plt.xlabel('FPR Test')
    plt.ylabel('TPR Test')
    plt.title('AUC : {}'.format(roc_area))
    plt.savefig(report_dir + '/roc_curve_test.png')
    plt.close()
    
    # plot training ROC 
    plt.plot(fpr_train, tpr_train)
    plt.xlabel('FPR Train')
    plt.ylabel('TPR Train')
    plt.title('AUC : {}'.format(roc_area_train))
    plt.savefig(report_dir + '/roc_curve_train.png')
    plt.close()    

    with tag('p'):
        text('ROC Curve for test data:')
    doc.stag('img', src=('roc_curve_test.png'))	
    with tag('p'):
        text('ROC Area for test data: ', roc_area)
        
    with tag('p'):
        text('ROC Curve for training data:')
    doc.stag('img', src=('roc_curve_train.png'))	
    with tag('p'):
        text('ROC Area for training data: ', roc_area_train)        

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
