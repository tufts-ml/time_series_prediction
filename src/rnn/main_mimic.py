import sys, os
import argparse
import numpy as np 
import pandas as pd
import json

import torch
import skorch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                            balanced_accuracy_score, confusion_matrix, 
                            roc_auc_score)
from yattag import Doc
import matplotlib.pyplot as plt

from dataset_loader import TidySequentialDataCSVLoader
from RNNBinaryClassifier import RNNBinaryClassifier


def main():
    parser = argparse.ArgumentParser(description='PyTorch RNN with variable-length numeric sequences wrapper')
    
    parser.add_argument('--train_vitals_csv', type=str, required=True,
                        help='Location of vitals data for training')
    parser.add_argument('--test_vitals_csv', type=str, required=True,
                        help='Location of vitals data for testing')
    parser.add_argument('--metadata_csv', type=str, required=True,
                        help='Location of metadata for testing and training')
    parser.add_argument('--data_dict', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of sequences per minibatch')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='Number of epochs')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--save', type=str,  default='RNNmodel.pt',
                        help='path to save the final model')
    parser.add_argument('--report_dir', type=str, default='results',
                        help='dir in which to save results report')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cpu'

    # extract data
    train_vitals = TidySequentialDataCSVLoader(
        per_tstep_csv_path=args.train_vitals_csv,
        per_seq_csv_path=args.metadata_csv,
        idx_col_names=['subject_id', 'episode_id'],
        x_col_names='__all__',
        y_col_name='inhospital_mortality',
        y_label_type='')

    test_vitals = TidySequentialDataCSVLoader(
        per_tstep_csv_path=args.test_vitals_csv,
        per_seq_csv_path=args.metadata_csv,
        idx_col_names=['subject_id', 'episode_id'],
        x_col_names='__all__',
        y_col_name='inhospital_mortality',
        y_label_type='')

    X_train, y_train = train_vitals.get_batch_data(batch_id=0)
    X_test, y_test = test_vitals.get_batch_data(batch_id=0)

    # hyperparameter space
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 1]
    hyperparameters = dict(lr=learning_rate)

    # grid search
    rnn = RNNBinaryClassifier(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        callbacks=[
            #skorch.callbacks.ProgressBar(),
        ],
        module__rnn_type='LSTM',
        module__n_inputs=X_train.shape[-1],
        module__n_hiddens=10,
        module__n_layers=1,
        optimizer=torch.optim.SGD,)

    classifier = GridSearchCV(rnn, hyperparameters, n_jobs=-1, cv=5, verbose=1)
    best_rnn = classifier.fit(X_train, y_train)

    # View best hyperparameters
    best_lr = best_rnn.best_estimator_.get_params()['lr']

    y_pred_proba = best_rnn.predict_proba(X_test)
    y_pred = convert_proba_to_binary(y_pred_proba)

    # Brief Summary
    print('Best lr:', best_rnn.best_estimator_.get_params()['lr'])
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_pred))
    print('Log Loss:', log_loss(y_test, y_pred_proba))
    conf_matrix = confusion_matrix(y_test, y_pred)
    true_neg = conf_matrix[0][0]
    true_pos = conf_matrix[1][1]
    false_neg = conf_matrix[1][0]
    false_pos = conf_matrix[0][1]
    print('True Positive Rate:', float(true_pos) / (true_pos + false_neg))
    print('True Negative Rate:', float(true_neg) / (true_neg + false_pos))
    print('Positive Predictive Value:', float(true_pos) / (true_pos + false_pos))
    print('Negative Predictive Value', float(true_neg) / (true_neg + false_pos))

    create_html_report(args.report_dir, y_test, y_pred, y_pred_proba, 
                       hyperparameters, best_lr)

def create_html_report(report_dir, y_test, y_pred, y_pred_proba, hyperparameters, best_lr):
    try:
        os.mkdir(report_dir)
    except OSError:
        pass

    # Set up HTML report
    doc, tag, text = Doc().tagtext()

    # Metadata
    with tag('h2'):
        text('Random rnn Classifier Results')
    with tag('h3'):
        text('Hyperparameters searched:')
    with tag('p'):
        text('Learning rate: ', str(hyperparameters['lr']))

    # Model
    with tag('h3'):
        text('Parameters of best model:')
    with tag('p'):
        text('Learning rate: ', best_lr)
    
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

    true_neg = conf_matrix[0][0]
    true_pos = conf_matrix[1][1]
    false_neg = conf_matrix[1][0]
    false_pos = conf_matrix[0][1]
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

def convert_proba_to_binary(probabilites):
    return [0 if probs[0] > probs[1] else 1 for probs in probabilites]

if __name__ == '__main__':
    main()

