import argparse
import numpy as np
import pandas as pd

import torch
import skorch
from sklearn.model_selection import GridSearchCV

from dataset_loader import TidySequentialDataCSVLoader
from RNNBinaryClassifier import RNNBinaryClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN with variable-length numeric sequences wrapper')

    parser.add_argument('--dataset_csv_path',
        type=str,
        default='my_dataset.csv',
        help='Location of dataset csv file')
    parser.add_argument(
        '--batch_size', type=int, default=3,
        help='Number of sequences per minibatch')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs')
    parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
    parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = 'cpu'

    train = TidySequentialDataCSVLoader(
        #per_tstep_csv_path='mimic_rnn_data_full/vitals_data_per_tstamp__normalized.csv',
        #per_seq_csv_path='mimic_rnn_data_full/metadata_per_seq.csv',
        per_tstep_csv_path='mimic_rnn_data_500seq/ts_train_balanced.csv',
        per_seq_csv_path='mimic_rnn_data_500seq/seq_train_balanced.csv',
        idx_col_names=['subject_id', 'episode_id'],
        x_col_names='__all__',
        y_col_name='inhospital_mortality',
        y_label_type='')
    X_train, y_train = train.get_batch_data(batch_id=0)

    test = TidySequentialDataCSVLoader(
        #per_tstep_csv_path='mimic_rnn_data_full/vitals_data_per_tstamp__normalized.csv',
        #per_seq_csv_path='mimic_rnn_data_full/metadata_per_seq.csv',
        per_tstep_csv_path='mimic_rnn_data_500seq/ts_test.csv',
        per_seq_csv_path='mimic_rnn_data_500seq/seq_test.csv',
        idx_col_names=['subject_id', 'episode_id'],
        x_col_names='__all__',
        y_col_name='inhospital_mortality',
        y_label_type='')
    X_test, y_test = test.get_batch_data(batch_id=0)

    rnn = RNNBinaryClassifier(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        callbacks=[
            #skorch.callbacks.Checkpoint(),
            skorch.callbacks.ProgressBar(),
        ],
        module__rnn_type='LSTM',
        module__n_inputs=X_train.shape[-1],
        module__n_hiddens=10,
        module__n_layers=1,
        optimizer=torch.optim.SGD,
        optimizer__lr=.01,
        )

    rnn.fit(X_train, y_train)

    # Evaluation
    pd.set_option('display.precision', 3)
    proba_0 = []
    proba_1 = []
    train_acc_outcomes = []
    test_acc_outcomes = []
    train_ypred = []
    test_ypred = []
    for n in range(train.N):
        X, y = train.get_single_sequence_data(n)
        yproba = float(rnn.forward(X)[:,1])
        train_ypred.append(round(yproba))
        train_acc_outcomes.append(round(yproba) == y)
    for n in range(test.N):
        X, y = test.get_single_sequence_data(n)
        yproba = float(rnn.forward(X)[:,1])
        test_ypred.append(round(yproba))
        test_acc_outcomes.append(round(yproba) == y)
    print(np.mean(train_acc_outcomes), np.mean(test_acc_outcomes))
    print(np.mean(train_ypred), np.mean(test_ypred))





"""

# Demonstrate the use of grid search by testing different learning
# rates while saving the best model at the end.

params = [
    {
        'lr': [10,20,30],
    },
]

pl = GridSearchCV(net, params)

pl.fit(corpus.train[:args.data_limit].numpy())

print("Results of grid search:")
print("Best parameter configuration:", pl.best_params_)
print("Achieved F1 score:", pl.best_score_)

print("Saving best model to '{}'.".format(args.save))
pl.best_estimator_.save_params(f_params=args.save)
"""
