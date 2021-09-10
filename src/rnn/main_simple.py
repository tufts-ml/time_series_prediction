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
        '--batch_size', type=int, default=20,
        help='Number of sequences per minibatch')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs')
    parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
    parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = 'cpu'

    dataset = TidySequentialDataCSVLoader('my_dataset.csv')
    X, y = dataset.get_batch_data(batch_id=0)

    rnn = RNNBinaryClassifier(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        callbacks=[
            #skorch.callbacks.Checkpoint(),
            skorch.callbacks.ProgressBar(),
        ],
        module__rnn_type='ELMAN+relu',
        module__n_inputs=X.shape[-1],
        module__n_hiddens=10,
        module__n_layers=1,
        optimizer=torch.optim.SGD,
        optimizer__lr=1.00,
        )

    X, y = dataset.get_batch_data(batch_id=0)
    rnn.fit(X, y)

    ### Evaluate
    pd.set_option('display.precision', 3)
    for n in range(dataset.N):
        X, y = dataset.get_single_sequence_data(n)
        yproba = rnn.forward(X)
        result_df = pd.DataFrame(y, columns=['y_true'])
        result_df['Pr(y=1)'] = yproba[:,1]
        print("\n### Seq %d" % n)
        print(result_df.__str__())

    


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
