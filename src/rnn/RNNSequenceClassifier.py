from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

class RNNSequenceClassifier(nn.Module):
    ''' RNNSequenceClassifier

    Args
    ----
    n_input_features : Number of features for each timestep 
    n_layers : Number of LSTMCell layers
    n_units_per_layer : Number of hidden units per LSTMCell

    '''

    def __init__(self,
            rnn_type='LSTM', n_inputs=1, n_hiddens=1, n_layers=1,
            dropout_proba=0.0, bidirectional=False):
        super(RNNSequenceClassifier, self).__init__()
        self.drop = nn.Dropout(dropout_proba)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                n_inputs, n_hiddens, n_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout_proba)
        elif rnn_type in ['RNN+tanh', 'RNN+relu']:
            nonlinearity = rnn_type.split("+")[1]
            self.rnn = nn.RNN(
                n_inputs, n_hiddens, n_layers,
                nonlinearity=nonlinearity,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout_proba)
        else:
            raise ValueError("Invalid option for --rnn_type: %s" % rnn_type)
        self.output = nn.Linear(self.n_hiddens, 2)

    def forward(self, inputs_NTF, hiddens_NTH=None):
        ''' Forward pass thru RNNSequenceClassifier

        Args
        ----
        inputs : 3D array (n_sequences, n_timesteps, n_features)
        hiddens : 3D array (n_sequences, n_timesteps, n_hiddens)

        Returns
        -------
        hiddens : 3D array (n_sequences, n_timesteps, n_hiddens)
        '''
        outputs_NTH, _ = self.rnn(inputs_NTF)
        out_proba_N2 = F.softmax(self.output(outputs_NTH[:,-1]), dim=-1)
        return out_proba_N2, outputs_NTH

if __name__ == '__main__':
    N = 5
    T = 100
    F = 3
    inputs_NTF = np.random.randn(N, T, F)

    rnn_clf = RNNSequenceClassifier(n_inputs=F, n_hiddens=H, n_layers=1)
    out = rnn_clf.forward(inputs_NTF)

    from IPython import embed; embed()
