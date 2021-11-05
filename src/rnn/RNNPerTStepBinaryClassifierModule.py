from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

class RNNPerTStepBinaryClassifierModule(nn.Module):
    ''' RNNPerTStepBinaryClassifierModule

    NeuralNet module for binary classification of sequences/time-series

    Examples
    --------
    # Apply random weights to data
    >>> RNNBinaryClassifierModule('LSTM', 1, 3, 1)

    Args
    ----
    rnn_type : str
        One of ['LSTM', 'GRU', 'ELMAN+relu', 'ELMAN+tanh']
    n_inputs : int
        Number of input features
    n_hiddens : int    
        Number of hidden units in each rnn cell 
    '''
    def __init__(self,
            rnn_type='LSTM', n_inputs=1, n_hiddens=1, n_layers=1,
            dropout_proba=0.0, bidirectional=False, seq_lens_N=None, initialization_gain=1.0):
        super(RNNPerTStepBinaryClassifierModule, self).__init__()
        self.drop = nn.Dropout(dropout_proba)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                n_inputs, n_hiddens, n_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout_proba)
        elif rnn_type in ['ELMAN+tanh', 'ELMAN+relu']:
            nonlinearity = rnn_type.split("+")[1]
            self.rnn = nn.RNN(
                n_inputs, n_hiddens, n_layers,
                nonlinearity=nonlinearity,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout_proba)
        else:
            raise ValueError("Invalid option for --rnn_type: %s" % rnn_type)
        self.output = nn.Linear(n_hiddens, 2)
        self.seq_lens_N=seq_lens_N
        
        
        # initialize with Glorot
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0 , gain=initialization_gain)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0 , gain=initialization_gain)
        nn.init.xavier_uniform_(self.output.weight, gain=initialization_gain)
        
        # Setup to use double-precision floats (like np.float64)
        self.float()

    def score(self, X, y, sample_weight=None):
        correct_predictions = 0
        total_predictions = 0

        results = self.forward(torch.DoubleTensor(X))
        for probabilties, outcome in zip(results, y):
            if probabilties[outcome] > 0.5:
                correct_predictions += 1
            total_predictions += 1

        return float(correct_predictions) / total_predictions

    def forward(self, inputs_NTF, seq_lens_N=None, pad_val=np.nan, return_hiddens=False, apply_log_softmax=True):
        ''' Forward pass of input data through NN module

        Cleanly handles variable-length sequences (though internals a bit messy).

        Args
        ----
        inputs_NTF : 3D array (n_sequences, n_timesteps, n_features)
            Each row is one sequence, padded to length T = n_timesteps
        seq_lens_N : 1D array-like (n_sequences)
            Each entry indicates how many timesteps the n-th sequence has.
            (Remaining entries are all padding and should be ignored).

        Returns
        -------
        yproba_NT2 : 3D array (n_sequences, n_timepoints, 2)
        yproba_NT2[i] gives a matrix of size (n_timepoints, 2) denoting class 0 or class 1 log probability at each timepoint 
        
        hiddens_NTH : 3D array (n_sequences, n_timesteps, n_hiddens)
            Each (n,t) index gives the hidden-state vector at sequence n, timestep t
        '''
        N, T, F = inputs_NTF.shape

        if self.seq_lens_N is None:
            keep_inds = torch.logical_not(torch.all(torch.isnan(inputs_NTF), dim=-1))
            seq_lens_N = torch.sum(keep_inds, axis=1)
        else:
            seq_lens_N = self.seq_lens_N
        
        ## Create PackedSequence representation to handle variable-length sequences
        # Requires sorting all sequences in current batch in descending order by length
        sorted_seq_lens_N, ids_N = seq_lens_N.sort(0, descending=True)
        _, rev_ids_N = ids_N.sort(0, descending=False)
        sorted_inputs_NTF = inputs_NTF[ids_N]
        packed_inputs_PF = nn.utils.rnn.pack_padded_sequence(sorted_inputs_NTF, sorted_seq_lens_N, batch_first=True)
        ## Apply the RNN
        packed_outputs_PH, _ = self.rnn(packed_inputs_PF)
        ## Unpack to N x T x H padded representation
        outputs_NTH, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs_PH, batch_first=True)
        ## Apply weights + softmax to final timestep of each sequence
        if apply_log_softmax:
            yproba_NT2 = nn.functional.log_softmax(self.output(outputs_NTH), dim=-1)
        else:
            yproba_NT2 = self.output(outputs_NTH)
        ## Unsort and return
        if return_hiddens:
            return yproba_NT2.index_select(0, rev_ids_N), outputs_NTH.index_select(0, rev_ids_N)
        else:
            return yproba_NT2.index_select(0, rev_ids_N)

if __name__ == '__main__':
    N = 5   # n_sequences
    T = 10  # n_timesteps
    F = 3   # n_features
    H = 2   # n_hiddens

    np.random.seed(0)
    torch.random.manual_seed(0)
    
    
    # Generate random sequence data
    inputs_NTF = np.random.randn(N, T, F)
    seq_lens_N = np.random.randint(low=1, high=T, size=N)
    
    # Convert numpy to torch
    inputs_NTF_ = torch.from_numpy(inputs_NTF).float()
    seq_lens_N_ = torch.from_numpy(seq_lens_N)

    rnn_clf = RNNPerTStepBinaryClassifierModule('LSTM', n_inputs=F, n_hiddens=H, n_layers=1)
    yproba_NT2_, hiddens_NTH_ = rnn_clf.forward(inputs_NTF_, seq_lens_N_, return_hiddens=True)
    yproba_NT2 = yproba_NT2_.detach().numpy()
    hiddens_NTH = hiddens_NTH_.detach().numpy()
    for n in range(N):
        print("==== Sequence %d" % n)
        print("X:")
        print(inputs_NTF[n, :seq_lens_N[n]])
        print("H:")
        print(hiddens_NTH[n, :seq_lens_N[n]])
        print("yproba:")
        print(yproba_NT2[n, :seq_lens_N[n]])