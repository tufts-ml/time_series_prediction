from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import binary_cross_entropy
from torch.autograd import Variable

class RNNBinaryClassifierModule(nn.Module):
    ''' RNNBinaryClassifierModule

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
            dropout_proba=0.0, dropout_proba_non_recurrent=0.0, bidirectional=False, convert_to_log_reg=False):
        super(RNNBinaryClassifierModule, self).__init__()
        self.drop = nn.Dropout(dropout_proba)
        self.dropout_proba_non_recurrent = dropout_proba_non_recurrent
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                n_inputs, n_hiddens, n_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout_proba)
            # convert to logistic regression for debugging/warm starting
            self.convert_to_log_reg = convert_to_log_reg
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
        self.output = nn.Linear(
            in_features=n_hiddens,
            out_features=2,
            bias=True) # weights initialized as samples from uniform[-1/n_features,1/n_features]

        # initalize weight for logistic regression conversion
        if self.convert_to_log_reg:
            init_weights_for_logistic_regression_conversion(self.rnn)
            self.first_pass=True
        self.double()
        
    def score(self, X, y, sample_weight=None):
        correct_predictions = 0
        total_predictions = 0
        results = self.forward(torch.DoubleTensor(X))
        for probabilties, outcome in zip(results, y):
            if probabilties[outcome] > 0.5:
                correct_predictions += 1
            total_predictions += 1

        return float(correct_predictions) / total_predictions
    
    def score_bce(self, X, y, sample_weight=None):
        
        # do forward computation
        results = self.forward(torch.DoubleTensor(X))
        return binary_cross_entropy(results[:,0],y.double())

    def forward(self, inputs_NTF, seq_lens_N=None, pad_val=0, return_hiddens=False):
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
        yproba_N2 : 2D array (n_sequences, 2)
            Each row gives probability that given sequence is class 0 or 1
            Each row sums to one
        
        hiddens_NTH : 3D array (n_sequences, n_timesteps, n_hiddens)
            Each (n,t) index gives the hidden-state vector at sequence n, timestep t
        '''
        N, T, F = inputs_NTF.shape

        if seq_lens_N is None:
            if T>1:
                seq_lens_N = torch.zeros(N, dtype=torch.int64)
                # account for collapsed features across time
                for n in range(N):
                    bmask_T = torch.all(inputs_NTF[n] == pad_val, dim=-1)
                    seq_lens_N[n] = np.searchsorted(bmask_T, 1)
            else:
                seq_lens_N = torch.ones(N, dtype=torch.int64)
                
                    

        ## Create PackedSequence representation to handle variable-length sequences
        # Requires sorting all sequences in current batch in descending order by length
        sorted_seq_lens_N, ids_N = seq_lens_N.sort(0, descending=True)
        _, rev_ids_N = ids_N.sort(0, descending=False)
        sorted_inputs_NTF = inputs_NTF[ids_N] 
        packed_inputs_PF = nn.utils.rnn.pack_padded_sequence(sorted_inputs_NTF, sorted_seq_lens_N, batch_first=True)
        
        # Apply dropout to the non-recurrent layer weights between LSTM layers before output ie is weights for h_(l-1)^t
        # See https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM for choosing the right weights
        if (self.dropout_proba_non_recurrent>0.0 and self.rnn.num_layers>1):
            dropout = nn.Dropout(p=self.dropout_proba_non_recurrent)
            self.rnn.weight_ih_l1 = torch.nn.Parameter(dropout(self.rnn.weight_ih_l1), requires_grad=True)
            self.rnn.bias_ih_l1 = torch.nn.Parameter(dropout(self.rnn.bias_ih_l1), requires_grad=True)
        
        # Apply the RNN
        if (self.convert_to_log_reg==False) :
            packed_outputs_PH, _ = self.rnn(packed_inputs_PF)
            # Unpack to N x T x H padded representation
            outputs_NTH, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs_PH, batch_first=True)
            # Apply weights + softmax to final timestep of each sequence
            end_hiddens_NH = outputs_NTH[range(N), sorted_seq_lens_N - 1]
            yproba_N2 = nn.functional.softmax(self.output(end_hiddens_NH), dim=-1)
            #yproba_N2 = nn.functional.logsigmoid(self.output(end_hiddens_NH))
            # Unsort and return
            if return_hiddens:
                return yproba_N2.index_select(0, rev_ids_N), outputs_NTH.index_select(0, rev_ids_N)
            else:
                return yproba_N2.index_select(0, rev_ids_N)    
        
        else:# convert to logistic regression
            assert (self.rnn.hidden_size == F),"Number of hidden units must equal number of input features for conversion to logistic regression!"
            
            if (self.first_pass==False):# weird handling of validation set of gridsearchcv and validation set of LSTM object
                if (N!=self.ht.shape[1])&(N!=self.htval.shape[1]):
                    init_weights_for_logistic_regression_conversion(self.rnn)
                    self.first_pass=True

            # set end hidden layer output to be same as input for logistic regression conversion
            h0 = torch.zeros(self.rnn.num_layers, N, self.rnn.hidden_size).double()
            c0 = torch.ones(self.rnn.num_layers, N, self.rnn.hidden_size).double()
            if (self.first_pass) & (self.training):
                packed_outputs_PH, (self.ht, self.ct) = self.rnn(packed_inputs_PF, (h0, c0))
            elif (self.first_pass==False) & (self.training):
                packed_outputs_PH, (self.ht, self.ct) = self.rnn(packed_inputs_PF, (self.ht,self.ct))
            elif (self.first_pass) & (self.training==False):# eval mode
                packed_outputs_PH, (self.htval, self.ctval) = self.rnn(packed_inputs_PF, (h0, c0))
                self.first_pass=False
            elif (self.first_pass==False) & (self.training==False):
                packed_outputs_PH, (self.htval, self.ctval) = self.rnn(packed_inputs_PF, (self.htval,self.ctval))
            outputs_NTH, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs_PH, batch_first=True)
            outputs_NTH = torch.log(outputs_NTH/(1-outputs_NTH))# inverse sigmoid the output of hidden units to get back input features
            outputs_NTH[torch.isinf(outputs_NTH)]=0 # remove inf's from sigmoid inversion
            end_hiddens_NH = outputs_NTH[range(N), sorted_seq_lens_N - 1]
            yproba_N2 = nn.functional.logsigmoid(self.output(end_hiddens_NH)).index_select(0, rev_ids_N)
            return yproba_N2

# function to handle weight initialization 
def init_weights(m):
    if type(m) == nn.Linear:
        torch.manual_seed(42)
        m.weight.data = torch.randn(m.weight.shape)
        print(m.weight)  
        
def init_weights_for_logistic_regression_conversion(rnn):
    # for this to work number of hidden units MUST be equal to number of input features
    
    try:
        # set the W_hi, W_hf, W_hg to large non-trainable values
        large_w_val = 10**6
        n_hiddens=int(rnn.weight_hh_l0.shape[0]/4)
        n_features=rnn.weight_hh_l0.shape[1]

        rnn.weight_hh_l0.requires_grad=False
        # W_hi, W_hf, W_hg
        rnn.weight_hh_l0[:3*n_hiddens,:]=torch.zeros((3*n_hiddens, n_features))
        # W_ho
        rnn.weight_hh_l0[3*n_hiddens:,:]=torch.zeros([n_hiddens, n_features])
        # set the b_ho to zero remove time-dependence and set the rest of the biases as large non-trainable values
        rnn.bias_hh_l0.requires_grad = False  
        # b_hi, b_hf, b_hg
        rnn.bias_hh_l0[:3*n_hiddens]=torch.zeros(3*n_hiddens)
        #b_ho
        rnn.bias_hh_l0[3*n_hiddens:] = torch.zeros(n_hiddens) 

        # set input to hidden connections such that W_io is identity and the rest of the input weights are all large non-trainable
        rnn.weight_ih_l0.requires_grad=False
        # W_ii, W_if, W_ig
        rnn.weight_ih_l0[:3*n_hiddens,:]=torch.zeros((3*n_hiddens, n_features))
        # W_io
        rnn.weight_ih_l0[3*n_hiddens:,:]=torch.eye(n_hiddens)


        # b_ii, b_if, b_ig
        rnn.bias_ih_l0.requires_grad=False
        rnn.bias_ih_l0[:3*n_hiddens]=torch.zeros(3*n_hiddens)+large_w_val
        # b_io
        rnn.bias_ih_l0[3*n_hiddens:] = torch.zeros(n_hiddens)    
    except RuntimeError:
        print('Number of hidden units must match number of input features for conversion to logistic regression!')
    

if __name__ == '__main__':
    N = 5   # n_sequences
    T = 10  # n_timesteps
    F = 3   # n_features
    H = 2   # n_hiddens

    np.random.seed(0)
    torch.random.manual_seed(0)

    # Generate random sequence data
    inputs_NTF = np.random.randn(N, T, F)
#     y_N = np.random.randint(0, 2, size=N)
    y_N_ = torch.rand(N, requires_grad = False)
    y_N = y_N_.detach().numpy().astype(int)
    seq_lens_N = np.random.randint(low=1, high=T, size=N)
    
    # Convert numpy to torch
    inputs_NTF_ = torch.from_numpy(inputs_NTF)
    seq_lens_N_ = torch.from_numpy(seq_lens_N)

    rnn_clf = RNNBinaryClassifierModule('LSTM', n_inputs=F, n_hiddens=H, n_layers=1)
    yproba_N2_, hiddens_NTH_ = rnn_clf.forward(inputs_NTF_, seq_lens_N_, return_hiddens=True)
    yproba_N2 = yproba_N2_.detach().numpy()
    hiddens_NTH = hiddens_NTH_.detach().numpy()
    accuracy_scores_N2 = rnn_clf.score(inputs_NTF, y_N)
    bce_scores_N2 = rnn_clf.score_bce(inputs_NTF_, y_N_)
    
    for n in range(N):
        print("==== Sequence %d" % n)
        print("X:")
        print(inputs_NTF[n, :seq_lens_N[n]])
        print("H:")
        print(hiddens_NTH[n, :seq_lens_N[n]])
        print("yproba:")
        print(yproba_N2[n])
        print("accuracy score:")
        print(accuracy_scores_N2)        
        print("BCE loss:")
        print(bce_scores_N2.detach().numpy())