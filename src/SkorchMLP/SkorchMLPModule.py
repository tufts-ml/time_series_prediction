from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SkorchMLPModule(nn.Module):
    ''' SkorchMLPModule
    NeuralNet module for linear layer plus softmax
    Examples
    --------
    >>> SkorchLogisticRegressionModule(l2_penalty=0.1)
    Args
    ----
    '''
    def __init__(self,
                 n_features=0, 
                 n_hiddens=32,
                 dropout=0.0,
            ):
        super(SkorchMLPModule, self).__init__()
        self.n_features = n_features
        self.dropout = nn.Dropout(dropout)
        
        # Define the neural net layer 
        self.hidden_layer = nn.Linear(in_features=n_features,
                                 out_features=n_hiddens,
                                 bias=True)
        
        
        # Define linear weight for each feature, plus bias
        self.output_layer = nn.Linear(in_features=n_hiddens,
            out_features=1,
            bias=True)
        
        # setup activation
        self.activation = F.relu
        
        # Setup to use double-precision floats (like np.float64)
        self.double()


    def forward(self, x_NF_, apply_logsigmoid=True):
        ''' Forward pass of input data through NN module
            
            Math : log(p(y=1|w,x)) = log(e^(w'x) / (1 + e^(w'x))) or log(p(y=1|w,x)) = log(sigmoid(w'x))
            
        Args
        ----
        x_NF_ : 2D Torch array (n_examples, n_features)
            Each row is one example feature vector
        Returns
        -------
        y_logproba_N_ : 2D Torch array (n_sequences, 1)
            Each row gives log probability that given example is positive class
        '''
        # forward pass through NN
        y_before_final_layer_N_ = self.activation(self.hidden_layer.forward(x_NF_))
        
        # apply dropout
        y_before_final_layer_N_ = self.dropout(y_before_final_layer_N_)
        
        # pass output to the final layer 
        y_beforesigmoid_N_ = self.output_layer(y_before_final_layer_N_)
        
        if apply_logsigmoid:
            y_logproba_N_ = nn.functional.logsigmoid(y_beforesigmoid_N_)
            return y_logproba_N_
        
        else:
            return y_beforesigmoid_N_

if __name__ == '__main__':
    N = 10   # n_examples
    n_features = 3   # n_features

    np.random.seed(0)
    torch.random.manual_seed(0)

    mlp_module = SkorchMLPModule(n_features=n_features)

    print("Random Data!")
    # Generate random data
    x_NF = np.random.randn(N, n_features)
    # Convert numpy to torch
    x_NF_ = torch.from_numpy(x_NF)
    
    y_logproba_N_ = mlp_module.forward(x_NF_)
    y_logproba_N = y_logproba_N_.detach().numpy()
    for n in range(N):
        print("==== Example %d" % n)
        print("x[n=%d]" % n)
        print(x_NF[n])
        print("y_logproba[n=%d]" % n)
        print(y_logproba_N[n])