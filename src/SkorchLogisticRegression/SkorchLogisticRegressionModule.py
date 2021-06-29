from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

class SkorchLogisticRegressionModule(nn.Module):
    ''' SkorchLogisticRegressionModule
    NeuralNet module for linear layer plus softmax
    Examples
    --------
    >>> SkorchLogisticRegressionModule(l2_penalty=0.1)
    Args
    ----
    '''
    def __init__(self,
            n_features=0, 
            ):
        super(SkorchLogisticRegressionModule, self).__init__()
        self.n_features = n_features
        # Define linear weight for each feature, plus bias
        self.linear_transform_layer = nn.Linear(
            in_features=n_features,
            out_features=1,
            bias=True)
        # Setup to use double-precision floats (like np.float64)
        self.double()

    def set_parameters(self, weights_F=None, bias=None):
        ''' Set internal parameters to specific passed values
        Args
        ----
        weights_F : 1D Numpy array-like (n_features,)
            Weight values (one per feature).
        bias : scalar or array-like
            Bias value
        Returns
        -------
        None. Weight and bias updated in-place.
        '''
        assert isinstance(weights_F, np.ndarray)
        weights_1F = np.asarray([weights_F.flatten()], dtype=np.float64)
        assert weights_1F.shape == (1, self.n_features)
        weights_1F_ = torch.from_numpy(weights_1F)
#         self.linear_transform_layer.weight = torch.nn.Parameter(weights_1F_)
        self.linear_transform_layer.weight.data = weights_1F_

        bias_1d_arr = np.asarray([float(bias)], dtype=np.float64)
        assert bias_1d_arr.shape == (1,)
        bias_ = torch.from_numpy(bias_1d_arr)
#         self.linear_transform_layer.bias = torch.nn.Parameter(bias_)
        self.linear_transform_layer.bias.data = bias_


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
        y_beforesigmoid_N_ = self.linear_transform_layer.forward(x_NF_)
        y_logproba_N_ = nn.functional.logsigmoid(y_beforesigmoid_N_)
#         y_logproba_N_ = nn.functional.logsoftmax(y_beforesigmoid_N_)
        return y_logproba_N_


if __name__ == '__main__':
    N = 10   # n_examples
    F = 3   # n_features

    np.random.seed(0)
    torch.random.manual_seed(0)

    lr_module = SkorchLogisticRegressionModule(n_features=F)
    print("Weights:")
    print(lr_module.linear_transform_layer.weight)
    print("Bias:")
    print(lr_module.linear_transform_layer.bias)

    lr_module.set_parameters(
        weights_F=0.001 * np.random.randn(F),
        bias=0.001 * np.random.randn())
    print("Weights:")
    print(lr_module.linear_transform_layer.weight)
    print("Bias:")
    print(lr_module.linear_transform_layer.bias)

    print("Random Data!")
    # Generate random data
    x_NF = np.random.randn(N, F)
    # Convert numpy to torch
    x_NF_ = torch.from_numpy(x_NF)
    
    y_logproba_N_ = lr_module.forward(x_NF_)
    y_logproba_N = y_logproba_N_.detach().numpy()
    for n in range(N):
        print("==== Example %d" % n)
        print("x[n=%d]" % n)
        print(x_NF[n])
        print("y_logproba[n=%d]" % n)
        print(y_logproba_N[n])