from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import torch
from torch import nn
import torch.nn.functional as F

class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

class conv1DBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride, padding, pool, activate=True):
        super(conv1DBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv1d(c_in, c_out, kernel, 1, (kernel-1)//2),
                                   nn.MaxPool1d(pool, stride = stride, padding = padding))
        self.block = self.block.double()
        self.acti = activate
#         self.float()

    def forward(self, x):
        out = self.block(x)
        if self.acti:
            out = F.relu(out)
        return out

class flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    

def compute_linear_layer_input_dims(x, channels, kernel_sizes, strides, paddings, pools):
    conv_layers = []
    for i in range(len(channels) - 1):
        conv_layers.append(conv1DBlock(
            channels[i], channels[i + 1], kernel_sizes[i], strides[i], 
            paddings[i], pools[i], True))

    conv_layers.append(flatten())
    conv_net = nn.Sequential(*conv_layers)
    conv_out = conv_net(x)
    return conv_out.size(1)

class CNNBinaryClassifierModule(nn.Module):
    '''
    channels : channels should start with the original number of channels and 
            have 1 more element comparing with everything else.
    '''
    def __init__(self, channels, kernel_sizes, strides, paddings, pools, linear_in_units, linear_out_units, dropout_proba):
        super(CNNBinaryClassifierModule, self).__init__()
        conv_layers = []
        for i in range(len(channels) - 1):
            conv_layers.append(conv1DBlock(
                channels[i], channels[i + 1], kernel_sizes[i], strides[i], 
                paddings[i], pools[i], True))
        
        conv_layers.append(flatten())
        
        self.linear_in_units = linear_in_units
        self.linear_out_units = linear_out_units
        
        # compute convolution layer output
        self.conv_layers = nn.Sequential(*conv_layers)
        self.dropout_proba = dropout_proba
        
        # add linear layer of input dimension same as the flattened output cnn dimension
        lin_layers = []
        lin_layers.append(nn.Linear(self.linear_in_units, self.linear_out_units, bias=True))
#         lin_layers.append(nn.BatchNorm1d(self.linear_units))
        lin_layers.append(nn.ReLU())
        lin_layers.append(nn.Dropout(self.dropout_proba))
        
        # add final linear layer to output probability of shape num classes
        lin_layers.append(nn.Linear(self.linear_out_units
                                    , 2, bias=True))
        self.lin_layers = nn.Sequential(*lin_layers)
        self.lin_layers = self.lin_layers.double()
        

    def forward(self, x):
        conv_out = self.conv_layers(x)
        linear_out = self.lin_layers(conv_out)
        return nn.functional.softmax(linear_out, dim=-1)



if __name__ == '__main__':
    
    # test on random input
    input = torch.randn(20, 1, 50) 
    cnn = CNNBinaryClassifierModule(channels = [1, 10, 10], 
                                    kernel_sizes =[3, 3, 3], 
                                    strides = [2, 2, 2], 
                                    paddings = [1, 1, 1], 
                                    pools = [3, 3, 3], 
                                    linear_units=128) 
    
    
    output = cnn(input)
    
