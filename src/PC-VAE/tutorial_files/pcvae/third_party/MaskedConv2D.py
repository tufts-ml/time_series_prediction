import math

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Convolution2D

class MaskedConvolution2D(Convolution2D):
    def __init__(self, *args, mask='A' , n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask

        self.mask = None

    def build(self, input_shape):
        super().build(input_shape)

        # Create a numpy array of ones in the shape of our convolution weights.
        self.mask = np.ones(self.W_shape)

        # We assert the height and width of our convolution to be equal as they should.
        assert mask.shape[0] == mask.shape[1]

        # Since the height and width are equal, we can use either to represent the size of our convolution.
        filter_size = self.mask.shape[0]
        filter_center = filter_size / 2

        # Zero out all weights below the center.
        self.mask[math.ceil(filter_center):] = 0

        # Zero out all weights to the right of the center.
        self.mask[math.floor(filter_center):, math.ceil(filter_center):] = 0

        # If the mask type is 'A', zero out the center weigths too.
        if self.mask_type == 'A':
            self.mask[math.floor(filter_center), math.floor(filter_center)] = 0

        # Convert the numpy mask into a tensor mask.
        self.mask = K.variable(self.mask)

    def call(self, x, mask=None):
        ''' I just copied the Keras Convolution2D call function so don't worry about all this code.
            The only important piece is: self.W * self.mask.
            Which multiplies the mask with the weights before calculating convolutions. '''
        output = K.conv2d(x, self.W * self.mask, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)
        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        output = self.activation(output)
        return output

    def get_config(self):
        # Add the mask type property to the config.
        return dict(list(super().get_config().items()) + list({'mask': self.mask_type}.items()))
