from tensorflow.keras.layers import Dense, Layer, Concatenate
from tensorflow.keras.layers import Flatten, Add, Lambda, Multiply, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, GlobalAveragePooling2D, AveragePooling2D, \
    UpSampling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, GlobalAveragePooling3D, AveragePooling3D, \
    UpSampling3D
from tensorflow.keras.layers import Reshape, Activation, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
from pcvae.util import is_number
import math
from ..util.util import LinearReshape
import functools
import tensorflow_addons as tfa

@tf.keras.utils.register_keras_serializable(package='Custom', name='PixelConstraint')
class PixelConstraint(tf.keras.constraints.Constraint):
  """Constrains weight tensors to be centered around `ref_value`."""

  def __init__(self, mask):
    self.mask = mask

  def __call__(self, w):
    return w * self.mask

  def get_config(self):
    return {'mask': self.mask}

def get_mask(input_dim, output_dim, kernel_size=3, mask_type=None):
    if mask_type not in ['a', 'b']:
        return lambda x: x

    mask_type, mask_n_channels = mask_type, 3

    mask = np.ones(
        (kernel_size, kernel_size, input_dim, output_dim),
        dtype='float32'
    )
    center = kernel_size // 2

    # Mask out future locations
    # filter shape is (height, width, input channels, output channels)
    mask[center + 1:, :, :, :] = 0.
    mask[center, center + 1:, :, :] = 0.

    # Mask out future channels
    for i in range(mask_n_channels):
        for j in range(mask_n_channels):
            if (mask_type == 'a' and i >= j) or (mask_type == 'b' and i > j):
                mask[
                center,
                center,
                i::mask_n_channels,
                j::mask_n_channels
                ] = 0.
    return PixelConstraint(mask)


def residual_block(inputs, filters=None, kernel_size=3, resample=None, mask=None, nonlinearity='elu', bn=True, wn=False,
                   name='',
                   conv_args={}):
    if mask != None and resample != None:
        raise Exception('Unsupported configuration')

    # Setup options
    input_filters = inputs.shape[-1]
    base_conv = functools.partial(tf.keras.layers.Conv2D, padding='same', **conv_args)
    base_transpose = functools.partial(tf.keras.layers.Conv2DTranspose, padding='same', strides=2, **conv_args)

    # WRN-style filter scaling as default
    if filters is None:
        if resample is None:
            filters = input_filters
        elif resample == 'down':
            filters = input_filters * 2
        else:
            filters = input_filters // 2

    if resample == 'down':
        conv_shortcut = functools.partial(base_conv, strides=2, filters=filters, kernel_size=1)
        conv_1 = functools.partial(base_conv, filters=input_filters, kernel_size=kernel_size)
        conv_2 = functools.partial(base_conv, strides=2, filters=filters, kernel_size=kernel_size)
    elif resample == 'up':
        conv_shortcut = functools.partial(base_transpose, strides=2, filters=filters, kernel_size=1)
        conv_1 = functools.partial(base_transpose, strides=2, filters=filters, kernel_size=kernel_size)
        conv_2 = functools.partial(base_conv, filters=filters, kernel_size=kernel_size)
    elif resample == None:
        conv_shortcut = functools.partial(base_conv, filters=filters, kernel_size=1)
        conv_1 = functools.partial(base_conv, filters=filters, kernel_size=kernel_size)
        conv_2 = functools.partial(base_conv, filters=filters, kernel_size=kernel_size)
    else:
        raise Exception('invalid resample value')

    if filters == input_filters and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut_mask = get_mask(input_filters, filters, kernel_size=1, mask_type=mask)
        shortcut = conv_shortcut(kernel_constraint=shortcut_mask, name=name+'_shortcut')(inputs)

    output = inputs
    if mask == None:
        output = tf.keras.layers.Activation(nonlinearity, name=name+'_actv_1')(output)
        output = conv_1(name=name+'_conv_1')(output)
        output = tf.keras.layers.Activation(nonlinearity, name=name+'_actv_2')(output)
        output = conv_2(use_bias=False, name=name+'_conv_2')(output)
        if bn:
            output = tf.keras.layers.BatchNormalization(gamma_initializer='glorot_uniform', name=name+'_bn')(output)
    else:
        # To use weight normalization for pixelcnn layers
        if wn:
            norm = tfa.layers.WeightNormalization
        else:
            norm = lambda x: x

        output = tf.keras.layers.Activation(nonlinearity, name=name+'_masked_actv_1')(output)
        mask_1 = get_mask(input_filters, filters, kernel_size=kernel_size, mask_type=mask)
        output_a = norm(conv_1(kernel_constraint=mask_1, name=name+'_masked_conv_1a'))(output)
        output_b = norm(conv_1(kernel_constraint=mask_1, name=name+'_masked_conv_1b'))(output)
        # Gated nonlinearity
        output = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x[0]) * tf.math.tanh(x[1]), name=name+'_masked_gated_atcv')([output_a, output_b])

        mask_2 = get_mask(filters, filters, kernel_size=kernel_size, mask_type=mask)
        output = norm(conv_2(kernel_constraint=mask_2, name=name+'_masked_conv_2'))(output)

    return tf.keras.layers.Add(name=name+'_res_add')([output, shortcut])


def dense_block(inputs, units=None, nonlinearity='elu', bn=True, name='', dense_args={}):
    inputs = tf.keras.layers.Flatten()(inputs)

    # Setup options
    input_units = inputs.shape[-1]

    if units == input_units:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = tf.keras.layers.Dense(units=units, **dense_args)(inputs)

    output = inputs
    output = tf.keras.layers.Activation(nonlinearity, name=name+'_dense_actv_1')(output)
    output = tf.keras.layers.Dense(units=units, name=name+'_dense_1', **dense_args)(output)
    output = tf.keras.layers.Activation(nonlinearity, name=name+'_dense_actv_2')(output)
    output = tf.keras.layers.Dense(units=units, name=name+'_dense_2', **dense_args)(output)
    if bn:
        output = tf.keras.layers.BatchNormalization(gamma_initializer='glorot_uniform', name=name+'_dense_bn')(output)

    return tf.keras.layers.Add(name=name+'_dense_res_add')([output, shortcut])


def residual_encoder(wrn_size=2, wrn_actv='elu', wrn_bn=True, wrn_kernel_size=3,
                     wrn_dense_layers=0, wrn_dense_units=200, input_shape=None, wrn_rescale=False,
                     wrn_blocks_per_resnet=3, wrn_levels=4, **kwargs):
    inputs = tf.keras.layers.Input(input_shape)
    output = Lambda(lambda a: a * 2. - 1., name='encoder_rescale')(inputs) if wrn_rescale else inputs
    output = residual_block(output, filters=wrn_size * 16, kernel_size=wrn_kernel_size, nonlinearity=wrn_actv,
                            bn=wrn_bn, name='encoder_initial')
    for level in range(wrn_levels):
        output = residual_block(output, resample='down', kernel_size=wrn_kernel_size, nonlinearity=wrn_actv, bn=wrn_bn
                                , name='encoder_downsample_' + str(level))
        for i in range(wrn_blocks_per_resnet - 1):
            output = residual_block(output, kernel_size=wrn_kernel_size, nonlinearity=wrn_actv, bn=wrn_bn, name='encoder_' + str(level) + '_' + str(i))

    for i in range(wrn_dense_layers):
        output = dense_block(output, units=wrn_dense_units, nonlinearity=wrn_actv, bn=wrn_bn, name='encoder_' + str(i))

    return tf.keras.Model(inputs=inputs, outputs=output, name='encoder')


def residual_decoder(wrn_size=2, wrn_actv='elu', wrn_bn=True, wrn_kernel_size=3,
                     wrn_dense_layers=0, wrn_dense_units=200, encoded_size=None, input_shape=None,
                     wrn_blocks_per_resnet=3, wrn_levels=4, **kwargs):
    inputs = tf.keras.layers.Input((encoded_size,))
    output = inputs
    for i in range(wrn_dense_layers):
        output = dense_block(output, units=wrn_dense_units, nonlinearity=wrn_actv, bn=wrn_bn, name='decoder_dense_' + str(i))

    starting_shape = (
    input_shape[0] // (2 ** wrn_levels), input_shape[1] // (2 ** wrn_levels), 16 * wrn_size * (2 ** wrn_levels))
    output = LinearReshape(starting_shape, name='decoder_reshape')(output)

    for level in range(wrn_levels):
        for i in range(wrn_blocks_per_resnet - 1):
            output = residual_block(output, kernel_size=wrn_kernel_size, nonlinearity=wrn_actv, bn=wrn_bn, name='decoder_' + str(level) + '_' + str(i))
        output = residual_block(output, resample='up', kernel_size=wrn_kernel_size, nonlinearity=wrn_actv, bn=wrn_bn, name='decoder_upsample_' + str(level))
    output = residual_block(output, kernel_size=wrn_kernel_size, nonlinearity=wrn_actv, bn=wrn_bn, name='decoder_output')

    return tf.keras.Model(inputs=inputs, outputs=output, name='decoder')

''' Old WRN code:
'''

class WRN(object):
    def __init__(self, wrn_size=2, wrn_actv=0.1, wrn_bn=True, wrn_kernel_size=3,
                 wrn_global_pool=False, wrn_combined_pool=False, wrn_weight_decay=0.0, wrn_skip=False,
                 wrn_dense_layers=0, wrn_dense_units=1000, input_shape=None, wrn_rescale=False, wrn_output_conv=0,
                 wrn_blocks_per_resnet=4, wrn_levels=4, wrn_mixed_precision=False, **kwargs):
        self.wrn_size = wrn_size
        self.wrn_actv = wrn_actv
        self.wrn_bn = wrn_bn
        self.wrn_global_pool = wrn_global_pool
        self.wrn_combined_pool = wrn_combined_pool
        self.wrn_weight_decay = wrn_weight_decay
        self.wrn_skip = wrn_skip
        self.wrn_dense_layers = wrn_dense_layers
        self.wrn_dense_units = wrn_dense_units
        self.input_shape = input_shape
        self.wrn_rescale = wrn_rescale
        self.wrn_blocks_per_resnet = wrn_blocks_per_resnet
        self.wrn_levels = wrn_levels
        self.wrn_output_conv = wrn_output_conv
        self._ks = wrn_kernel_size
        self._stride = 2
        self.policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16') if wrn_mixed_precision else tf.float32

    def ks(self):
        return tuple([min(self.input_shape[i], self._ks) for i in range(len(self.input_shape) - 1)])

    def stride(self):
        return tuple([(2 if self.input_shape[i] > 3 else 1) for i in range(len(self.input_shape) - 1)])

    def ones(self):
        return tuple([1] * (len(self.input_shape) - 1))

    def Conv(self):
        return [None, Conv2D, Conv3D][len(self.input_shape) - 2]

    def ConvTranspose(self):
        return [None, Conv2DTranspose, Conv3DTranspose][len(self.input_shape) - 2]

    def MaxPooling(self):
        return [None, MaxPooling2D, MaxPooling3D][len(self.input_shape) - 2]

    def GlobalAveragePooling(self):
        return [None, GlobalAveragePooling2D, GlobalAveragePooling3D][len(self.input_shape) - 2]

    def AveragePooling(self):
        return [None, AveragePooling2D, AveragePooling3D][len(self.input_shape) - 2]

    def Upsampling(self):
        return [None, UpSampling2D, UpSampling3D][len(self.input_shape) - 2]

    def pad(self, in_filter, out_filter):
        arr = [[0, 0]] * len(self.input_shape) + [
            [(out_filter - in_filter) // 2, math.ceil((out_filter - in_filter) / 2)]]
        return (lambda a: tf.pad(a, arr))

    def slice(self, out_filter):
        if len(self.input_shape) == 2:
            return (lambda a: a[:, :, :out_filter])
        elif len(self.input_shape) == 4:
            return (lambda a: a[:, :, :, :, :out_filter])
        return (lambda a: a[:, :, :, :out_filter])

    def layer(self, ltype, x, index, layers, call=False, args={}):
        if call:
            return layers[index].call(x), index + 1
        elif len(layers) > index:
            return layers[index](x), index + 1

        l = ltype(**args)
        layers.append(l)
        return l(x), index + 1

    def residual_block(self,
                       x, index, layers, in_filter, out_filter, stride, activate_before_residual=False, BN=False,
                       call=False, transpose=False,
                       wd=0.0):
        """Adds residual connection to `x` in addition to applying BN->ReLU->3x3 Conv.
        Args:
          x: Tensor that is the output of the previous layer in the model.
          in_filter: Number of filters `x` has.
          out_filter: Number of filters that the output of this layer will have.
          stride: Integer that specified what stride should be applied to `x`.
          activate_before_residual: Boolean on whether a BN->ReLU should be applied
            to x before the convolution is applied.
        Returns:
          A Tensor that is the result of applying two sequences of BN->ReLU->3x3 Conv
          and then adding that Tensor to `x`.
        """
        org_index, org_layers = index, layers
        if call or len(layers) > index:
            layers = layers[index]
        else:
            layers = []
            org_layers.append(layers)

        padding = 'same'  # if transpose else 'valid'
        Conv = self.ConvTranspose() if transpose else self.Conv()
        index = 0
        if activate_before_residual:  # Pass up RELU and BN activation for resnet
            with tf.name_scope('shared_activation'):
                if BN:
                    x, index = self.layer(BatchNormalization, x, index, layers, call,
                                          dict(gamma_initializer='glorot_uniform'))
                x, index = self.layer(self.get_activation(), x, index, layers, call, dict())
                orig_x = x
        else:
            orig_x = x

        block_x = x
        if not activate_before_residual:
            with tf.name_scope('residual_only_activation'):
                if BN:
                    block_x, index = self.layer(BatchNormalization, block_x, index, layers, call,
                                                dict(gamma_initializer='glorot_uniform'))
                block_x, index = self.layer(self.get_activation(), block_x, index, layers, call, dict())

        with tf.name_scope('sub1'):
            block_x, index = self.layer(Conv, block_x, index, layers, call,
                                        dict(filters=out_filter, kernel_size=self.ks(),
                                             strides=stride, padding=padding,
                                             kernel_regularizer=l2(wd),
                                             bias_regularizer=l2(wd),
                                             ))

        with tf.name_scope('sub2'):
            if BN:
                block_x, index = self.layer(BatchNormalization, block_x, index, layers, call,
                                            dict(gamma_initializer='glorot_uniform'))
            block_x, index = self.layer(self.get_activation(), block_x, index, layers, call, dict())
            block_x, index = self.layer(Conv, block_x, index, layers, call,
                                        dict(filters=out_filter, kernel_size=self.ks(),
                                             strides=1, padding=padding,
                                             kernel_regularizer=l2(wd),
                                             bias_regularizer=l2(wd),
                                             ))

        with tf.name_scope(
                'sub_add'):  # If number of filters do not agree then zero pad them
            if (stride[0] > 1 or stride[1] > 1) and not transpose:
                orig_x, index = self.layer(self.AveragePooling(), orig_x, index, layers, call,
                                           dict(pool_size=self.ks(), padding=padding, strides=stride))
            elif (stride[0] > 1 or stride[1] > 1):
                orig_x, index = self.layer(self.Upsampling(), orig_x, index, layers, call,
                                           dict(size=stride))

            if in_filter < out_filter:
                orig_x, index = self.layer(Lambda, orig_x, index, layers, call,
                                           dict(function=self.pad(in_filter, out_filter)))
            elif in_filter > out_filter:
                orig_x, index = self.layer(Lambda, orig_x, index, layers, call,
                                           dict(function=self.slice(out_filter)))

        x, _ = self.layer(Add, [orig_x, block_x], index, layers, call, dict())
        return x, org_index + 1

    def _res_add(self, index, layers, in_filter, out_filter, stride, x, orig_x, transpose=False, call=False):
        """Adds `x` with `orig_x`, both of which are layers in the model.
        Args:
          in_filter: Number of filters in `orig_x`.
          out_filter: Number of filters in `x`.
          stride: Integer specifying the stide that should be applied `orig_x`.
          x: Tensor that is the output of the previous layer.
          orig_x: Tensor that is the output of an earlier layer in the network.
        Returns:
          A Tensor that is the result of `x` and `orig_x` being added after
          zero padding and striding are applied to `orig_x` to get the shapes
          to match.
        """

        if (stride[0] > 1 or stride[1] > 1) and not transpose:
            orig_x, index = self.layer(self.AveragePooling(), orig_x, index, layers, call,
                                       dict(pool_size=self.ks(), padding='same', strides=stride))
        elif (stride[0] > 1 or stride[1] > 1):
            orig_x, index = self.layer(self.Upsampling(), orig_x, index, layers, call,
                                       dict(size=stride))

        if in_filter < out_filter:
            orig_x, index = self.layer(Lambda, orig_x, index, layers, call,
                                       dict(function=self.pad(in_filter, out_filter)))
        elif in_filter > out_filter:
            orig_x, index = self.layer(Lambda, orig_x, index, layers, call,
                                       dict(function=self.slice(out_filter)))

        x, index = self.layer(Add, [orig_x, x], index, layers, call, dict())
        orig_x = x
        return x, orig_x, index

    def run_encoder(self, images, layers, wrn_size=16, call=False, BN=False, global_pool=False, combined_pool=False,
                    wd=0.0,
                    skip_connections=False, wrn_dense_layers=0, wrn_dense_units=1000):
        """Builds the WRN model.
        Build the Wide ResNet model from https://arxiv.org/abs/1605.07146.
        Args:
          images: Tensor of images that will be fed into the Wide ResNet Model.
          num_classes: Number of classed that the model needs to predict.
          wrn_size: Parameter that scales the number of filters in the Wide ResNet
            model.
        Returns:
          The logits of the Wide ResNet model.
        """
        index = 0
        kernel_size = wrn_size
        num_blocks_per_resnet = self.wrn_blocks_per_resnet
        filters = [min(kernel_size, 16)] + [int(kernel_size * 2 ** i) for i in range(self.wrn_levels)]
        strides = [self.ones()] + [self.stride() for i in range(self.wrn_levels)]  # stride for each resblock

        # Run the first conv
        with tf.name_scope('init'):
            x = images
            output_filters = filters[0]
            x, index = self.layer(self.Conv(), x, index, layers, call,
                                  dict(filters=output_filters, kernel_size=self.ks(),
                                       strides=1, padding='same',
                                       kernel_regularizer=l2(wd),
                                       bias_regularizer=l2(wd), ))

        for block_num in range(1, self.wrn_levels):
            with tf.name_scope('unit_{}_0'.format(block_num)):
                activate_before_residual = True if block_num == 1 else False
                x, index = self.residual_block(
                    x, index, layers,
                    filters[block_num - 1],
                    filters[block_num],
                    strides[block_num - 1],
                    activate_before_residual=activate_before_residual,
                    BN=BN, call=call, wd=wd)
            for i in range(1, num_blocks_per_resnet):
                with tf.name_scope('unit_{}_{}'.format(block_num, i)):
                    x, index = self.residual_block(
                        x, index, layers,
                        filters[block_num],
                        filters[block_num],
                        self.ones(),
                        activate_before_residual=False,
                        BN=BN, call=call, wd=wd)

        self.final_shape = K.int_shape(x)[1:]
        self.initial_shape = K.int_shape(images)[1:]
        with tf.name_scope('unit_last'):
            if BN:
                x, index = self.layer(BatchNormalization, x, index, layers, call,
                                      dict(gamma_initializer='glorot_uniform'))
            x, index = self.layer(self.get_activation(), x, index, layers, call, dict())
            if global_pool:
                x, index = self.layer(self.GlobalAveragePooling(), x, index, layers, call, dict())
            elif combined_pool:
                pooled, index = self.layer(self.GlobalAveragePooling(), x, index, layers, call, dict())
                x, index = self.layer(self.Conv(), x, index, layers, call, dict(filters=wrn_size, kernel_size=1,
                                                                                strides=1))
                x, index = self.layer(Flatten, x, index, layers, call, dict())
                x, index = self.layer(Concatenate, [x, pooled], index, layers, call, dict())
            else:
                x, index = self.layer(Flatten, x, index, layers, call, dict())

        for l in range(wrn_dense_layers):
            x, index = self.layer(Dense, x, index, layers, call, dict(units=wrn_dense_units))
            x, index = self.layer(self.get_activation(), x, index, layers, call, dict())

        return x

    def run_decoder(self, images, layers, wrn_size=16, call=False, wd=0.0, skip_connections=False, BN=False,
                    wrn_dense_layers=0, wrn_dense_units=1000,
                    input_shape=None, output_shapes=None):
        """Builds the WRN model.
        Build the Wide ResNet model from https://arxiv.org/abs/1605.07146.
        Args:
          images: Tensor of images that will be fed into the Wide ResNet Model.
          num_classes: Number of classed that the model needs to predict.
          wrn_size: Parameter that scales the number of filters in the Wide ResNet
            model.
        Returns:
          The logits of the Wide ResNet model.
        """
        index = 0
        kernel_size = wrn_size
        num_blocks_per_resnet = self.wrn_blocks_per_resnet
        filters = [min(kernel_size, 16)] + [int(kernel_size * 2 ** i) for i in range(self.wrn_levels - 1)]
        strides = [self.ones()] + [self.stride() for i in range(self.wrn_levels - 1)]  # stride for each resblock

        input_shape = tuple(input_shape)
        if type(output_shapes) is dict:
            input_shape = tuple(list(output_shapes.items())[0][1])
        total_strides = [int(np.prod([s[i] for s in strides[:-1]])) for i in range(len(input_shape) - 1)]
        final_shape = tuple([input_shape[i] // total_strides[i] for i in range(len(input_shape) - 1)]) + tuple(
            filters[-1:])
        with tf.name_scope('decode_unit_last'):
            x = images
            for l in range(wrn_dense_layers):
                x, index = self.layer(Dense, x, index, layers, call, dict(units=wrn_dense_units))
                if BN:
                    x, index = self.layer(BatchNormalization, x, index, layers, call,
                                          dict(gamma_initializer='glorot_uniform'))
                x, index = self.layer(self.get_activation(), x, index, layers, call, dict())

            x, index = self.layer(Dense, x, index, layers, call, dict(units=int(np.prod(final_shape)),
                                                                      kernel_regularizer=l2(wd),
                                                                      bias_regularizer=l2(wd), ))
            if BN:
                x, index = self.layer(BatchNormalization, x, index, layers, call,
                                      dict(gamma_initializer='glorot_uniform'))
            x, index = self.layer(self.get_activation(), x, index, layers, call, dict())
            x, index = self.layer(Reshape, x, index, layers, call, dict(target_shape=final_shape))

        first_x = x  # Res from the beginning
        orig_x = x  # Res from previous block

        for block_num in reversed(range(1, self.wrn_levels)):
            with tf.name_scope('decoder_unit_{}_0'.format(block_num)):
                activate_before_residual = True if block_num == 1 else False
                x, index = self.residual_block(
                    x, index, layers,
                    filters[block_num],
                    filters[block_num - 1],
                    strides[block_num - 1],
                    activate_before_residual=activate_before_residual,
                    BN=BN, call=call, transpose=True, wd=wd)
            for i in range(1, num_blocks_per_resnet):
                with tf.name_scope('decoder_unit_{}_{}'.format(block_num, i)):
                    x, index = self.residual_block(
                        x, index, layers,
                        filters[block_num - 1],
                        filters[block_num - 1],
                        self.ones(),
                        activate_before_residual=False,
                        BN=BN, call=call, transpose=True, wd=wd)

        for i in range(self.wrn_output_conv):
            cond = Dense(np.prod(self.input_shape))(images)
            cond = self.get_activation()()(cond)
            cond = Reshape(self.input_shape)(cond)
            x = Concatenate()([x, cond])
            x, index = self.residual_block(
                x, index, layers,
                filters[0] + self.input_shape[-1],
                filters[0],
                self.ones(),
                activate_before_residual=False,
                BN=BN, call=call, wd=wd)

        # Run the first conv
        if (output_shapes is None):
            with tf.name_scope('decoder_init'):
                output_filters = self.input_shape[-1]
                x, index = self.layer(self.Conv(), x, index, layers, call,
                                      dict(filters=output_filters, kernel_size=self.ks(),
                                           padding='same',
                                           strides=1))
            return x
        else:
            outputs = {}
            for output, oshape in output_shapes.items():
                output_filters = int(np.prod(oshape[len(self.input_shape) - 1:]))
                xout, index = self.layer(self.Conv(), x, index, layers, call,
                                         dict(filters=output_filters, kernel_size=self.ks(),
                                              padding='same',
                                              strides=1))
                xout, index = self.layer(Reshape, xout, index, layers, call, dict(target_shape=oshape))
                outputs[output] = xout
            return outputs

    def get_activation(self):
        # Get an activation function
        if self.wrn_actv is None or self.wrn_actv is None:
            return lambda: Activation('linear')
        if is_number(self.wrn_actv):
            return lambda: LeakyReLU(alpha=float(self.wrn_actv))
        return lambda: Activation(self.wrn_actv)

    def apply_encoder(self, x, layers=None, saved_layers=None):
        if not hasattr(self, 'encoder_layers') or not self.encoder_layers:
            self.encoder_layers = []
        layers = self.encoder_layers if saved_layers is None else saved_layers
        if self.wrn_rescale:
            x = Lambda(lambda a: a * 2. - 1.)(x)
        return self.run_encoder(x, layers, wrn_size=self.wrn_size,
                                call=False, BN=self.wrn_bn, global_pool=self.wrn_global_pool,
                                combined_pool=self.wrn_combined_pool, wd=self.wrn_weight_decay,
                                skip_connections=self.wrn_skip)

    def apply_decoder(self, x, output_shapes=None, layers=None, saved_layers=None):
        if not hasattr(self, 'decoder_layers') or not self.decoder_layers:
            self.decoder_layers = []
        layers = self.decoder_layers if saved_layers is None else saved_layers
        return self.run_decoder(x, layers, wrn_size=self.wrn_size, output_shapes=output_shapes,
                                input_shape=self.input_shape, BN=self.wrn_bn,
                                call=False, wd=self.wrn_weight_decay, skip_connections=self.wrn_skip)



