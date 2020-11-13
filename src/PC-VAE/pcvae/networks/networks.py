from tensorflow.keras.layers import Flatten, Dense, Lambda, Reshape, LSTM, GRU, Bidirectional, RepeatVector
import numpy as np
from .wrn import WRN, residual_decoder, residual_encoder
import tensorflow as tf

def get_rnn_encoder(layers=1, units=50, rnn_type='gru', **kwargs):
    rnn = GRU if rnn_type == 'gru' else LSTM
    def func(x):
        x = Lambda(lambda a: tf.squeeze(a, axis=1))(x)
        for l in range(layers):
            print(x.shape)
            x = Bidirectional(rnn(units, return_sequences=True))(x)
        print(x.shape)
        x = rnn(units)(x)
        print(x.shape)
        return x
    return func

def get_rnn_decoder(layers=1, units=50, rnn_type='gru', **kwargs):
    rnn = GRU if rnn_type == 'gru' else LSTM
    def func(x, output_shapes=None):
        example_shape = list(output_shapes.values())[0]
        x = RepeatVector(example_shape[-2])(x)
        for l in range(layers):
            print(x.shape)
            x = Bidirectional(rnn(units, return_sequences=True))(x)

        print(x.shape)
        x = rnn(units, return_sequences=True)(x)

        outputs = {}
        for output, oshape in output_shapes.items():
          out = Dense(oshape[-1])(x)
          print(x.shape, oshape, out.shape)
          outputs[output] = Lambda(lambda a: tf.expand_dims(a, axis=1))(out)
        return outputs
    return func

def get_default_network(layers=1, units=1000, activation='softplus', **kwargs):
    def func(x):
        x = Flatten()(x)
        for l in range(layers):
            x = Dense(units=units, activation=activation)(x)
        return x
    return func

def get_encoder_network(network, input_shape=None, **kwargs):
    if isinstance(network, tf.keras.layers.Layer):
        return network
    if network == 'wrn':
        return residual_encoder(input_shape=input_shape, **kwargs)
    if network == 'wrn_old':
        return WRN(input_shape=input_shape, **kwargs).apply_encoder
    if network == 'rnn':
        return get_rnn_encoder(**kwargs)
    return get_default_network(**kwargs)

def passthrough(x, output_shapes=None):
    outputs = {}
    ind = 0
    x = Flatten()(x)
    for output, oshape in output_shapes.items():
        nelems = oshape.num_elements()
        outputs[output] = Reshape(oshape)(Lambda(lambda a: a[:, ind:(ind + nelems)])(x))
        ind += nelems
    return outputs

def get_decoder_network(network, input_shape=None, encoded_size=None, **kwargs):
    if isinstance(network, tf.keras.layers.Layer):
        return network
    if network == 'wrn':
        return residual_decoder(encoded_size=encoded_size, input_shape=input_shape, **kwargs)
    if network == 'wrn_old':
        return WRN(input_shape=input_shape, **kwargs).apply_decoder
    if network == 'none':
        return lambda x: Flatten()(x)
    if network == 'passthrough':
        return passthrough
    if network == 'rnn':
        return get_rnn_decoder(**kwargs)
    return get_default_network(**kwargs)

def get_predictor_network(network, predictor_l2_weight=0., predictor_time_reducer='mean', predictor_conv_layers=0,
                          predictor_conv_args={}, predictor_dense_layers=0,
                          predictor_dense_units=100, predictor_dense_activation='elu', **kwargs):
    if isinstance(network, tf.keras.layers.Layer):
        return network
    def func(x, output_shapes=None):
        # Specializations for HMM
        if len(x.shape) == 3:
            for cl in range(predictor_conv_layers):
                x = tf.keras.layers.Conv1D(**predictor_conv_args)(x)

        if len(x.shape) == 3 and predictor_time_reducer:
            x = Lambda(lambda a: tf.reduce_mean(a, axis=1))(x)

        for dl in range(predictor_dense_layers):
            kernel_reg = tf.keras.regularizers.l2(predictor_l2_weight)
            bias_reg = tf.keras.regularizers.l2(predictor_l2_weight)
            x = Dense(predictor_dense_units, activation=predictor_dense_activation,
                                                kernel_regularizer=kernel_reg,
                                                bias_regularizer=bias_reg)(x)
            x = tf.keras.layers.BatchNormalization(gamma_initializer='glorot_uniform')(x)

        params, tensors = [], []
        for (param, shape) in output_shapes.items():
            params.append(param)
            kernel_reg = tf.keras.regularizers.l2(predictor_l2_weight)
            bias_reg = tf.keras.regularizers.l2(predictor_l2_weight)
            tensors.append(Reshape(shape)(Dense(shape.num_elements(),
                                                kernel_regularizer=kernel_reg,
                                                bias_regularizer=bias_reg)(x)))
        return dict(zip(params, tensors))
    return func

def get_bridge_network(network, **kwargs):
    if isinstance(network, tf.keras.layers.Layer):
        return network
    return get_default_network(**kwargs)