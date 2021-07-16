import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import prefer_static as ps
from ..networks.wrn import residual_block
import math


def _get_static_splits(splits):
    # Convert to a static value so that one could run tf.split on TPU.
    static_splits = tf.get_static_value(splits)
    return splits if static_splits is None else static_splits


class Blockwise(tfb.Bijector):
    """Bijector which applies a list of bijectors to blocks of a `Tensor`.
    More specifically, given [F_0, F_1, ... F_n] which are scalar or vector
    bijectors this bijector creates a transformation which operates on the vector
    [x_0, ... x_n] with the transformation [F_0(x_0), F_1(x_1) ..., F_n(x_n)]
    where x_0, ..., x_n are blocks (partitions) of the vector.
    Example Use:
    ```python
    blockwise = tfb.Blockwise(
        bijectors=[tfb.Exp(), tfb.Sigmoid()], block_sizes=[2, 1]
      )
    y = blockwise.forward(x)
    # Equivalent to:
    x_0, x_1 = tf.split(x, [2, 1], axis=-1)
    y_0 = tfb.Exp().forward(x_0)
    y_1 = tfb.Sigmoid().forward(x_1)
    y = tf.concat([y_0, y_1], axis=-1)
    ```
    """

    def __init__(self,
                 bijectors,
                 block_sizes=None,
                 validate_args=False,
                 maybe_changes_size=True,
                 name=None):
        """Creates the bijector.
        Args:
          bijectors: A non-empty list of bijectors.
          block_sizes: A 1-D integer `Tensor` with each element signifying the
            length of the block of the input vector to pass to the corresponding
            bijector. The length of `block_sizes` must be be equal to the length of
            `bijectors`. If left as None, a vector of 1's is used.
          validate_args: Python `bool` indicating whether arguments should be
            checked for correctness.
          maybe_changes_size: Python `bool` indicating that this bijector might
            change the event size. If this is known to be false and set
            appropriately, then this will lead to improved static shape inference
            when the block sizes are not statically known.
          name: Python `str`, name given to ops managed by this object. Default:
            E.g., `Blockwise([Exp(), Softplus()]).name ==
            'blockwise_of_exp_and_softplus'`.
        Raises:
          NotImplementedError: If there is a bijector with `event_ndims` > 1.
          ValueError: If `bijectors` list is empty.
          ValueError: If size of `block_sizes` does not equal to the length of
            bijectors or is not a vector.
        """
        parameters = dict(locals())
        if not name:
            name = 'blockwise_of_' + '_and_'.join([b.name for b in bijectors])
            name = name.replace('/', '')
        with tf.name_scope(name) as name:
            super(Blockwise, self).__init__(
                forward_min_event_ndims=1,
                validate_args=validate_args,
                parameters=parameters,
                name=name)

            self._bijectors = bijectors
            self._maybe_changes_size = maybe_changes_size

            if block_sizes is None:
                block_sizes = tf.ones(len(bijectors), dtype=tf.int32)
            self._block_sizes = tf.convert_to_tensor(
                block_sizes, name='block_sizes', dtype_hint=tf.int32)

            self._block_sizes = _validate_block_sizes(self._block_sizes, bijectors,
                                                      validate_args)

    @property
    def bijectors(self):
        return self._bijectors

    @property
    def block_sizes(self):
        return self._block_sizes

    def _output_block_sizes(self):
        return [
            b.forward_event_shape_tensor(bs[tf.newaxis])[0]
            for b, bs in zip(self.bijectors,
                             tf.unstack(self.block_sizes, num=len(self.bijectors)))
        ]

    def _forward_event_shape(self, input_shape):
        input_shape = tensorshape_util.with_rank_at_least(input_shape, 1)
        static_block_sizes = tf.get_static_value(self.block_sizes)
        if static_block_sizes is None:
            return tensorshape_util.concatenate(input_shape[:-1], [None])

        output_size = sum(
            b.forward_event_shape([bs])[0]
            for b, bs in zip(self.bijectors, static_block_sizes))

        return tensorshape_util.concatenate(input_shape[:-1], [output_size])

    def _forward_event_shape_tensor(self, input_shape):
        output_size = ps.reduce_sum(self.block_sizes)
        return ps.concat([input_shape[:-1], output_size[tf.newaxis]], -1)

    def _inverse_event_shape(self, output_shape):
        output_shape = tensorshape_util.with_rank_at_least(output_shape, 1)
        static_block_sizes = tf.get_static_value(self.block_sizes)
        if static_block_sizes is None:
            return tensorshape_util.concatenate(output_shape[:-1], [None])

        input_size = sum(static_block_sizes)

        return tensorshape_util.concatenate(output_shape[:-1], [input_size])

    def _inverse_event_shape_tensor(self, output_shape):
        input_size = ps.reduce_sum(self.block_sizes)
        return ps.concat([output_shape[:-1], input_size[tf.newaxis]], -1)

    def _forward(self, x):
        split_x = tf.split(x, _get_static_splits(self.block_sizes), axis=-1,
                           num=len(self.bijectors))
        split_y = [b.forward(x_) for b, x_ in zip(self.bijectors, split_x)]
        y = tf.concat(split_y, axis=-1)
        if not self._maybe_changes_size:
            tensorshape_util.set_shape(y, x.shape)
        return y

    def _inverse(self, y):
        split_y = tf.split(y, _get_static_splits(self.block_sizes),
                           axis=-1, num=len(self.bijectors))
        split_x = [b.inverse(y_) for b, y_ in zip(self.bijectors, split_y)]
        x = tf.concat(split_x, axis=-1)
        if not self._maybe_changes_size:
            tensorshape_util.set_shape(x, y.shape)
        return x

    def _forward_log_det_jacobian(self, x):
        split_x = tf.split(x, _get_static_splits(self.block_sizes), axis=-1,
                           num=len(self.bijectors))
        fldjs = [
            b.forward_log_det_jacobian(x_, event_ndims=1)
            for b, x_ in zip(self.bijectors, split_x)
        ]
        return sum(fldjs)

    def _inverse_log_det_jacobian(self, y):
        split_y = tf.split(y, _get_static_splits(self.self.block_sizes),
                           axis=-1, num=len(self.bijectors))
        ildjs = [
            b.inverse_log_det_jacobian(y_, event_ndims=1)
            for b, y_ in zip(self.bijectors, split_y)
        ]
        return sum(ildjs)


def _validate_block_sizes(block_sizes, bijectors, validate_args):
    """Helper to validate block sizes."""
    block_sizes_shape = block_sizes.shape
    if tensorshape_util.is_fully_defined(block_sizes_shape):
        if (tensorshape_util.rank(block_sizes_shape) != 1 or
                (tensorshape_util.num_elements(block_sizes_shape) != len(bijectors))):
            raise ValueError(
                '`block_sizes` must be `None`, or a vector of the same length as '
                '`bijectors`. Got a `Tensor` with shape {} and `bijectors` of '
                'length {}'.format(block_sizes_shape, len(bijectors)))
        return block_sizes
    elif validate_args:
        message = ('`block_sizes` must be `None`, or a vector of the same length '
                   'as `bijectors`.')
        with tf.control_dependencies([
            assert_util.assert_equal(
                tf.size(block_sizes), len(bijectors), message=message),
            assert_util.assert_equal(tf.rank(block_sizes), 1)
        ]):
            return tf.identity(block_sizes)
    else:
        return block_sizes


def _replace_event_shape_in_shape_tensor(
        input_shape, event_shape_in, event_shape_out, validate_args):
    """Replaces the rightmost dims in a `Tensor` representing a shape.
    Args:
      input_shape: a rank-1 `Tensor` of integers
      event_shape_in: the event shape expected to be present in rightmost dims
        of `shape_in`.
      event_shape_out: the event shape with which to replace `event_shape_in` in
        the rightmost dims of `input_shape`.
      validate_args: Python `bool` indicating whether arguments should
        be checked for correctness.
    Returns:
      output_shape: A rank-1 integer `Tensor` with the same contents as
        `input_shape` except for the event dims, which are replaced with
        `event_shape_out`.
    """
    output_tensorshape, is_validated = _replace_event_shape_in_tensorshape(
        tensorshape_util.constant_value_as_shape(input_shape),
        event_shape_in,
        event_shape_out)

    # TODO(b/124240153): Remove map(tf.identity, deps) once tf.function
    # correctly supports control_dependencies.
    validation_dependencies = (
        map(tf.identity, (event_shape_in, event_shape_out))
        if validate_args else ())

    if (tensorshape_util.is_fully_defined(output_tensorshape) and
            (is_validated or not validate_args)):
        with tf.control_dependencies(validation_dependencies):
            output_shape = tf.convert_to_tensor(
                tensorshape_util.as_list(output_tensorshape), name='output_shape',
                dtype_hint=tf.int32)
        return output_shape, output_tensorshape

    with tf.control_dependencies(validation_dependencies):
        event_shape_in_ndims = (
            tf.size(event_shape_in)
            if tensorshape_util.num_elements(event_shape_in.shape) is None else
            tensorshape_util.num_elements(event_shape_in.shape))
        input_non_event_shape, input_event_shape = tf.split(
            input_shape, num_or_size_splits=[-1, event_shape_in_ndims])

    additional_assertions = []
    if is_validated:
        pass
    elif validate_args:
        # Check that `input_event_shape` and `event_shape_in` are compatible in the
        # sense that they have equal entries in any position that isn't a `-1` in
        # `event_shape_in`. Note that our validations at construction time ensure
        # there is at most one such entry in `event_shape_in`.
        mask = event_shape_in >= 0
        explicit_input_event_shape = tf.boolean_mask(input_event_shape, mask=mask)
        explicit_event_shape_in = tf.boolean_mask(event_shape_in, mask=mask)
        additional_assertions.append(
            assert_util.assert_equal(
                explicit_input_event_shape,
                explicit_event_shape_in,
                message='Input `event_shape` does not match `event_shape_in`.'))
        # We don't explicitly additionally verify
        # `tf.size(input_shape) > tf.size(event_shape_in)` since `tf.split`
        # already makes this assertion.

    with tf.control_dependencies(additional_assertions):
        output_shape = tf.concat([input_non_event_shape, event_shape_out], axis=0,
                                 name='output_shape')

    return output_shape, output_tensorshape


def _replace_event_shape_in_tensorshape(
        input_tensorshape, event_shape_in, event_shape_out):
    """Replaces the event shape dims of a `TensorShape`.
    Args:
      input_tensorshape: a `TensorShape` instance in which to attempt replacing
        event shape.
      event_shape_in: `Tensor` shape representing the event shape expected to
        be present in (rightmost dims of) `tensorshape_in`. Must be compatible
        with the rightmost dims of `tensorshape_in`.
      event_shape_out: `Tensor` shape representing the new event shape, i.e.,
        the replacement of `event_shape_in`,
    Returns:
      output_tensorshape: `TensorShape` with the rightmost `event_shape_in`
        replaced by `event_shape_out`. Might be partially defined, i.e.,
        `TensorShape(None)`.
      is_validated: Python `bool` indicating static validation happened.
    Raises:
      ValueError: if we can determine the event shape portion of
        `tensorshape_in` as well as `event_shape_in` both statically, and they
        are not compatible. "Compatible" here means that they are identical on
        any dims that are not -1 in `event_shape_in`.
    """
    event_shape_in_ndims = tensorshape_util.num_elements(event_shape_in.shape)
    if tensorshape_util.rank(
            input_tensorshape) is None or event_shape_in_ndims is None:
        return tf.TensorShape(None), False  # Not is_validated.

    input_non_event_ndims = tensorshape_util.rank(
        input_tensorshape) - event_shape_in_ndims
    if input_non_event_ndims < 0:
        raise ValueError(
            'Input has lower rank ({}) than `event_shape_ndims` ({}).'.format(
                tensorshape_util.rank(input_tensorshape), event_shape_in_ndims))

    input_non_event_tensorshape = input_tensorshape[:input_non_event_ndims]
    input_event_tensorshape = input_tensorshape[input_non_event_ndims:]

    # Check that `input_event_shape_` and `event_shape_in` are compatible in the
    # sense that they have equal entries in any position that isn't a `-1` in
    # `event_shape_in`. Note that our validations at construction time ensure
    # there is at most one such entry in `event_shape_in`.
    event_shape_in_ = tf.get_static_value(event_shape_in)
    is_validated = (
            tensorshape_util.is_fully_defined(input_event_tensorshape) and
            event_shape_in_ is not None)
    if is_validated:
        input_event_shape_ = np.int32(input_event_tensorshape)
        mask = event_shape_in_ >= 0
        explicit_input_event_shape_ = input_event_shape_[mask]
        explicit_event_shape_in_ = event_shape_in_[mask]
        if not np.all(explicit_input_event_shape_ == explicit_event_shape_in_):
            raise ValueError(
                'Input `event_shape` does not match `event_shape_in` '
                '({} vs {}).'.format(input_event_shape_, event_shape_in_))

    event_tensorshape_out = tensorshape_util.constant_value_as_shape(
        event_shape_out)
    if tensorshape_util.rank(event_tensorshape_out) is None:
        output_tensorshape = tf.TensorShape(None)
    else:
        output_tensorshape = tensorshape_util.concatenate(
            input_non_event_tensorshape, event_tensorshape_out)

    return output_tensorshape, is_validated


def _maybe_check_valid_shape(shape, validate_args):
    """Check that a shape Tensor is int-type and otherwise sane."""
    if not dtype_util.is_integer(shape.dtype):
        raise TypeError('`{}` dtype (`{}`) should be `int`-like.'.format(
            shape, dtype_util.name(shape.dtype)))

    assertions = []

    message = '`{}` rank should be <= 1.'
    if tensorshape_util.rank(shape.shape) is not None:
        if tensorshape_util.rank(shape.shape) > 1:
            raise ValueError(message.format(shape))
    elif validate_args:
        assertions.append(assert_util.assert_less(
            tf.rank(shape), 2, message=message.format(shape)))

    shape_ = tf.get_static_value(shape)

    message = '`{}` elements must have at most one `-1`.'
    if shape_ is not None:
        if sum(shape_ == -1) > 1:
            raise ValueError(message.format(shape))
    elif validate_args:
        assertions.append(
            assert_util.assert_less(
                tf.reduce_sum(tf.cast(tf.equal(shape, -1), tf.int32)),
                2,
                message=message.format(shape)))

    message = '`{}` elements must be either positive integers or `-1`.'
    if shape_ is not None:
        if np.any(shape_ < -1):
            raise ValueError(message.format(shape))
    elif validate_args:
        assertions.append(assert_util.assert_greater(
            shape, -2, message=message.format(shape)))

    return assertions


def _rank_from_shape(x):
    """Returns the rank implied by this shape."""
    if not hasattr(x, 'shape'):
        return tf.TensorShape(x).rank
    else:
        # If the input is a Tensor, we can't make a `TensorShape` out of it
        # directly:
        # - In graph mode, `TensorShape` complains that it can't iterate over a
        #   Tensor.
        # - In eager mode, the underlying `Dimension` complains that a scalar
        #   integer Tensor is actually an ambiguous dimension, because it !=
        #   int(it).
        # However, the (static) size of `x` is also the rank of the Tensor
        # it represents, which is what we want.
        return tf.TensorShape(x.shape).num_elements()


class Squeeze(tfp.bijectors.Reshape):
    def __init__(self, event_shape_out, event_shape_in=(-1,), **kwargs):
        self._shape_out_tuple = event_shape_out
        self._shape_in_tuple = event_shape_in
        super(Squeeze, self).__init__(event_shape_out, event_shape_in, **kwargs)

    def interleave(self, a, b, axis=0, output_shape=None):
        a_shape, b_shape = tuple([(-1 if d is None else d) for d in a.shape]), tuple(b.shape)
        out = tf.concat([tf.expand_dims(a, axis + 1), tf.expand_dims(b, axis + 1)], axis=axis + 1)
        return tf.reshape(out, a_shape[:axis] + (a_shape[axis] + b_shape[axis],) + a_shape[(axis + 1):])

    def _forward(self, x):
        output_shape, output_tensorshape = _replace_event_shape_in_shape_tensor(
            tf.shape(x), self._event_shape_in, self._event_shape_out,
            self.validate_args)
        y0 = x[:, ::2, ::2, :]
        y1 = x[:, 1::2, ::2, :]
        y2 = x[:, ::2, 1::2, :]
        y3 = x[:, 1::2, 1::2, :]
        y = tf.concat([y0, y1, y2, y3], axis=-1)
        tensorshape_util.set_shape(y, output_tensorshape)
        return y

    def _inverse(self, y):
        input_shape, input_tensorshape = _replace_event_shape_in_shape_tensor(
            tf.shape(y), self._event_shape_out, self._event_shape_out,
            self.validate_args)
        y = tf.reshape(y, input_shape)
        output_shape, output_tensorshape = _replace_event_shape_in_shape_tensor(
            tf.shape(y), self._event_shape_out, self._event_shape_in,
            self.validate_args)
        x0, x1, x2, x3 = tf.split(y, 4, axis=-1)
        xa = self.interleave(x0, x1, axis=1)
        xb = self.interleave(x2, x3, axis=1)
        #xa = tf.reshape(xa, (-1,) + self._shape_in_tuple[:1] + self._shape_out_tuple[1:])
        #xb = tf.reshape(xb, (-1,) + self._shape_in_tuple[:1] + self._shape_out_tuple[1:])
        x = self.interleave(xa, xb, axis=2)
        #x = tf.reshape(x, (-1,) + self._shape_in_tuple)
        tensorshape_util.set_shape(x, output_tensorshape)
        return x


class SqueezeOld(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='Squeeze'):
        super(Squeeze, self).__init__(
            is_constant_jacobian=True,
            validate_args=validate_args,
            forward_min_event_ndims=1,
            name=name)

    @property
    def permutation(self):
        return self._permutation

    @property
    def axis(self):
        return self._axis

    def _forward(self, x):
        print(('f', x.shape))
        y0 = x[:, ::2, ::2, :]
        y1 = x[:, 1::2, ::2, :]
        y2 = x[:, ::2, 1::2, :]
        y3 = x[:, 1::2, 1::2, :]
        return tf.concat([y0, y1, y2, y3], axis=-1)

    def interleave(self, a, b, axis=0):
        a_shape, b_shape = tuple([(-1 if d is None else d) for d in a.shape]), tuple(b.shape)
        out = tf.concat([tf.expand_dims(a, axis + 1), tf.expand_dims(b, axis + 1)], axis=axis + 1)
        return tf.reshape(out, a_shape[:axis] + (a_shape[axis] + b_shape[axis],) + a_shape[(axis + 1):])

    def _inverse(self, y):
        print('r' + str(y.shape))
        x0, x1, x2, x3 = tf.split(y, 4, axis=-1)
        xa = self.interleave(x0, x1, axis=1)
        xb = self.interleave(x2, x3, axis=1)
        return self.interleave(xa, xb, axis=2)

    def _inverse_log_det_jacobian(self, y):
        # is_constant_jacobian = True for this bijector, hence the
        # `log_det_jacobian` need only be specified for a single input, as this will
        # be tiled to match `event_ndims`.
        return tf.constant(0., dtype=tf.float32)

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., dtype=tf.float32)


class LogScale(tf.keras.layers.Layer):
    def __init__(self, logscale_factor=3, name=''):
        super(LogScale, self).__init__(name=name)
        with tf.name_scope(name or 'log_scale'):
            self.logscale_factor = logscale_factor
            w_init = tf.zeros_initializer()
            self.w = tf.Variable(
                initial_value=w_init(shape=(), dtype="float32"),
                trainable=True, name='log_scale_w'
            )

    def call(self, inputs):
        return tf.exp(self.logscale_factor * self.w) * inputs

def GLOWNet(input_shape, output_units, width=256, kernel_size=(3,3), name='', **kwargs):
    inputs = tf.keras.layers.Input(input_shape)
    output = tf.keras.layers.Conv2D(filters=width, kernel_size=(3, 3), padding='same', activation='relu', name='%s_conv_0' % name)(inputs)
    output = tf.keras.layers.Conv2D(filters=width, kernel_size=(1, 1), padding='same', activation='relu', name='%s_conv_1' % name)(output)
    output = tf.keras.layers.Conv2D(filters=2 * output_units, kernel_size=kernel_size, padding='same',
                                    kernel_initializer='zeros', name='%s_conv_2' % name)(output)
    output = LogScale(name='%s_log_scale' % name)(output)
    output_loc, output_log_scale = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, axis=-1))(output)
    output_log_scale = tf.keras.layers.Lambda(
        lambda x: tf.math.log(0.5 + tf.math.sigmoid(x)))(output_log_scale)
    model = tf.keras.Model(inputs=inputs, outputs=[output_loc, output_log_scale])
    return model


def GLOWResNet(input_shape, output_units, **kwargs):
    inputs = tf.keras.layers.Input(input_shape)
    output = residual_block(inputs, filters=64, name='glow_1')
    output_loc = residual_block(output, filters=output_units, name='glow_loc')
    output_log_scale = residual_block(output, filters=output_units, name='glow_scale')
    output_log_scale = Lambda(lambda x: tf.math.log(tf.nn.softplus(x)))(output_log_scale)
    model = tf.keras.Model(inputs=inputs, outputs=[output_loc, output_log_scale])
    return model


def trainable_lu_factorization(
        event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
    with tf.name_scope(name or 'trainable_lu_factorization'):
        event_size = tf.convert_to_tensor(
            event_size, dtype_hint=tf.int32, name='event_size')
        batch_shape = tf.convert_to_tensor(
            batch_shape, dtype_hint=event_size.dtype, name='batch_shape')
        random_matrix = tf.random.uniform(
            shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
            dtype=dtype,
            seed=seed)
        random_orthonormal = tf.linalg.qr(random_matrix)[0]
        lower_upper, permutation = tf.linalg.lu(random_orthonormal)
        lower_upper = tf.Variable(
            initial_value=lower_upper,
            trainable=True,
            name='lower_upper')
        # Initialize a non-trainable variable for the permutation indices so
        # that its value isn't re-sampled from run-to-run.
        permutation = tf.Variable(
            initial_value=permutation,
            trainable=False,
            name='permutation')
        return lower_upper, permutation


class GLOW(tfp.layers.DistributionLambda):
    def flow_step(self, input_shape, name=''):
        norm = tfb.BatchNormalization(name='%s_bn' % name)  # batchnorm_layer=tf.keras.layers.BatchNormalization(gamma_initializer=tf.keras.initializers.RandomUniform(0.5, 1.5), gamma_constraint=lambda g: tf.nn.relu(g) + 1e-6))

        network = GLOWNet(input_shape[:-1] + (math.ceil(input_shape[-1] / 2),), input_shape[-1] // 2, width=self.glow_width, kernel_size=self.glow_kernel, name='%s_net' % name)
        self.resnets.append(network)

        def call_net(x, depth, **kwargs):
            return network(x)

        if self.glow_permute == 'conv':
            lower_upper, permutation = trainable_lu_factorization(input_shape[-1], name='%s_scalemvlu_vars' % name)
            self.extra_vars.append(lower_upper)
            conv_1_1 = tfb.Invert(tfb.ScaleMatvecLU(lower_upper, permutation, validate_args=True, name='%s_scalemvlu' % name))
        elif self.glow_permute == 'none':
            conv_1_1 = tfb.Identity()
        else:
            permutation = np.arange(input_shape[-1])[::-1]
            if self.glow_permute == 'random':
                np.random.shuffle(permutation)
            conv_1_1 = tfb.Permute(permutation)

        affine_coupling = tfb.RealNVP(fraction_masked=0.5, shift_and_log_scale_fn=call_net)
        return tfb.Chain([norm, conv_1_1, tfb.Invert(affine_coupling)])

    def flow_level(self, input_shape, levels, base_level=True):
        bijector_chain = []

        # Glow blocks at the individual pixel level
        if self.level_0_blocks and base_level:
            bijector_chain.append(tfb.Chain([self.flow_step(input_shape, name='glow_base_%d' % i) for i in range(int(self.level_0_blocks))]))

        original_input_shape = input_shape
        recurse_shape = (input_shape[0] // 2, input_shape[1] // 2, input_shape[2] * 2)
        input_shape = (input_shape[0] // 2, input_shape[1] // 2, input_shape[2] * 4)

        steps = tfb.Chain([self.flow_step(input_shape, name='glow_l%d_%d' % (levels, i)) for i in range(self.glow_blocks)])
        if self.glow_squeeze == 'squeeze':
            bijector_chain.append(tfb.Invert(Squeeze(input_shape, original_input_shape)))
        else:
            bijector_chain.append(tfb.Invert(tfb.Reshape(input_shape, original_input_shape)))

        bijector_chain.append(steps)

        if levels > 0:
            if self.blockwise:
                recurse_level = self.flow_level(recurse_shape, levels - 1, False)
                bijector_chain.append(
                    Blockwise([recurse_level, tfb.Identity()], block_sizes=[recurse_shape[-1], recurse_shape[-1]]))
            else:
                recurse_level = self.flow_level(input_shape, levels - 1, False)
                bijector_chain.append(recurse_level)

        if self.glow_squeeze == 'squeeze':
            bijector_chain.append(Squeeze(input_shape, original_input_shape))
        else:
            bijector_chain.append(tfb.Reshape(input_shape, original_input_shape))

        if base_level and self.glow_constrain:
            bijector_chain.append(tfb.Invert(tfb.Tanh()))
        return tfb.Chain(bijector_chain)

    def __init__(self, input_shape=(32, 32, 3), glow_width=256, glow_levels=0, glow_blocks=8,
                 glow_permute='conv', glow_squeeze='squeeze', glow_kernel=(3,3), glow_constrain=True,
                 level_0_blocks=False, blockwise=False, **kwargs):
        self.resnets = []
        self.extra_vars = []
        self.glow_levels = glow_levels
        self.glow_width = glow_width
        self.glow_blocks = glow_blocks
        self.glow_permute = glow_permute
        self.glow_squeeze = glow_squeeze
        self.glow_kernel = glow_kernel
        self.level_0_blocks = level_0_blocks
        self.blockwise = blockwise
        self.glow_constrain = glow_constrain
        self.bijector = self.flow_level(input_shape, glow_levels)
        super(GLOW, self).__init__(self._transform, **kwargs)

    def _transform(self, inputs):
        return tfp.distributions.TransformedDistribution(inputs, (self.bijector))
