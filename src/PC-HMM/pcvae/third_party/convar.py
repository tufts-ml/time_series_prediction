import  tensorflow as tf
import numpy as np
from tensorflow_probability.python.internal import prefer_static as ps
import six

class ConvolutionalAutoregressiveNetwork(tf.keras.layers.Layer):
  r"""Masked Autoencoder for Distribution Estimation [Germain et al. (2015)][1].
  A `AutoregressiveNetwork` takes as input a Tensor of shape `[..., event_size]`
  and returns a Tensor of shape `[..., event_size, params]`.
  The output satisfies the autoregressive property.  That is, the layer is
  configured with some permutation `ord` of `{0, ..., event_size-1}` (i.e., an
  ordering of the input dimensions), and the output `output[batch_idx, i, ...]`
  for input dimension `i` depends only on inputs `x[batch_idx, j]` where
  `ord(j) < ord(i)`.  The autoregressive property allows us to use
  `output[batch_idx, i]` to parameterize conditional distributions:
    `p(x[batch_idx, i] | x[batch_idx, j] for ord(j) < ord(i))`
  which give us a tractable distribution over input `x[batch_idx]`:
    `p(x[batch_idx]) = prod_i p(x[batch_idx, ord(i)] | x[batch_idx, ord(0:i)])`
  For example, when `params` is 2, the output of the layer can parameterize
  the location and log-scale of an autoregressive Gaussian distribution.
  #### Example
  ```python
  # Generate data -- as in Figure 1 in [Papamakarios et al. (2017)][2]).
  n = 2000
  x2 = np.random.randn(n).astype(dtype=np.float32) * 2.
  x1 = np.random.randn(n).astype(dtype=np.float32) + (x2 * x2 / 4.)
  data = np.stack([x1, x2], axis=-1)
  # Density estimation with MADE.
  made = tfb.AutoregressiveNetwork(params=2, hidden_units=[10, 10])
  distribution = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(made),
      event_shape=[2])
  # Construct and fit model.
  x_ = tfkl.Input(shape=(2,), dtype=tf.float32)
  log_prob_ = distribution.log_prob(x_)
  model = tfk.Model(x_, log_prob_)
  model.compile(optimizer=tf.optimizers.Adam(),
                loss=lambda _, log_prob: -log_prob)
  batch_size = 25
  model.fit(x=data,
            y=np.zeros((n, 0), dtype=np.float32),
            batch_size=batch_size,
            epochs=1,
            steps_per_epoch=1,  # Usually `n // batch_size`.
            shuffle=True,
            verbose=True)
  # Use the fitted distribution.
  distribution.sample((3, 1))
  distribution.log_prob(np.ones((3, 2), dtype=np.float32))
  ```
  #### Examples: Handling Rank-2+ Tensors
  `AutoregressiveNetwork` can be used as a building block to achieve different
  autoregressive structures over rank-2+ tensors.  For example, suppose we want
  to build an autoregressive distribution over images with dimension `[weight,
  height, channels]` with `channels = 3`:
   1. We can parameterize a 'fully autoregressive' distribution, with
      cross-channel and within-pixel autoregressivity:
      ```
          r0    g0   b0     r0    g0   b0       r0   g0    b0
          ^   ^      ^         ^   ^   ^         ^      ^   ^
          |  /  ____/           \  |  /           \____  \  |
          | /__/                 \ | /                 \__\ |
          r1    g1   b1     r1 <- g1   b1       r1   g1 <- b1
                                               ^          |
                                                \_________/
      ```
      as:
      ```python
      # Generate random images for training data.
      images = np.random.uniform(size=(100, 8, 8, 3)).astype(np.float32)
      n, width, height, channels = images.shape
      # Reshape images to achieve desired autoregressivity.
      event_shape = [height * width * channels]
      reshaped_images = tf.reshape(images, [n, event_shape])
      # Density estimation with MADE.
      made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape,
                                       hidden_units=[20, 20], activation='relu')
      distribution = tfd.TransformedDistribution(
          distribution=tfd.Normal(loc=0., scale=1.),
          bijector=tfb.MaskedAutoregressiveFlow(made),
          event_shape=event_shape)
      # Construct and fit model.
      x_ = tfkl.Input(shape=event_shape, dtype=tf.float32)
      log_prob_ = distribution.log_prob(x_)
      model = tfk.Model(x_, log_prob_)
      model.compile(optimizer=tf.optimizers.Adam(),
                    loss=lambda _, log_prob: -log_prob)
      batch_size = 10
      model.fit(x=data,
                y=np.zeros((n, 0), dtype=np.float32),
                batch_size=batch_size,
                epochs=10,
                steps_per_epoch=n // batch_size,
                shuffle=True,
                verbose=True)
      # Use the fitted distribution.
      distribution.sample((3, 1))
      distribution.log_prob(np.ones((5, 8, 8, 3), dtype=np.float32))
      ```
   2. We can parameterize a distribution with neither cross-channel nor
      within-pixel autoregressivity:
      ```
          r0    g0   b0
          ^     ^    ^
          |     |    |
          |     |    |
          r1    g1   b1
      ```
      as:
      ```python
      # Generate fake images.
      images = np.random.choice([0, 1], size=(100, 8, 8, 3))
      n, width, height, channels = images.shape
      # Reshape images to achieve desired autoregressivity.
      reshaped_images = np.transpose(
          np.reshape(images, [n, width * height, channels]),
          axes=[0, 2, 1])
      made = tfb.AutoregressiveNetwork(params=1, event_shape=[width * height],
                                       hidden_units=[20, 20], activation='relu')
      # Density estimation with MADE.
      #
      # NOTE: Parameterize an autoregressive distribution over an event_shape of
      # [channels, width * height], with univariate Bernoulli conditional
      # distributions.
      distribution = tfd.Autoregressive(
          lambda x: tfd.Independent(
              tfd.Bernoulli(logits=tf.unstack(made(x), axis=-1)[0],
                            dtype=tf.float32),
              reinterpreted_batch_ndims=2),
          sample0=tf.zeros([channels, width * height], dtype=tf.float32))
      # Construct and fit model.
      x_ = tfkl.Input(shape=(channels, width * height), dtype=tf.float32)
      log_prob_ = distribution.log_prob(x_)
      model = tfk.Model(x_, log_prob_)
      model.compile(optimizer=tf.optimizers.Adam(),
                    loss=lambda _, log_prob: -log_prob)
      batch_size = 10
      model.fit(x=reshaped_images,
                y=np.zeros((n, 0), dtype=np.float32),
                batch_size=batch_size,
                epochs=10,
                steps_per_epoch=n // batch_size,
                shuffle=True,
                verbose=True)
      distribution.sample(7)
      distribution.log_prob(np.ones((4, 8, 8, 3), dtype=np.float32))
      ```
      Note that one set of weights is shared for the mapping for each channel
      from image to distribution parameters -- i.e., the mapping
      `layer(reshaped_images[..., channel, :])`, where `channel` is 0, 1, or 2.
      To use separate weights for each channel, we could construct an
      `AutoregressiveNetwork` and `TransformedDistribution` for each channel,
      and combine them with a `tfd.Blockwise` distribution.
  #### References
  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
  [2]: George Papamakarios, Theo Pavlakou, Iain Murray, Masked Autoregressive
       Flow for Density Estimation.  In _Neural Information Processing Systems_,
       2017. https://arxiv.org/abs/1705.07057
  """

  def __init__(self,
               params,
               event_shape=None,
               kernel_shape=None,
               conditional=False,
               conditional_event_shape=None,
               conditional_input_layers='all_layers',
               hidden_units=None,
               input_order='left-to-right',
               hidden_degrees='equal',
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               validate_args=False,
               **kwargs):
    """Constructs the MADE layer.
    Arguments:
      params: Python integer specifying the number of parameters to output
        per input.
      event_shape: Python `list`-like of positive integers (or a single int),
        specifying the shape of the input to this layer, which is also the
        event_shape of the distribution parameterized by this layer.  Currently
        only rank-1 shapes are supported.  That is, event_shape must be a single
        integer.  If not specified, the event shape is inferred when this layer
        is first called or built.
      conditional: Python boolean describing whether to add conditional inputs.
      conditional_event_shape: Python `list`-like of positive integers (or a
        single int), specifying the shape of the conditional input to this layer
        (without the batch dimensions). This must be specified if `conditional`
        is `True`.
      conditional_input_layers: Python `str` describing how to add conditional
        parameters to the autoregressive network. When "all_layers" the
        conditional input will be combined with the network at every layer,
        whilst "first_layer" combines the conditional input only at the first
        layer which is then passed through the network
        autoregressively. Default: 'all_layers'.
      hidden_units: Python `list`-like of non-negative integers, specifying
        the number of units in each hidden layer.
      input_order: Order of degrees to the input units: 'random',
        'left-to-right', 'right-to-left', or an array of an explicit order. For
        example, 'left-to-right' builds an autoregressive model:
        `p(x) = p(x1) p(x2 | x1) ... p(xD | x<D)`.  Default: 'left-to-right'.
      hidden_degrees: Method for assigning degrees to the hidden units:
        'equal', 'random'.  If 'equal', hidden units in each layer are allocated
        equally (up to a remainder term) to each degree.  Default: 'equal'.
      activation: An activation function.  See `tf.keras.layers.Dense`. Default:
        `None`.
      use_bias: Whether or not the dense layers constructed in this layer
        should have a bias term.  See `tf.keras.layers.Dense`.  Default: `True`.
      kernel_initializer: Initializer for the `Dense` kernel weight
        matrices.  Default: 'glorot_uniform'.
      bias_initializer: Initializer for the `Dense` bias vectors. Default:
        'zeros'.
      kernel_regularizer: Regularizer function applied to the `Dense` kernel
        weight matrices.  Default: None.
      bias_regularizer: Regularizer function applied to the `Dense` bias
        weight vectors.  Default: None.
      kernel_constraint: Constraint function applied to the `Dense` kernel
        weight matrices.  Default: None.
      bias_constraint: Constraint function applied to the `Dense` bias
        weight vectors.  Default: None.
      validate_args: Python `bool`, default `False`. When `True`, layer
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      **kwargs: Additional keyword arguments passed to this layer (but not to
        the `tf.keras.layer.Dense` layers constructed by this layer).
    """
    super(ConvolutionalAutoregressiveNetwork, self).__init__(**kwargs)

    self._params = params
    self._event_shape = _list(event_shape) if event_shape is not None else None
    self._kernel_shape = kernel_shape
    self._conditional = conditional
    self._conditional_event_shape = (
        _list(conditional_event_shape)
        if conditional_event_shape is not None else None)
    self._conditional_layers = conditional_input_layers
    self._hidden_units = hidden_units if hidden_units is not None else []
    self._input_order_param = input_order
    self._hidden_degrees = hidden_degrees
    self._activation = activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = bias_constraint
    self._validate_args = validate_args
    self._kwargs = kwargs

    if self._event_shape is not None:
      self._event_size = self._event_shape[-1]
      self._event_ndims = len(self._event_shape)

    # To be built in `build`.
    self._input_order = None
    self._masks = None
    self._network = None

  def build(self, input_shape):
    """See tfkl.Layer.build."""
    if self._event_shape is None:
      # `event_shape` wasn't specied at __init__, so infer from `input_shape`.
      self._event_shape = input_shape[-3:]
      self._event_size = self._event_shape[-1]
      self._event_ndims = len(self._event_shape)
      # Should we throw if input_shape has rank > 2?

    if input_shape[-1] != self._event_shape[-1]:
      raise ValueError('Invalid final dimension of `input_shape`. '
                       'Expected `{!r}`, but got `{!r}`'.format(
                           self._event_shape[-1], input_shape[-1]))

    # Construct the masks.
    self._input_order = _create_input_order(
        self._event_size,
        self._input_order_param,
    )
    self._masks = _make_convolutional_autoregressive_masks(
        params=self._params,
        event_size=tuple(self._kernel_shape) + tuple(self._event_shape[-1:]),
        hidden_units=self._hidden_units,
        input_order=self._input_order,
        hidden_degrees=self._hidden_degrees,
    )

    outputs = [tf.keras.Input(self._event_shape, dtype=self.dtype)]
    inputs = outputs[0]

    # Input-to-hidden, hidden-to-hidden, and hidden-to-output layers:
    #  [..., self._event_size] -> [..., self._hidden_units[0]].
    #  [..., self._hidden_units[k-1]] -> [..., self._hidden_units[k]].
    #  [..., self._hidden_units[-1]] -> [..., event_size * self._params].
    layer_output_sizes = self._hidden_units + [self._event_size * self._params]
    for k in range(len(self._masks)):
      autoregressive_output = tf.keras.layers.Conv2D(
          layer_output_sizes[k],
          self._kernel_shape,
          padding='same',
          activation=None,
          use_bias=self._use_bias,
          kernel_initializer=_make_masked_initializer(
              self._masks[k], self._kernel_initializer),
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          kernel_constraint=_make_masked_constraint(
              self._masks[k], self._kernel_constraint),
          bias_constraint=self._bias_constraint,
          dtype=self.dtype)(outputs[-1])

      outputs.append(autoregressive_output)
      if k + 1 < len(self._masks):
        outputs.append(
            tf.keras.layers.Activation(self._activation)
            (outputs[-1]))
    self._network = tf.keras.models.Model(
        inputs=inputs,
        outputs=outputs[-1])
    # Record that the layer has been built.
    super(ConvolutionalAutoregressiveNetwork, self).build(input_shape)

  def call(self, x, conditional_input=None):
    """Transforms the inputs and returns the outputs.
    Suppose `x` has shape `batch_shape + event_shape` and `conditional_input`
    has shape `conditional_batch_shape + conditional_event_shape`. Then, the
    output shape is:
    `broadcast(batch_shape, conditional_batch_shape) + event_shape + [params]`.
    Also see `tfkl.Layer.call` for some generic discussion about Layer calling.
    Args:
      x: A `Tensor`. Primary input to the layer.
      conditional_input: A `Tensor. Conditional input to the layer. This is
        required iff the layer is conditional.
    Returns:
      y: A `Tensor`. The output of the layer. Note that the leading dimensions
         follow broadcasting rules described above.
    """
    with tf.name_scope(self.name or 'AutoregressiveNetwork_call'):
      x = tf.convert_to_tensor(x, dtype=self.dtype, name='x')
      # TODO(b/67594795): Better support for dynamic shapes.
      input_shape = ps.shape(x)
      output_shape = input_shape
      return tf.reshape(self._network(x),
                        tf.concat([output_shape, [self._params]], axis=0))

  def compute_output_shape(self, input_shape):
    """See tfkl.Layer.compute_output_shape."""
    return input_shape + (self._params,)

  @property
  def event_shape(self):
    return self._event_shape

  @property
  def params(self):
    return self._params

def _make_convolutional_autoregressive_masks(
    params,
    event_size,
    hidden_units,
    input_order='left-to-right',
    hidden_degrees='equal',
    seed=None,
):
    kernel_size = event_size[:-1]
    conv_constraint = _make_kernel_constraint(kernel_size, input_order=input_order)
    dense_masks = _make_dense_autoregressive_masks(params=params, event_size=event_size[-1], hidden_units=hidden_units,
                                             input_order=input_order, hidden_degrees=hidden_degrees, seed=seed)

    conv_masks = []
    for dmask in dense_masks:
        cmask = conv_constraint(np.ones(kernel_size + dmask.shape))
        cmask[kernel_size[0] // 2, kernel_size[1] // 2, :, :] = cmask[kernel_size[0] // 2, kernel_size[1] // 2, :, :] * dmask
        conv_masks.append(cmask)
    return conv_masks


def _make_kernel_constraint(kernel_size, incl_center=True, input_order='left-to-right'):
    """Make the masking function for layer kernels."""
    kernel_size = tuple((np.array(kernel_size) * np.ones((2,)).astype(int)))
    mask = np.zeros(kernel_size)
    vcenter, hcenter = kernel_size[0] // 2, kernel_size[1] // 2
    mask[0:vcenter] = 1.
    mask[vcenter, 0:(hcenter + int(incl_center))] = 1.
    mask = mask[:, :, np.newaxis, np.newaxis]
    if input_order == 'right-to-left':
        mask = mask[::-1][:, ::-1]
    return lambda x: x * mask


def _make_dense_autoregressive_masks(
    params,
    event_size,
    hidden_units,
    input_order='left-to-right',
    hidden_degrees='equal',
    seed=None,
):
  """Creates masks for use in dense MADE [Germain et al. (2015)][1] networks.
  See the documentation for `AutoregressiveNetwork` for the theory and
  application of MADE networks. This function lets you construct your own dense
  MADE networks by applying the returned masks to each dense layer. E.g. a
  consider an autoregressive network that takes `event_size`-dimensional vectors
  and produces `params`-parameters per input, with `num_hidden` hidden layers,
  with `hidden_size` hidden units each.
  ```python
  def random_made(x):
    masks = tfb._make_dense_autoregressive_masks(
        params=params,
        event_size=event_size,
        hidden_units=[hidden_size] * num_hidden)
    output_sizes = [hidden_size] * num_hidden
    input_size = event_size
    for (mask, output_size) in zip(masks, output_sizes):
      mask = tf.cast(mask, tf.float32)
      x = tf.matmul(x, tf.random.normal([input_size, output_size]) * mask)
      x = tf.nn.relu(x)
      input_size = output_size
    x = tf.matmul(
        x,
        tf.random.normal([input_size, params * event_size]) * masks[-1])
    x = tf.reshape(x, [-1, event_size, params])
    return x
  y = random_made(tf.zeros([1, event_size]))
  assert [1, event_size, params] == y.shape
  ```
  Each mask is a Numpy boolean array. All masks have the shape `[input_size,
  output_size]`. For example, if we `hidden_units` is a list of two integers,
  the mask shapes will be: `[event_size, hidden_units[0]], [hidden_units[0],
  hidden_units[1]], [hidden_units[1], params * event_size]`.
  You can extend this example with trainable parameters and constraints if
  necessary.
  Args:
    params: Python integer specifying the number of parameters to output
      per input.
    event_size: Python integer specifying the shape of the input to this layer.
    hidden_units: Python `list`-like of non-negative integers, specifying
      the number of units in each hidden layer.
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_degrees: Method for assigning degrees to the hidden units:
      'equal', 'random'. If 'equal', hidden units in each layer are allocated
      equally (up to a remainder term) to each degree. Default: 'equal'.
    seed: If not `None`, seed to use for 'random' `input_order` and
      `hidden_degrees`.
  Returns:
    masks: A list of masks that should be applied the dense matrices of
      individual densely connected layers in the MADE network. Each mask is a
      Numpy boolean array.
  #### References
  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
  """
  if seed is None:
    input_order_seed = None
    degrees_seed = None
  else:
    input_order_seed, degrees_seed = np.random.RandomState(seed).randint(
        2**31, size=2)
  input_order = _create_input_order(
      event_size, input_order, seed=input_order_seed)
  masks = _create_masks(_create_degrees(
      input_size=event_size,
      hidden_units=hidden_units,
      input_order=input_order,
      hidden_degrees=hidden_degrees,
      seed=degrees_seed))
  # In the final layer, we will produce `params` outputs for each of the
  # `event_size` inputs.  But `masks[-1]` has shape `[hidden_units[-1],
  # event_size]`.  Thus, we need to expand the mask to `[hidden_units[-1],
  # event_size * params]` such that all units for the same input are masked
  # identically.  In particular, we tile the mask so the j-th element of
  # `tf.unstack(output, axis=-1)` is a tensor of the j-th parameter/unit for
  # each input.
  #
  # NOTE: Other orderings of the output could be faster -- should benchmark.
  masks[-1] = np.reshape(
      np.tile(masks[-1][..., tf.newaxis], [1, 1, params]),
      [masks[-1].shape[0], event_size * params])
  return masks


def _list(xs):
  """Convert the given argument to a list."""
  try:
    return list(xs)
  except TypeError:
    return [xs]


def _create_input_order(input_size, input_order='left-to-right', seed=None):
  """Returns a degree vectors for the input."""
  if isinstance(input_order, six.string_types):
    if input_order == 'left-to-right':
      return np.arange(start=1, stop=input_size + 1)
    elif input_order == 'right-to-left':
      return np.arange(start=input_size, stop=0, step=-1)
    elif input_order == 'random':
      ret = np.arange(start=1, stop=input_size + 1)
      if seed is None:
        rng = np.random
      else:
        rng = np.random.RandomState(seed)
      rng.shuffle(ret)
      return ret
  elif np.all(np.sort(np.array(input_order)) == np.arange(1, input_size + 1)):
    return np.array(input_order)

  raise ValueError('Invalid input order: "{}".'.format(input_order))


def _create_degrees(input_size,
                    hidden_units=None,
                    input_order='left-to-right',
                    hidden_degrees='equal',
                    seed=None):
  """Returns a list of degree vectors, one for each input and hidden layer.
  A unit with degree d can only receive input from units with degree < d. Output
  units always have the same degree as their associated input unit.
  Args:
    input_size: Number of inputs.
    hidden_units: list with the number of hidden units per layer. It does not
      include the output layer. Each hidden unit size must be at least the size
      of length (otherwise autoregressivity is not possible).
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_degrees: Method for assigning degrees to the hidden units:
      'equal', 'random'.  If 'equal', hidden units in each layer are allocated
      equally (up to a remainder term) to each degree.  Default: 'equal'.
    seed: If not `None`, use as a seed for the 'random' hidden_degrees.
  Raises:
    ValueError: invalid input order.
    ValueError: invalid hidden degrees.
  """
  input_order = _create_input_order(input_size, input_order)
  degrees = [input_order]

  if hidden_units is None:
    hidden_units = []

  for units in hidden_units:
    if isinstance(hidden_degrees, six.string_types):
      if hidden_degrees == 'random':
        if seed is None:
          rng = np.random
        else:
          rng = np.random.RandomState(seed)
        # samples from: [low, high)
        degrees.append(
            rng.randint(low=min(np.min(degrees[-1]), input_size - 1),
                        high=input_size,
                        size=units))
      elif hidden_degrees == 'equal':
        min_degree = min(np.min(degrees[-1]), input_size - 1)
        degrees.append(np.maximum(
            min_degree,
            # Evenly divide the range `[1, input_size - 1]` in to `units + 1`
            # segments, and pick the boundaries between the segments as degrees.
            np.ceil(np.arange(1, units + 1)
                    * (input_size - 1) / float(units + 1)).astype(np.int32)))
    else:
      raise ValueError('Invalid hidden order: "{}".'.format(hidden_degrees))

  return degrees


def _create_masks(degrees):
  """Returns a list of binary mask matrices enforcing autoregressivity."""
  return [
      # Create input->hidden and hidden->hidden masks.
      inp[:, np.newaxis] <= out
      for inp, out in zip(degrees[:-1], degrees[1:])
  ] + [
      # Create hidden->output mask.
      degrees[-1][:, np.newaxis] < degrees[0]
  ]


def _make_masked_initializer(mask, initializer):
  """Returns a masked version of the given initializer."""
  initializer = tf.keras.initializers.get(initializer)
  def masked_initializer(shape, dtype=None, partition_info=None):
    # If no `partition_info` is given, then don't pass it to `initializer`, as
    # `initializer` may be a `tf.initializers.Initializer` (which don't accept a
    # `partition_info` argument).
    if partition_info is None:
      x = initializer(shape, dtype)
    else:
      x = initializer(shape, dtype, partition_info)
    return tf.cast(mask, x.dtype) * x
  return masked_initializer


def _make_masked_constraint(mask, constraint=None):
  constraint = tf.keras.constraints.get(constraint)
  def masked_constraint(x):
    x = tf.convert_to_tensor(x, dtype_hint=tf.float32, name='x')
    if constraint is not None:
      x = constraint(x)
    return tf.cast(mask, x.dtype) * x
  return masked_constraint


def _validate_bijector_fn(bijector_fn):
  """Validates the output of `bijector_fn`."""

  def _wrapper(x, **condition_kwargs):
    """A wrapper that validates `bijector_fn`."""
    bijector = bijector_fn(x, **condition_kwargs)
    if bijector.forward_min_event_ndims != bijector.inverse_min_event_ndims:
      # Current code won't really work with this, but in principle we could
      # implement this.
      raise ValueError('Bijectors which alter `event_ndims` are not supported.')
    if bijector.forward_min_event_ndims > 0:
      # Mustn't break auto-regressivity,
      raise ValueError(
          'Bijectors with `forward_min_event_ndims` > 0 are not supported.')
    return bijector
