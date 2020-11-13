import functools

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import logistic
from tensorflow_probability.python.distributions import mixture_same_family
from tensorflow_probability.python.distributions import quantized_distribution
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.layers import weight_norm

class PixelCNNNetwork(tf.keras.layers.Layer):
  """Keras `Layer` to parameterize a Pixel CNN++ distribution.
  This is a Keras implementation of the Pixel CNN++ network, as described in
  Salimans et al. (2017)[1] and van den Oord et al. (2016)[2].
  (https://github.com/openai/pixel-cnn).
  #### References
  [1]: Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma.
       PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture
       Likelihood and Other Modifications. In _International Conference on
       Learning Representations_, 2017.
       https://pdfs.semanticscholar.org/9e90/6792f67cbdda7b7777b69284a81044857656.pdf
       Additional details at https://github.com/openai/pixel-cnn
  [2]: Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt,
       Alex Graves, and Koray Kavukcuoglu. Conditional Image Generation with
       PixelCNN Decoders. In _30th Conference on Neural Information Processing
       Systems_, 2016.
       https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf
  """

  def __init__(
      self,
      dropout_p=0.5,
      num_resnet=5,
      num_hierarchies=3,
      num_filters=160,
      num_logistic_mix=10,
      receptive_field_dims=(3, 3),
      resnet_activation='concat_elu',
      use_weight_norm=True,
      use_data_init=True,
      dtype=tf.float32):
    """Initialize the neural network for the Pixel CNN++ distribution.
    Args:
      dropout_p: `float`, the dropout probability. Should be between 0 and 1.
      num_resnet: `int`, the number of layers (shown in Figure 2 of [2]) within
        each highest-level block of Figure 2 of [1].
      num_hierarchies: `int`, the number of hightest-level blocks (separated by
        expansions/contractions of dimensions in Figure 2 of [1].)
      num_filters: `int`, the number of convolutional filters.
      num_logistic_mix: `int`, number of components in the logistic mixture
        distribution.
      receptive_field_dims: `tuple`, height and width in pixels of the receptive
        field of the convolutional layers above and to the left of a given
        pixel. The width (second element of the tuple) should be odd. Figure 1
        (middle) of [2] shows a receptive field of (3, 5) (the row containing
        the current pixel is included in the height). The default of (3, 3) was
        used to produce the results in [1].
      resnet_activation: `string`, the type of activation to use in the resnet
        blocks. May be 'concat_elu', 'elu', or 'relu'.
      use_weight_norm: `bool`, if `True` then use weight normalization.
      use_data_init: `bool`, if `True` then use data-dependent initialization
        (has no effect if `use_weight_norm` is `False`).
      dtype: Data type of the layer.
    """
    super(PixelCNNNetwork, self).__init__(dtype=dtype)
    self._dropout_p = dropout_p
    self._num_resnet = num_resnet
    self._num_hierarchies = num_hierarchies
    self._num_filters = num_filters
    self._num_logistic_mix = num_logistic_mix
    self._receptive_field_dims = receptive_field_dims
    self._resnet_activation = resnet_activation

    if use_weight_norm:
      def layer_wrapper(layer):
        def wrapped_layer(*args, **kwargs):
          return weight_norm.WeightNorm(
              layer(*args, **kwargs), data_init=use_data_init)
        return wrapped_layer
      self._layer_wrapper = layer_wrapper
    else:
      self._layer_wrapper = lambda layer: layer

  def build(self, input_shape):
    dtype = self.dtype
    if len(input_shape) == 2:
      batch_image_shape, batch_conditional_shape = input_shape
      conditional_input = tf.keras.layers.Input(
          shape=batch_conditional_shape[1:], dtype=dtype)
    else:
      batch_image_shape = input_shape
      conditional_input = None

    image_shape = batch_image_shape[1:]
    image_input = tf.keras.layers.Input(shape=image_shape, dtype=dtype)

    if self._resnet_activation == 'concat_elu':
      activation = tf.keras.layers.Lambda(
          lambda x: tf.nn.elu(tf.concat([x, -x], axis=-1)), dtype=dtype)
    else:
      activation = tf.keras.activations.get(self._resnet_activation)

    # Define layers with default inputs and layer wrapper applied
    Conv2D = functools.partial(  # pylint:disable=invalid-name
        self._layer_wrapper(tf.keras.layers.Convolution2D),
        filters=self._num_filters,
        padding='same',
        dtype=dtype)

    Dense = functools.partial(  # pylint:disable=invalid-name
        self._layer_wrapper(tf.keras.layers.Dense), dtype=dtype)

    Conv2DTranspose = functools.partial(  # pylint:disable=invalid-name
        self._layer_wrapper(tf.keras.layers.Conv2DTranspose),
        filters=self._num_filters,
        padding='same',
        strides=(2, 2),
        dtype=dtype)

    rows, cols = self._receptive_field_dims

    # Define the dimensions of the valid (unmasked) areas of the layer kernels
    # for stride 1 convolutions in the internal layers.
    kernel_valid_dims = {'vertical': (rows - 1, cols),
                         'horizontal': (2, cols // 2 + 1)}

    # Define the size of the kernel necessary to center the current pixel
    # correctly for stride 1 convolutions in the internal layers.
    kernel_sizes = {'vertical': (2 * rows - 3, cols), 'horizontal': (3, cols)}

    # Make the kernel constraint functions for stride 1 convolutions in internal
    # layers.
    kernel_constraints = {
        k: _make_kernel_constraint(kernel_sizes[k], (0, v[0]), (0, v[1]))
        for k, v in kernel_valid_dims.items()}

    # Build the initial vertical stack/horizontal stack convolutional layers,
    # as shown in Figure 1 of [2]. The receptive field of the initial vertical
    # stack layer is a rectangular area centered above the current pixel.
    vertical_stack_init = Conv2D(
        kernel_size=(2 * rows - 1, cols),
        kernel_constraint=_make_kernel_constraint(
            (2 * rows - 1, cols), (0, rows - 1), (0, cols)))(image_input)

    # In Figure 1 [2], the receptive field of the horizontal stack is
    # illustrated as the pixels in the same row and to the left of the current
    # pixel. [1] increases the height of this receptive field from one pixel to
    # two (`horizontal_stack_left`) and additionally includes a subset of the
    # row of pixels centered above the current pixel (`horizontal_stack_up`).
    horizontal_stack_up = Conv2D(
        kernel_size=(3, cols),
        kernel_constraint=_make_kernel_constraint(
            (3, cols), (0, 1), (0, cols)))(image_input)

    horizontal_stack_left = Conv2D(
        kernel_size=(3, cols),
        kernel_constraint=_make_kernel_constraint(
            (3, cols), (0, 2), (0, cols // 2)))(image_input)

    horizontal_stack_init = tf.keras.layers.add(
        [horizontal_stack_up, horizontal_stack_left], dtype=dtype)

    layer_stacks = {
        'vertical': [vertical_stack_init],
        'horizontal': [horizontal_stack_init]}

    # Build the downward pass of the U-net (left-hand half of Figure 2 of [1]).
    # Each `i` iteration builds one of the highest-level blocks (identified as
    # 'Sequence of 6 layers' in the figure, consisting of `num_resnet=5` stride-
    # 1 layers, and one stride-2 layer that contracts the height/width
    # dimensions). The `_` iterations build the stride 1 layers. The layers of
    # the downward pass are stored in lists, since we'll later need them to make
    # skip-connections to layers in the upward pass of the U-net (the skip-
    # connections are represented by curved lines in Figure 2 [1]).
    for i in range(self._num_hierarchies):
      for _ in range(self._num_resnet):
        # Build a layer shown in Figure 2 of [2]. The 'vertical' iteration
        # builds the layers in the left half of the figure, and the 'horizontal'
        # iteration builds the layers in the right half.
        for stack in ['vertical', 'horizontal']:
          input_x = layer_stacks[stack][-1]
          x = activation(input_x)
          x = Conv2D(kernel_size=kernel_sizes[stack],
                     kernel_constraint=kernel_constraints[stack])(x)

          # Add the vertical-stack layer to the horizontal-stack layer
          if stack == 'horizontal':
            h = activation(layer_stacks['vertical'][-1])
            h = Dense(self._num_filters)(h)
            x = tf.keras.layers.add([h, x], dtype=dtype)

          x = activation(x)
          x = tf.keras.layers.Dropout(self._dropout_p, dtype=dtype)(x)
          x = Conv2D(filters=2*self._num_filters,
                     kernel_size=kernel_sizes[stack],
                     kernel_constraint=kernel_constraints[stack])(x)

          if conditional_input is not None:
            h_projection = _build_and_apply_h_projection(
                conditional_input, self._num_filters, dtype=dtype)
            x = tf.keras.layers.add([x, h_projection], dtype=dtype)

          x = _apply_sigmoid_gating(x)

          # Add a residual connection from the layer's input.
          out = tf.keras.layers.add([input_x, x], dtype=dtype)
          layer_stacks[stack].append(out)

      if i < self._num_hierarchies - 1:
        # Build convolutional layers that contract the height/width dimensions
        # on the downward pass between each set of layers (e.g. contracting from
        # 32x32 to 16x16 in Figure 2 of [1]).
        for stack in ['vertical', 'horizontal']:
          # Define kernel dimensions/masking to maintain the autoregressive
          # property.
          x = layer_stacks[stack][-1]
          h, w = kernel_valid_dims[stack]
          kernel_height = 2 * h
          if stack == 'vertical':
            kernel_width = w + 1
          else:
            kernel_width = 2 * w

          kernel_size = (kernel_height, kernel_width)
          kernel_constraint = _make_kernel_constraint(
              kernel_size, (0, h), (0, w))
          x = Conv2D(strides=(2, 2), kernel_size=kernel_size,
                     kernel_constraint=kernel_constraint)(x)
          layer_stacks[stack].append(x)

    # Upward pass of the U-net (right-hand half of Figure 2 of [1]). We stored
    # the layers of the downward pass in a list, in order to access them to make
    # skip-connections to the upward pass. For the upward pass, we need to keep
    # track of only the current layer, so we maintain a reference to the
    # current layer of the horizontal/vertical stack in the `upward_pass` dict.
    # The upward pass begins with the last layer of the downward pass.
    upward_pass = {key: stack.pop() for key, stack in layer_stacks.items()}

    # As with the downward pass, each `i` iteration builds a highest level block
    # in Figure 2 [1], and the `_` iterations build individual layers within the
    # block.
    for i in range(self._num_hierarchies):
      num_resnet = self._num_resnet if i == 0 else self._num_resnet + 1

      for _ in range(num_resnet):
        # Build a layer as shown in Figure 2 of [2], with a skip-connection
        # from the symmetric layer in the downward pass.
        for stack in ['vertical', 'horizontal']:
          input_x = upward_pass[stack]
          x_symmetric = layer_stacks[stack].pop()

          x = activation(input_x)
          x = Conv2D(kernel_size=kernel_sizes[stack],
                     kernel_constraint=kernel_constraints[stack])(x)

          # Include the vertical-stack layer of the upward pass in the layers
          # to be added to the horizontal layer.
          if stack == 'horizontal':
            x_symmetric = tf.keras.layers.Concatenate(axis=-1, dtype=dtype)(
                [upward_pass['vertical'], x_symmetric])

          # Add a skip-connection from the symmetric layer in the downward
          # pass to the layer `x` in the upward pass.
          h = activation(x_symmetric)
          h = Dense(self._num_filters)(h)
          x = tf.keras.layers.add([h, x], dtype=dtype)

          x = activation(x)
          x = tf.keras.layers.Dropout(self._dropout_p, dtype=dtype)(x)
          x = Conv2D(filters=2*self._num_filters,
                     kernel_size=kernel_sizes[stack],
                     kernel_constraint=kernel_constraints[stack])(x)

          if conditional_input is not None:
            h_projection = _build_and_apply_h_projection(
                conditional_input, self._num_filters, dtype=dtype)
            x = tf.keras.layers.add([x, h_projection], dtype=dtype)

          x = _apply_sigmoid_gating(x)
          upward_pass[stack] = tf.keras.layers.add([input_x, x], dtype=dtype)

    # Define deconvolutional layers that expand height/width dimensions on the
    # upward pass (e.g. expanding from 8x8 to 16x16 in Figure 2 of [1]), with
    # the correct kernel dimensions/masking to maintain the autoregressive
    # property.
      if i < self._num_hierarchies - 1:
        for stack in ['vertical', 'horizontal']:
          h, w = kernel_valid_dims[stack]
          kernel_height = 2 * h - 2
          if stack == 'vertical':
            kernel_width = w + 1
            kernel_constraint = _make_kernel_constraint(
                (kernel_height, kernel_width), (h - 2, kernel_height), (0, w))
          else:
            kernel_width = 2 * w - 2
            kernel_constraint = _make_kernel_constraint(
                (kernel_height, kernel_width), (h - 2, kernel_height),
                (w - 2, kernel_width))

          x = upward_pass[stack]
          x = Conv2DTranspose(kernel_size=(kernel_height, kernel_width),
                              kernel_constraint=kernel_constraint)(x)
          upward_pass[stack] = x

    x_out = tf.keras.layers.ELU(dtype=dtype)(upward_pass['horizontal'])

    inputs = (image_input if conditional_input is None
              else [image_input, conditional_input])
    self._network = tf.keras.Model(inputs=inputs, outputs=x_out)
    super(PixelCNNNetwork, self).build(input_shape)

  def call(self, inputs, training=None):
    """Call the Pixel CNN network model.
    Args:
      inputs: 4D `Tensor` of image data with dimensions [batch size, height,
        width, channels] or a 2-element `list`. If `list`, the first element is
        the 4D image `Tensor` and the second element is a `Tensor` with
        conditional input data (e.g. VAE encodings or class labels) with the
        same leading batch dimension as the image `Tensor`.
      training: `bool` or `None`. If `bool`, it controls the dropout layer,
        where `True` implies dropout is active. If `None`, it it defaults to
        `tf.keras.backend.learning_phase()`
    Returns:
      outputs: a 3- or 4-element `list` of `Tensor`s in the following order:
        component_logits: 4D `Tensor` of logits for the Categorical distribution
          over Quantized Logistic mixture components. Dimensions are
          `[batch_size, height, width, num_logistic_mix]`.
        locs: 4D `Tensor` of location parameters for the Quantized Logistic
          mixture components. Dimensions are `[batch_size, height, width,
          num_logistic_mix, num_channels]`.
        scales: 4D `Tensor` of location parameters for the Quantized Logistic
          mixture components. Dimensions are `[batch_size, height, width,
          num_logistic_mix, num_channels]`.
        coeffs: 4D `Tensor` of coefficients for the linear dependence among
          color channels, included only if the image has more than one channel.
          Dimensions are `[batch_size, height, width, num_logistic_mix,
          num_coeffs]`, where
          `num_coeffs = num_channels * (num_channels - 1) // 2`.
    """
    return self._network(inputs, training=training)


def _make_kernel_constraint(kernel_size, valid_rows, valid_columns):
  """Make the masking function for layer kernels."""
  mask = np.zeros(kernel_size)
  lower, upper = valid_rows
  left, right = valid_columns
  mask[lower:upper, left:right] = 1.
  mask = mask[:, :, np.newaxis, np.newaxis]
  return lambda x: x * mask


def _build_and_apply_h_projection(h, num_filters, dtype):
  """Project the conditional input."""
  #h = tf.keras.layers.Flatten(dtype=dtype)(h)
  h_projection = tf.keras.layers.Dense(
      2*num_filters, kernel_initializer='random_normal', dtype=dtype)(h)
  if len(h_projection.shape) == 4:
      return h_projection
  return h_projection[..., tf.newaxis, tf.newaxis, :]



def _apply_sigmoid_gating(x):
  """Apply the sigmoid gating in Figure 2 of [2]."""
  activation_tensor, gate_tensor = tf.split(x, 2, axis=-1)
  sigmoid_gate = tf.sigmoid(gate_tensor)
  return tf.keras.layers.multiply(
      [sigmoid_gate, activation_tensor], dtype=x.dtype)