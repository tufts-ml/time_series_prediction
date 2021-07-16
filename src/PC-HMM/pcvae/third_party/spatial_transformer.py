import numpy as np
import tensorflow as tf

# from tensorlayer.layers.core import LayersConfig
# from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES



def transformer(U, theta, out_size, name='SpatialTransformer2dAffine'):
    """Spatial Transformer Layer for `2D Affine Transformation <https://en.wikipedia.org/wiki/Affine_transformation>`__
    , see :class:`SpatialTransformer2dAffine` class.

    Parameters
    ----------
    U : list of float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the localisation network should be [num_batch, 6], value range should be [0, 1] (via tanh).
    out_size: tuple of int
        The size of the output of the network (height, width)
    name: str
        Optional function name

    Returns
    -------
    Tensor
        The transformed tensor.

    References
    ----------
    - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`__
    - `TensorFlow/Models <https://github.com/tensorflow/models/tree/master/transformer>`__

    Notes
    -----
    To initialize the network to the identity transform init.

    >>> import tensorflow as tf
    >>> # ``theta`` to
    >>> identity = np.array([[1., 0., 0.], [0., 1., 0.]])
    >>> identity = identity.flatten()
    >>> theta = tf.Variable(initial_value=identity)

    """

    def _repeat(x, n_repeats):
        rep = tf.transpose(a=tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), perm=[1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch = tf.shape(input=im)[0]
        height = tf.shape(input=im)[1]
        width = tf.shape(input=im)[2]
        channels = tf.shape(input=im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(input=im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(input=im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        #im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output

    def _meshgrid(height, width):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(a=tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), perm=[1, 0])
        )
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1), tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
        return grid

    def _transform(theta, input_dim, out_size):
        num_batch = tf.shape(input=input_dim)[0]
        num_channels = tf.shape(input=input_dim)[3]
        theta = tf.reshape(theta, (-1, 2, 3))
        theta = tf.cast(theta, 'float32')

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width)
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_batch]))
        grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(theta, grid)
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])

        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)

        output = tf.reshape(input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
        return output

    output = _transform(theta, U, out_size)
    return output

def create_affine(a1, a2, a3, b1, b2, b3):
    a = tf.stack([a1, a2, a3], axis=-1)
    b = tf.stack([b1, b2, b3], axis=-1)
    c = tf.stack([tf.zeros_like(a1), tf.zeros_like(a1), tf.ones_like(a1)], axis=-1)
    output = tf.stack([a, b, c], axis=-2)
    return output

def create_translate(inputs, h_range, v_range):
    inputs = tf.tanh(inputs)
    one, zero = tf.ones_like(inputs[:, 0]), tf.zeros_like(inputs[:, 0])
    return create_affine(one, zero, inputs[:, 0] * v_range, zero, one, inputs[:, 1] * h_range)

def create_rotate(inputs, rotation_range, shear_range):
    one, zero = tf.ones_like(inputs[:, 0]), tf.zeros_like(inputs[:, 0])
    shear_inputs = tf.tanh(inputs[:, 1]) * shear_range
    inputs = tf.tanh(inputs[:, 0]) * rotation_range
    return create_affine(tf.cos(inputs), -tf.sin(inputs + shear_inputs), zero, tf.sin(inputs), tf.cos(inputs + shear_inputs), zero)

def create_scale(inputs, h_range, v_range):
    inputs = -tf.nn.softplus(-tf.tanh(inputs))
    one, zero = tf.ones_like(inputs[:, 0]), tf.zeros_like(inputs[:, 0])
    h = tf.pow(h_range, inputs[:, 1])
    v = tf.pow(v_range, inputs[:, 0])
    scale = 1. / (1.5 * 1.5)
    return create_affine(scale * v, zero, zero, zero, scale * h, zero)

@tf.custom_gradient
def clip_gradients(y):
    def backward(dy):
        return tf.clip_by_norm(dy, 0.1, axes=-1)
    return y, backward

class SpatialTransformer(tf.keras.layers.Layer):

    def __init__(self, out_shape=None, rotation_range=0, shear_range=0, h_translation_range=0, v_translation_range=0, h_scale_range=1, v_scale_range=1):
        super(SpatialTransformer, self).__init__()
        self.rotation_range = rotation_range
        self.h_translation_range = h_translation_range
        self.v_translation_range = v_translation_range
        self.h_scale_range = h_scale_range
        self.v_scale_range = v_scale_range
        self.shear_range = shear_range
        self.out_shape = out_shape

    def call(self, inputs):
        transforms, image = inputs
        #transforms = clip_gradients(transforms[:, :6])
        height, width = tuple(image.shape[1:3]) if self.out_shape is None else self.out_shape[:2]
        hpad, vpad = width // 4, height // 4
        paddings = tf.constant([[0, 0], [vpad, vpad], [hpad, hpad], [0, 0]])
        image = tf.pad(image, paddings, 'REFLECT')

        translation = create_translate(transforms[:, :2], self.h_translation_range, self.v_translation_range)
        rotation = create_rotate(transforms[:, 2:4], self.rotation_range, self.shear_range)
        scale = create_scale(transforms[:, 4:6], self.h_scale_range, self.v_scale_range)
        transformation = tf.matmul(scale, tf.matmul(translation, rotation))[:, :2, :]
        transformation = tf.reshape(transformation, (-1, 6))
        return transformer(image, transformation, (height, width))

def test_transformer(image, translation, rotation, scale):
    image = tf.reshape(tf.convert_to_tensor(image, dtype=tf.float32), (1, image.shape[0], image.shape[1], 1))
    transf = tf.constant([list(translation) + [rotation] + list(scale)], dtype=tf.float32)
    t = SpatialTransformer(np.pi / 2., 0.1, 0.1, 1.5, 1.5)
    with tf.GradientTape() as g:
      g.watch(image)
      g.watch(transf)
      imout = t.call([transf, image])
      loss = tf.reduce_sum(imout)
      gimage, gtransf = g.gradient(loss, [image, transf])
    return image, gimage, gtransf

