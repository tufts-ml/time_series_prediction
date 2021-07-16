from tensorflow.keras.layers import Lambda, Dropout, Add, Input, Dense, Concatenate, Flatten, Reshape, Conv2D, \
    LeakyReLU, Activation
from tensorflow.keras.models import Model
import numpy as np
from ..util.distributions import *
from ..util.util import get_mask
from .wrn import residual_block

class PixelCNNNetwork(object):
    def __init__(self, input_shape=None, distribution=None, distribution_model=None, pcnn_network=None, input_filters=64, **kwargs):
        self.input_shape = input_shape
        self.distribution = distribution
        self.distribution_model = distribution_model
        self.args = kwargs
        self.pcnnModel = pcnn_network
        self.input_filters = input_filters
        self.decoder_shape = self.input_shape[:-1] + (self.input_filters,)

    @classmethod
    def get(cls, input_shape=None, distribution=None, **kwargs):
        obj = cls(input_shape=input_shape, distribution=distribution, **kwargs)
        return obj.create_networks() + (obj.decoder_shape,)

    @classmethod
    def get_no_dist(cls, input_shape=None, distribution=None, distribution_model=None, **kwargs):
        obj = cls(input_shape=input_shape, distribution=distribution, distribution_model=distribution_model, **kwargs)
        return obj.create_networks(return_dist=False)

    def create_networks(self, return_dist=True):
        self.build_pixel_cnn(**self.args)

        sample_input = Input(self.input_shape[:-1] + (self.input_filters,), name='pixel_CNN_sample_input')
        mean_input = Input(self.input_shape[:-1] + (self.input_filters,), name='pixel_CNN_mean_input')
        sample_model = Lambda(lambda a: self.pixel_cnn_generate(a), name='pixel_cnn_sampler')(sample_input)
        mean_model = Lambda(lambda a: self.pixel_cnn_generate(a, mean=True), name='pixel_cnn_mean')(mean_input)
        sample_model = Model(inputs=sample_input, outputs=sample_model)
        mean_model = Model(inputs=mean_input, outputs=mean_model)
        if return_dist:
            return self.pcnnModel, sample_model, mean_model
        return self.no_dist_model, sample_model, mean_model

    def make_kernel_constraint(self, kernel_size, incl_center=True):
        """Make the masking function for layer kernels."""
        kernel_size = tuple((np.array(kernel_size) * np.ones((2,)).astype(int)))
        mask = np.zeros(kernel_size)
        vcenter, hcenter = kernel_size[0] // 2, kernel_size[1] // 2
        mask[0:vcenter] = 1.
        mask[vcenter, 0:(hcenter + int(incl_center))] = 1.
        mask = mask[:, :, np.newaxis, np.newaxis]
        return lambda x: x * mask

    def add_pixel_cnn_conditional(self, x, conditional, pixelcnn_kernel_size=3, bn=True, inc_cond=True, activate=True, add=False):
        if len(conditional.shape) > 2:
            filters = x.shape[-1]
            cx = Conv2D(filters=filters, kernel_size=1, padding='same',
                        use_bias=True)(conditional)
        else:
            numel = int(np.prod(x.shape[1:]))
            cx = Reshape(x.shape[1:])(Dense(numel)(conditional))
        if activate:
            #x = tf.keras.layers.BatchNormalization(gamma_initializer='glorot_uniform')(x) if bn else x
            x = tf.keras.layers.Lambda(lambda a: tf.nn.tanh(tf.concat([a, -a], axis=-1)))(x)
            #cx = tf.keras.layers.BatchNormalization(gamma_initializer='glorot_uniform')(cx) if bn else cx
            cx = tf.keras.layers.Lambda(lambda a: tf.nn.tanh(tf.concat([a, -a], axis=-1)))(cx)
        if inc_cond and not add:
            x = Concatenate()([x, cx])
        elif inc_cond:
            x = Lambda(lambda x: x)(x)
            x = Add()([x, cx])
        return x

    def build_pixel_cnn(self, pixelcnn_filters=64, pixelcnn_kernel_size=3, pixelcnn_dropout_p=0., pixelcnn_layers=3,
                        pixelcnn_bn=True, pixelcnn_all_layers_cond=True, pixelcnn_use_resnet=True, pixelcnn_resnet_args={},
                        **kwargs):
        shape = self.input_shape
        distribution = self.distribution
        img_input = Input(shape, name='pixel_CNN_input')
        conditional_input = Input(shape[:-1] + (self.input_filters,), name='pixel_CNN_cond_input')

        if self.pcnnModel is None:
            if pixelcnn_use_resnet:
                x = Concatenate(name='PixelCNN_concat')([img_input, conditional_input])
                x = residual_block(x, filters=pixelcnn_filters, kernel_size=pixelcnn_kernel_size, mask='a',
                               bn=pixelcnn_bn, wn=True, conv_args={}, name='pixel_block_initial')
                for i in range(pixelcnn_layers):
                    x = residual_block(x, filters=pixelcnn_filters, kernel_size=pixelcnn_kernel_size, mask='b',
                                       bn=pixelcnn_bn, wn=True, conv_args={}, name='pixel_block_' + str(i))

            else:
                raise
                x = Conv2D(filters=pixelcnn_filters, kernel_size=pixelcnn_kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.L2(1.),
                           use_bias=False, kernel_constraint=self.make_kernel_constraint(pixelcnn_kernel_size, False))(
                    img_input)
                x = self.add_pixel_cnn_conditional(x, conditional_input, pixelcnn_kernel_size, pixelcnn_bn)

                for _ in range(pixelcnn_layers):
                    x = Dropout(pixelcnn_dropout_p)(x) if pixelcnn_dropout_p > 0 else x
                    x = tf.keras.layers.BatchNormalization(gamma_initializer='glorot_uniform')(x)
                    x = Conv2D(filters=pixelcnn_filters, kernel_size=pixelcnn_kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.L2(1.),
                               use_bias=False, kernel_constraint=self.make_kernel_constraint(pixelcnn_kernel_size))(x)
                    x = self.add_pixel_cnn_conditional(x, conditional_input, pixelcnn_kernel_size, pixelcnn_bn, inc_cond=pixelcnn_all_layers_cond)

                x = Conv2D(filters=conditional_input.shape[-1], kernel_size=1, padding='same',
                           kernel_regularizer=tf.keras.regularizers.L2(100.),
                           use_bias=False)(x)
                x = self.add_pixel_cnn_conditional(x, conditional_input, pixelcnn_kernel_size, pixelcnn_bn,
                                                   inc_cond=True, activate=False, add=True)
        else:
            x = tf.keras.layers.Concatenate()([img_input, conditional_input])
            x = self.pcnnModel(x)

        def netbuild(xin, output_shapes=None):
            return {k: Reshape(s)(
                Conv2D(filters=np.prod(s[2:]),
                       kernel_size=1,
                       kernel_constraint=get_mask(xin.shape[-1], np.prod(s[2:]), kernel_size=1, mask_type='b'))(xin))
                    for k, s in output_shapes.items()}

        if not (self.distribution_model is None):
            dist = self.distribution_model(x)
        else:
            dist = self.to_distribution(x, distribution, self.input_shape, netbuild)
        self.pcnnModel = Model(inputs=[img_input, conditional_input], outputs=dist)
        self.no_dist_model = Model(inputs=[img_input, conditional_input], outputs=x)

    def pixel_cnn_generate(self, x, mean=False, approx_iters=0):
        samples0 = tf.expand_dims(tf.zeros(self.input_shape), 0)
        samples0 = samples0 + 0 * tf.reshape(tf.reduce_sum(x, list(range(len(x.shape)))[1:]), (-1, 1, 1, 1))

        image_height, image_width = self.input_shape[0], self.input_shape[1]

        def loop_body(index, args):
            samples, cond = args
            samples_new = self.pcnnModel([samples, cond])
            samples_new = samples_new.mean() if mean else samples_new.sample()
            if approx_iters:  # Approximate the pixel-by-pixel sampling
                return index + 1, (samples_new, cond)

            # Update the current pixel
            samples = tf.transpose(samples, [1, 2, 3, 0])  # h w c b
            samples_new = tf.transpose(samples_new, [1, 2, 3, 0])
            row, col = index // image_width, index % image_width
            updates = samples_new[row, col, ...][tf.newaxis, ...]  # 1 c b
            samples = tf.tensor_scatter_nd_update(samples, [[row, col]], updates)
            samples = tf.transpose(samples, [3, 0, 1, 2])
            return index + 1, (samples, cond)

        index0 = tf.zeros([], dtype=tf.int32)

        # Construct the while loop for sampling
        total_pixels = image_height * image_width if approx_iters <= 0 else approx_iters
        loop_cond = lambda ind, _: tf.less(ind, total_pixels)
        init_vars = (index0, (samples0, x))
        _, (samples, _) = tf.while_loop(loop_cond, loop_body, init_vars,
                                        parallel_iterations=1)
        return samples

    def to_distribution(self, input, distribution, shape, network_builder=lambda x: x, independent=True,
                        sample=True, name='dist'):
        # Get the parameter shapes for the target distribution
        output_shapes = distribution.param_static_shapes(shape) if distribution else None

        try:  # Case where network builder can take specified output shapes
            output = network_builder(input, output_shapes=output_shapes)
        except Exception as e:
            output = network_builder(input)

        # Get list of parameters for the distribution
        params, tensors = [], []
        for (param, shape) in output_shapes.items():
            params.append(param)
            if type(output) is dict:
                tensors.append(output[param])
            else:
                tensors.append(Reshape(shape)(Dense(shape.num_elements())(output)))

        # Create the distribution object
        independent_transform = tfd.Independent if independent else lambda x: x
        convert_fn = tfd.Distribution.sample if sample else tfd.Distribution.mean

        def distribution_lambda(tensors):
            return independent_transform(distribution(
                **{p: vt for (p, vt) in zip(params, tensors)}))

        output_dist = tfpl.DistributionLambda(distribution_lambda,
                                              convert_to_tensor_fn=convert_fn, name=name)(tensors)
        return output_dist




