from tensorflow.keras.layers import Dense, Layer, Concatenate, Input, Reshape
from tensorflow.keras.layers import Flatten, Add, Lambda, Multiply, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
from .wrn import WRN
from ..util.distributions import *
from ..util.util import interleave
from .pixel import PixelCNNNetwork

class UpscalingNetwork(WRN):
    def __init__(self, distribution=None, upscale_size=2, upscale_actv=0.1, upscale_bn=True, upscale_spatial_cond=True,
                 upscale_transpose=False, upscale_use_conv=True, upscale_pixelcnn=False,
        upscale_resblocks=3, input_shape=None, wrn_rescale=False, upscale_levels=2, input_filters=64, **kwargs):
        self.wrn_size = upscale_size
        self.wrn_actv = upscale_actv
        self.wrn_bn = upscale_bn
        self.input_shape = input_shape
        self.wrn_rescale = wrn_rescale
        self.wrn_resblocks = upscale_resblocks
        self.upscale_levels = upscale_levels
        self.transpose = upscale_transpose
        self.spatial_cond = upscale_spatial_cond
        self.use_conv = upscale_use_conv
        self._ks = 3
        self._stride = 2
        self.distribution = distribution
        self.pixelcnn = upscale_pixelcnn
        self.kwargs = kwargs
        self.input_filters = input_filters
        self.decoder_shape = self.input_shape[:-1] + (self.input_filters,)

    @classmethod
    def get(cls, input_shape=None, distribution=None, **kwargs):
        obj = cls(input_shape=input_shape, distribution=distribution, **kwargs)
        return obj.create_networks() + (obj.decoder_shape,)

    def create_network(self, x):
        for i in range(self.wrn_resblocks):
            layers = []
            x, _ = self.residual_block(
                           x, 0, layers, x.shape[-1], self.wrn_size, tuple([1] * (len(self.input_shape) - 1)),
                transpose=self.transpose, activate_before_residual=False, BN=self.wrn_bn)
        return x

    def create_distribution_model(self, distribution):
        dist_input = Input(tuple([None for i in self.input_shape[:-1]] + [self.wrn_size]))
        outputs = {}
        for output, oshape in distribution.param_static_shapes(self.input_shape).items():
            output_filters = int(np.prod(oshape[len(self.input_shape)-1:]))
            output_dist_shape = oshape[len(self.input_shape)-1:].as_list()
            xout = self.Conv()(filters=output_filters, kernel_size=1, padding='same', strides=1)(dist_input)
            xout = Lambda(lambda a: tf.reshape(a, tf.concat([tf.shape(a)[:-1], tf.constant(output_dist_shape, dtype=tf.int32)], axis=0)))(xout)
            outputs[output] = xout

        # Get list of parameters for the distribution
        params, tensors = [], []
        for (param, shape) in outputs.items():
            params.append(param)
            tensors.append(outputs[param])

        # Create the distribution object
        def distribution_lambda(tensors):
            return tfd.Independent(distribution(
                **{p: vt for (p, vt) in zip(params, tensors)}))

        output_dist = tfpl.DistributionLambda(distribution_lambda, name='upscale_dist')(tensors)
        distribution_model = Model(inputs=dist_input, outputs=output_dist)
        return distribution_model

    def create_networks(self):
        self.distribution_model = self.create_distribution_model(self.distribution)

        xin = Input(self.input_shape)
        condin = Input(self.decoder_shape)
        samplein = Input(self.input_shape)
        meanin = Input(self.input_shape)

        x, sample, mean = self.create_upscale_level(self.upscale_levels, xin, condin, samplein, meanin, self.input_shape)
        x = self.distribution_model(x)
        dist_model = Model(inputs=[xin, condin], outputs=x)
        sample_model = Model(inputs=samplein, outputs=sample)
        mean_model = Model(inputs=meanin, outputs=mean)

        return dist_model, sample_model, mean_model

    def strided_slice(self, startdim=0):
        dims = len(self.input_shape)
        def helper(a):
            if dims >= 2 and startdim == 0:
                a = a[:, ::2]
            if dims >= 3 and startdim <= 1:
                a = a[:, :, ::2]
            if dims >= 4 and startdim <= 2:
                a = a[:, :, :, ::2]
            return a

        return Lambda(helper)

    def interleave_layer(self, dim=0):
        return Lambda(lambda a: interleave(a[0], a[1], dim+1))


    def create_upscale_level(self, level, xin, cond, samplecond, meancond, shape):
        sample_layer = Lambda(lambda a: a.sample())
        mean_layer = Lambda(lambda a: a.mean())

        if level == 0: # Base case, sample the first pixels
            level0in = Input(cond.shape[1:])

            # If we're not keeping spatial structure in conditional, get it to the right shape
            if not self.spatial_cond:
                level0x = Flatten()(level0in)
                level0x = Dense(np.prod(shape))(level0x)
                level0x = Reshape(shape)(level0x)
            else:
                level0x = level0in

            if self.pixelcnn:
                # Use a pixelcnn as the base-level distribution rather than sampling independently
                level0Network = Model(inputs=level0in, outputs=level0x)
                pxnet, pxsample, pxmean = PixelCNNNetwork.get_no_dist(shape, self.distribution, self.distribution_model, **self.kwargs)
                xout = pxnet([xin, level0Network(cond)])
                sample = pxsample(level0Network(samplecond))
                mean = pxmean(level0Network(meancond))
            else:
                # Independent sampling of lowest res (base) image
                level0Network = Model(inputs=level0in, outputs=self.create_network(level0x))
                xout = level0Network(cond)
                sampleout = level0Network(samplecond)
                meanout = level0Network(meancond)

                sample = sample_layer(self.distribution_model(sampleout))
                mean = mean_layer(self.distribution_model(meanout))

            return xout, sample, mean

        # Get the shape of the next smaller image in the pyramid
        shapesliced = tuple([(os // 2 if os > 1 else os) for os in shape[:-1]]) + (shape[-1],)

        # Transform the conditional tensor to the size of the smaller scale image
        if not self.spatial_cond:
            ds_layer = Lambda(lambda a: a)
        else:
            # Downsample the conditional image if we're using a spatially-aligned conditional
            strides = [(2 if s > 1 else 1) for s in shape[:-1]]
            pool_size = [((s + 1) if s > 1 else 1) for s in strides]
            if self.use_conv:
                ds_layer = self.Conv()(filters=self.input_shape[-1], kernel_size=pool_size, strides=strides, padding='same')
            else:
                ds_layer = self.AveragePooling()(pool_size=pool_size, strides=strides, padding='same')
        condsliced = ds_layer(cond)
        samplecondsliced = ds_layer(samplecond)
        meancondsliced = ds_layer(meancond)

        # Slice the input to get a lower res true input
        xsliced = self.strided_slice()(xin)

        # Recurse down, getting the sampled lower-res image
        xout, sample, mean = self.create_upscale_level(level-1, xsliced, condsliced, samplecondsliced, meancondsliced, shapesliced)

        # Loop through each spacial dimension to upscale it by a factor of 2
        for (dim, (input_dim, output_dim)) in enumerate(zip(shapesliced[:-1], shape[:-1])):
            # Skip singleton dimensions
            if output_dim > input_dim:
                # Get the current shape of the input
                currentshape = shape[:dim] + shapesliced[dim:]
                xinput = Input(currentshape)
                condinput = Input(cond.shape[1:])

                # If we're not using spatial conditioning, transform out conditional to be the right size
                if not self.spatial_cond:
                    condpartial = Flatten()(condinput)
                    condpartial = Dense(np.prod(currentshape))(condpartial)
                    condpartial = Reshape(currentshape)(condpartial)
                else:
                    # Downsample conditional to match input shape
                    strides = [(cd // sd) for (cd, sd) in zip(shape[:-1], currentshape[:-1])]
                    pool_size = [((s + 1) if s > 1 else 1) for s in strides]
                    if self.use_conv:
                        ds_layer = self.Conv()(filters=self.input_shape[-1], kernel_size=pool_size, strides=strides,
                                               padding='same')
                    else:
                        ds_layer = self.AveragePooling()(pool_size=pool_size, strides=strides, padding='same')
                    condpartial = ds_layer(condinput)

                # Create a new model for this upscaling step
                x = Concatenate()([xinput, condpartial])
                upscale_model = Model(inputs=[xinput, condinput], outputs=self.create_network(x))

                # Get the output based on the current pixels and stitch into full image
                xout = self.interleave_layer(dim)([xout, upscale_model([xsliced, cond])])

                # Get samples of the new pixels
                samplenew = upscale_model([sample, samplecond])
                samplenew = sample_layer(self.distribution_model(samplenew))
                sample = self.interleave_layer(dim)([sample, samplenew])

                # Get the means of the new pixels
                meannew = upscale_model([mean, meancond])
                meannew = sample_layer(self.distribution_model(meannew))
                mean = self.interleave_layer(dim)([mean, meannew])

                # Get the partially sliced true input for the next dimension
                xsliced = self.strided_slice(dim + 1)(xin)

        return xout, sample, mean


