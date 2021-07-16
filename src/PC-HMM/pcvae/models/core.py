import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Dropout, Add, Input, Dense, Concatenate, Flatten, Reshape, Conv2D, \
    LeakyReLU, Activation
from tensorflow.keras.models import Model
from .base import BaseVAE
from pcvae.third_party.keras_patch import ConsistantKModel
import numpy as np
from ..util.distributions import *
from ..util.optimizers import get_optimizer
from ..util.samplers import get_sampler
from ..networks.networks import get_decoder_network, get_encoder_network, get_predictor_network, get_bridge_network
from scipy.special import logsumexp
from ..networks.priors import get_prior
from ..networks.pixel import PixelCNNNetwork
from ..networks.upscaling import UpscalingNetwork
from ..third_party.spatial_transformer import SpatialTransformer
from sklearn.metrics import confusion_matrix as sk_confusion
from ..util.util import AddWeights, LinearReshape
from ..third_party.glow import GLOW

'''
TODO:
    - Add metrics for log-liklihood ( x )
    - Use nans for unseen labels ( x )
    - Updates to warmup lambda, alpha, beta (   )
    - Move Distributions to new file ( x )
    - Fix minimize entropy ( x )
    - Sampling from prior ( x )
    - Custom samplers ( x )
    - Update WRN for new model (   )
    - Different priors ( x )
    - VAT (   )
    - 2-layer model (   )
'''


class CPC(BaseVAE):
    def __init__(self, input_shape=None, label_shape=None, alpha=1, beta=1, lam=1, recon_weight=1, entropy_weight=0.,
                 use_exact_kl=True,
                 encoded_size=50, variational_dist=None, reconstruction_dist=None, predictor_dist=None,
                 encoder_network=None, decoder_network=None, predictor_network=None, latent_prior=None,
                 consistency_loss=None, optimizer=None, debug=False, align_prior=False, ar_model=None,
                 custom_sampler=None, sampled_reconstructions=False, llik_samples=100,
                 clip_consistency=False, consistency_augmentations=False, translation_range=0, rotation_range=0,
                 scale_range=1, shear_range=0, z_consistancy=False, class_entropy_loss=0., class_entropy_weight=0.,
                 unlabeled_balance=1., pixelcnn_global_loss=True, consistency_grad=True, use_iaf=False, output_iaf=False,
                 recon_noise=0., glow=False, glow_args={}, flow_base_loss=False,
                 *args, **kwargs):

        super(CPC, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.alpha = self.parameter(alpha)
        self.beta =self.parameter(beta)
        self.lam = self.parameter(lam, scale=np.prod(self.input_shape) if self.input_shape else 1.)
        self.recon_weight = self.parameter(recon_weight)
        self.entropy_weight = self.parameter(entropy_weight)
        self.class_entropy_weight = self.parameter(class_entropy_weight)

        self.use_exact_kl = use_exact_kl
        self.encoded_size = encoded_size
        self.kwargs = kwargs
        self.is_setup = False
        self.variational_dist = variational_dist
        self.reconstruction_dist = reconstruction_dist
        self.predictor_dist = predictor_dist
        self.encoder_network = encoder_network
        self.decoder_network = decoder_network
        self.predictor_network = predictor_network
        self.latent_prior = latent_prior
        self.consistency_loss = consistency_loss
        self.optimizer = optimizer
        self.custom_sampler = custom_sampler
        self.sampled_reconstructions = sampled_reconstructions
        self.debug = debug
        self.prior = None
        self.align_prior = align_prior
        self.ar_model = ar_model
        self.llik_samples = llik_samples
        self.unlabeled_balance = unlabeled_balance
        self.pixelcnn_global_loss = pixelcnn_global_loss
        self.consistency_grad = consistency_grad
        self.clip_consistency = clip_consistency
        self.consistency_augmentations = consistency_augmentations
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.z_consistancy = z_consistancy
        self.class_entropy_loss = class_entropy_loss
        self.use_iaf = use_iaf
        self.output_iaf = output_iaf
        self.recon_noise = recon_noise
        self.glow = glow
        self.glow_args = glow_args
        self.flow_base_loss = flow_base_loss
        self.metric = []

    def setup(self, data=None):
        if self.is_setup:
            return

        # Get defaults from dataset if available
        self.variational_dist = self.variational_dist if self.variational_dist else 'MultivariateNormalDiag'
        if data is None:
            self.reconstruction_dist = self.reconstruction_dist if self.reconstruction_dist else 'Normal'
            self.predictor_dist = self.predictor_dist if self.predictor_dist else 'Categorical'
        else:
            self.input_shape = data.shape()
            self.label_shape = data.dim()
            self.metric = data.get_metrics()
            if hasattr(data, 'reconstruction_dist') and not self.reconstruction_dist:
                self.reconstruction_dist = data.reconstruction_dist
            else:
                self.reconstruction_dist = self.reconstruction_dist if self.reconstruction_dist else 'Normal'
            if hasattr(data, 'predictor_dist') and not self.predictor_dist:
                self.predictor_dist = data.predictor_dist
            else:
                self.predictor_dist = self.predictor_dist if self.predictor_dist else 'Categorical'

        # Run functions to get  distributions etc. from string inputs
        self.variational_dist = get_tfd_distribution(self.variational_dist, **self.kwargs)
        self.reconstruction_dist = get_tfd_distribution(self.reconstruction_dist, **self.kwargs)
        self.predictor_dist = get_tfd_distribution(self.predictor_dist, **self.kwargs)
        self.encoder_network = get_encoder_network(self.encoder_network, input_shape=self.input_shape, **self.kwargs)
        self.decoder_network = get_decoder_network(self.decoder_network, input_shape=self.input_shape, encoded_size=self.encoded_size, **self.kwargs)
        self.predictor_network = get_predictor_network(self.predictor_network, **self.kwargs)
        self.latent_prior = get_prior(self.latent_prior, **self.kwargs)
        self.consistency_loss = get_consistency_loss(self.consistency_loss, **self.kwargs)
        self.optimizer = get_optimizer(self.optimizer, **self.kwargs)
        self.custom_sampler = get_sampler(self.custom_sampler, **self.kwargs)

    def to_distribution(self, input, distribution, shape, network_builder=lambda x: x, independent=True,
                        sample=True, transform=False, name='dist'):

        # Separate out spatial transformer parameteres
        if transform:
            tinput = input
            transformed_shape = shape
            shape = (shape[0] + (shape[0] // 2), shape[1] + (shape[1] // 2), shape[2])
            transformation, input = Lambda(lambda x: (x[:, :6], x[:, 6:]))(input)

        # Get the parameter shapes for the target distribution
        try:
            output_shapes = distribution.param_static_shapes(shape)
        except:
            output_shapes = None

        try:  # Case where network builder can take specified output shapes
            output = network_builder(input, output_shapes=output_shapes)
        except Exception as e:
            output = network_builder(input)

        # Get list of parameters for the distribution
        preconstrained = True
        params, tensors = [], []
        if output_shapes:
            for (param, outshape) in output_shapes.items():
                params.append(param)
                if type(output) is dict:
                    tensor = output[param]
                else:
                    tensor = LinearReshape(output_shape=outshape, name='%s_%s' % (name, param))(output)

                try:
                    tensor = distribution.param_constrain(param, tensor)
                except:
                    preconstrained = False
                tensors.append(tensor)
        else:
            preconstrained = False
            params, tensors = ['x'], [output]

        # Apply spatial transformer
        if transform:
            t_range = (np.array(self.translation_range) * np.ones((2,))).astype(np.float32)
            s_range = (np.array(self.scale_range) * np.ones((2,))).astype(np.float32)

            tensors = [SpatialTransformer(transformed_shape, self.rotation_range, self.shear_range,
                                          t_range[0], t_range[1],
                                          s_range[0], s_range[1])([transformation, t]) for t in tensors]

        # Create the distribution object
        independent_transform = tfd.Independent if independent else lambda x: x
        convert_fn = tfd.Distribution.sample if sample else tfd.Distribution.mean

        def distribution_lambda(tensors_in):
            if preconstrained:
                return independent_transform(distribution(preconstrained=True,
                                                          **{p: vt for (p, vt) in zip(params, tensors_in)}))
            return independent_transform(distribution(
                **{p: vt for (p, vt) in zip(params, tensors_in)}))

        output_dist = tfpl.DistributionLambda(distribution_lambda,
                                              convert_to_tensor_fn=convert_fn, name=name)(tensors)

        return output_dist

    def build_encoder(self):
        # Create the base encoder network
        input = Input(self.input_shape, name='encoder_precursor_input')
        output = Flatten()(self.encoder_network(input))
        self.encoder_precursor = Model(inputs=input, outputs=output, name='encoder_precursor')

        # Create network to map encoder output to latent variational distribution
        latent_input = Input(self.encoder_precursor.output_shape[1:], name='latent_input')
        latent_z = self.to_distribution(latent_input, self.variational_dist, self.encoded_size, name='encoded',
                                        independent=False)
        if self.use_iaf:
            latent_z = IAFNetwork(latent_z, latent_input)

        self.latent_model = Model(inputs=latent_input, outputs=latent_z)

        # Create the full encoder network
        encoder_input = Input(self.input_shape, name='encoder_input')
        precursor_output = self.encoder_precursor(encoder_input)
        encoder_output = self.latent_model(precursor_output)

        # Apply the KL divergence from the prior
        self.prior = self.latent_prior(int(np.prod(self.encoded_size)))  # Latent prior distribution
        prior_dist = self.prior(Lambda(lambda x: K.sum(x), name='prior_lambda')(encoder_input))  # Make prior depend on input
        if self.align_prior:
            encoder_output = tfpl.DistributionLambda(lambda x: GaussianMixtureDiag.align(x[0], x[1]))(
                [encoder_output, prior_dist])

        self._encoder = Model(inputs=encoder_input, outputs=encoder_output, name='encoder')
        self._encoder_with_prior = Model(inputs=encoder_input, outputs=[encoder_output, prior_dist, precursor_output],
                                         name='encoder_with_prior')

    def build_decoder(self):
        # Create the base decoder network
        input = Input(self.encoded_size, name='decoder_input')
        rdist = Normal if self.ar_model else self.reconstruction_dist
        if self.ar_model:
            output_dist = self.decoder_network(input)
            output_dist = LinearReshape(self.decoder_shape, name='lr_reshape')(output_dist)
        else:
            output_dist = self.to_distribution(input, rdist, self.input_shape, self.decoder_network,
                                           True, transform=self.consistency_augmentations,
                                           sample=self.sampled_reconstructions, name='reconstruction')
            if self.glow:
                print('Using glow')
                self.preglow = tf.keras.Model(inputs= input, outputs=output_dist)
                self.glow_model = GLOW(input_shape=self.input_shape, **self.glow_args)
                output_dist = self.glow_model((output_dist))
        self._decoder = Model(inputs=input, outputs=output_dist, name='decoder')

    def build_ar_model(self):
        if self.ar_model == 'pixelcnn':
            ar_net, sample_net, mean_net, decoder_shape = PixelCNNNetwork.get(input_shape=self.input_shape,
                                                               distribution=self.reconstruction_dist, **self.kwargs)
        elif self.ar_model == 'upscaling':
            ar_net, sample_net, mean_net, decoder_shape = UpscalingNetwork.get(input_shape=self.input_shape,
                                                                distribution=self.reconstruction_dist, **self.kwargs)
        self.ar_net = ar_net
        self.ar_sampler = sample_net
        self.ar_mean = mean_net
        self.decoder_shape = decoder_shape

    def build_predictor(self):
        # Create a model to predict labels from encoded space
        input = Input(self.encoded_size, name='predictor_input')
        x = input
        if self.consistency_augmentations:
            x = Lambda(lambda a: a[:, 6:])(x)
        output_dist = self.to_distribution(x, self.predictor_dist, self.label_shape, self.predictor_network, True,
                                           sample=False, name='prediction')
        self._predictor = Model(inputs=input, outputs=output_dist, name='predictor')

    def add_consistency_loss(self, latent_z, latent_z_sample, prediction, input_x, input_y):
        original_latent_z = latent_z
        # Allow for custom sampling of latent space (e.g. spherical, increased variance or VAT)
        if self.custom_sampler:
            latent_z = Lambda(
                lambda z: self.custom_sampler(z, predictor_model=self._predictor, const_loss=self.consistency_loss))(
                latent_z)
            latent_z_sample = Lambda(lambda z: z[0] + 0 * z[1])([latent_z_sample, latent_z])
            latent_z = Lambda(lambda x: tf.stop_gradient(x))(latent_z)
            # prediction = self._predictor(latent_z)
            cst_reconstruction = self._decoder(latent_z)
            if self.ar_model and self.sampled_reconstructions:
                cst_reconstruction = self.ar_sampler(cst_reconstruction)
            elif self.ar_model:
                cst_reconstruction = self.ar_mean(cst_reconstruction)

        # Run the decoder and predictor
        reconstruction_0 = self._decoder(latent_z_sample)
        vae_reconstruction = reconstruction_0
        if self.ar_model:  # PixelCNN Special Case
            vae_reconstruction = self.ar_net([input_x, vae_reconstruction])

        if not self.consistency_loss:
            return None, vae_reconstruction

        if not self.custom_sampler:
            cst_reconstruction = vae_reconstruction

        if self.clip_consistency:
            cst_reconstruction = Lambda(lambda x: tf.stop_gradient(tf.clip_by_value(x.sample(), -1, 1)))(
                cst_reconstruction)

        sample_weight = 1.
        if self.output_iaf:
            cst_reconstruction = ConvolutionalIAFNetwork(cst_reconstruction)
            sample_weight = cst_reconstruction.sample_weight()

        # Re-run through encoder and predictor
        latent_z2 = self._encoder(cst_reconstruction)
        prediction_2 = self._predictor(latent_z2)

        # Add the loss to the training model (scaled by lambda and the consistency weight)
        if not self.consistency_grad:
            prediction = stop_grad_distribution(prediction)

        # Get of mask of labeled examples
        mask = Flatten()(input_y)
        mask = Lambda(lambda x: tf.where(tf.math.is_nan(tf.reduce_sum(x, axis=-1, keepdims=True)), 0., 1.))(mask)

        labeled_loss = mask * nll()(input_y, prediction_2)
        unlabeled_loss = (1. - mask) * self.consistency_loss(prediction, prediction_2)
        loss = sample_weight * (labeled_loss + unlabeled_loss)

        if np.any(self.class_entropy_loss):
            dist = np.ones((input_y.shape[-1])) * self.class_entropy_loss
            dist = dist / np.sum(dist)
            dist = Categorical(probs=tf.convert_to_tensor(dist.astype(np.float32)))

            # Original prediction
            predicted_dist = tf.reduce_sum((1. - mask) * prediction.distribution.probs_parameter(), axis=0)
            predicted_dist = predicted_dist / tf.reduce_sum(predicted_dist)
            predicted_dist = Categorical(probs=predicted_dist)
            loss = loss + self.class_entropy_weight * self.consistency_loss(dist, predicted_dist)

            # Consistency prediction
            predicted_dist = tf.reduce_sum((1. - mask) * prediction_2.distribution.probs_parameter(), axis=0)
            predicted_dist = predicted_dist / tf.reduce_sum(predicted_dist)
            predicted_dist = Categorical(probs=predicted_dist)
            loss = loss + sample_weight * self.class_entropy_weight * self.consistency_loss(dist, predicted_dist)

        try:
            self.const_autoencoder = Model(inputs=[input_x, input_y], outputs=cst_reconstruction)
        except:
            print('Could not make consistancy autoencoder!')
            pass
        loss = tf.where(tf.math.is_finite(loss), loss, 0.)
        return loss, vae_reconstruction

    def build_model(self):
        # Model inputs
        input_x = Input(self.input_shape, name='model_input')
        input_y = Input(self.label_shape, name='label_input')

        # Get the mask of labeled observations and compute balancer weight for VAE terms
        lbal, ubal = (2. / (self.unlabeled_balance + 1)), (2 * self.unlabeled_balance / (self.unlabeled_balance + 1))
        mask = Flatten()(input_y)
        balancer = Lambda(lambda x: tf.where(tf.math.is_nan(tf.reduce_sum(x, axis=-1, keepdims=True)), ubal, lbal))(
            mask)

        # Run the encoder and add KL-divergence
        latent_z, prior_dist, _ = self._encoder_with_prior(input_x)
        latent_z_sample = TrainableKLLoss(weight=self.beta)([latent_z, prior_dist, balancer])

        # Run the decoder and predictor
        prediction = self._predictor(latent_z)
        cst_loss, reconstruction = self.add_consistency_loss(latent_z, latent_z_sample, prediction, input_x, input_y)

        self.model = tf.keras.Model(inputs=[input_x, input_y], outputs=[reconstruction, prediction],
                                      name='training_model')

        # Add the main losses
        #self.model.add_loss(tf.reduce_mean(nll(self.recon_weight * balancer, self.recon_noise)(input_x, reconstruction)))
        #self.model.add_loss(tf.reduce_mean(nll(self.lam)(input_y, prediction)))

        # Add additional losses
        if self.consistency_loss:  # Consistency constraint
            self.model.add_loss(self.alpha * self.lam * K.mean(cst_loss))
        if self.entropy_weight:  # Minimize prediction entropy
            ent_loss = minent(self.entropy_weight * self.lam)(prediction)
            self.model.add_loss(ent_loss)

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss=[nll(self.recon_weight, self.recon_noise, self.flow_base_loss), nll(self.lam)],
                           run_eagerly=self.debug,
                           metrics=[[], self.metric], experimental_run_tf_function=True)


    def llik(self, data, split=None, samples=0):
        # Compute the log-likelihood of held out data
        samples = samples if samples else self.llik_samples
        if not (split is None):
            data = data.get(split)
        X = data.numpy()[0]
        p_z = self.prior(0.)
        p_x = []
        for xi in np.array_split(X, X.shape[0] // 500):
            p_x_chunk = []
            q_z = self._encoder(xi)
            q_z._reparameterize = False
            for s in range(samples):
                z = q_z.sample()
                log_q_z = q_z.log_prob(z)
                log_p_z = p_z.log_prob(z)
                log_p_x = self._decoder(z).log_prob(xi)
                p_x_chunk.append(log_p_x + log_p_z - log_q_z)
            p_x.append(np.stack(p_x_chunk))
        p_x = np.concatenate(p_x, axis=1)
        p_x = logsumexp(p_x, axis=0) - np.log(samples)
        return p_x.sum()

    def consistency_confusion_matrix(self, data, split=None):
        if not (split is None):
            data = data.get(split)
        Y = data.numpy_labels().argmax(axis=1)
        return sk_confusion(self.predict(data),
                            np.argmax(self.latent_predictor.predict(self.encode(data, secondary=True)), axis=-1))

    def build(self, data=None):
        self.setup(data)
        if self.ar_model:
            self.build_ar_model()
        self.build_encoder()
        self.build_decoder()
        self.build_predictor()
        self.build_model()

        # Build the networks used elsewhere
        encoder_input = Input(self.input_shape)
        encoded = Lambda(lambda x: [x.mean(), K.log(x.variance()), x.sample()])(self._encoder(encoder_input))
        self.encoder = tf.keras.Model(inputs=encoder_input, outputs=encoded)
        self.mean_encoder = tf.keras.Sequential([self._encoder, Lambda(lambda x: x.mean())])
        self.sample_encoder = tf.keras.Sequential([self._encoder, Lambda(lambda x: x.sample())])
        try:
            self.mixture_encoder = tf.keras.Sequential(
                [self._encoder, Lambda(lambda x: x.mixture_distribution.probs_parameter())])
        except:
            pass

        try:
            self.decoder = tf.keras.Sequential([self._decoder, Lambda(lambda x: x.mean())])
            self.sample_decoder = tf.keras.Sequential([self._decoder, Lambda(lambda x: x.sample())])

            self.autoencoder = tf.keras.Sequential([self._encoder, self._decoder, Lambda(lambda x: x.mean())])
            self.sample_autoencoder = tf.keras.Sequential([self._encoder, self._decoder, Lambda(lambda x: x.sample())])
        except:
            self.decoder = tf.keras.Sequential([self._decoder, Lambda(lambda x: x)])
            self.sample_decoder = tf.keras.Sequential([self._decoder, Lambda(lambda x: x)])

            self.autoencoder = tf.keras.Sequential([self._encoder, self._decoder, Lambda(lambda x: x)])
            self.sample_autoencoder = tf.keras.Sequential([self._encoder, self._decoder, Lambda(lambda x: x)])

        self.predictor = tf.keras.Sequential(
            [self._encoder, Lambda(lambda x: x.mean()), self._predictor, Lambda(lambda x: x.mean())])
        self.latent_predictor = tf.keras.Sequential([self._predictor, Lambda(lambda x: x.mean())])

        self.sample_prior = lambda n: self.prior(0.).sample(n)

        if not hasattr(self, 'const_autoencoder'):
            self.const_autoencoder = self.autoencoder

        try:
            encoder_input = Input(self.input_shape)
            encoded = self.encoder(self._decoder(self._encoder(encoder_input)))
            self.secondary_encoder = tf.keras.Model(inputs=encoder_input, outputs=encoded)
        except:
            pass

        if self.ar_model:
            lamlayer = Lambda(lambda x: x)
            self.decoder = tf.keras.Sequential([self._decoder, lamlayer, self.ar_sampler])
            self.sample_decoder = tf.keras.Sequential([self._decoder, lamlayer, self.ar_sampler])

            self.autoencoder = tf.keras.Sequential([self._encoder, self._decoder, lamlayer, self.ar_sampler])
            self.sample_autoencoder = tf.keras.Sequential([self._encoder, self._decoder, lamlayer, self.ar_sampler])




