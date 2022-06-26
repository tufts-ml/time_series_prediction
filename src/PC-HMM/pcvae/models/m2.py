import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from .core import CPC
from ..util.distributions import *
from ..networks.priors import categorical_prior
from pcvae.third_party.keras_patch import ConsistantKModel


class M2(CPC):
    def __init__(self, p_zy_reg=0.0001, **kwargs):
        super(M2, self).__init__(**kwargs)
        self.p_zy_reg = p_zy_reg

    @classmethod
    def name(cls):
        return 'M2'

    def build_encoder(self):
        # Create the base encoder network
        input = Input(self.input_shape, name='encoder_precursor_input')
        label_input = Input(self.label_shape, name='encoder_precursor_label_input')
        output = Concatenate()([Flatten()(input), label_input])
        output = Flatten()(self.encoder_network(output))
        self.encoder_precursor = Model(inputs=[input, label_input], outputs=output, name='encoder_precursor')

        # Create network to map encoder output to latent variational distribution
        latent_input = Input(self.encoder_precursor.output_shape[1:], name='latent_input')
        latent_z = self.to_distribution(latent_input, self.variational_dist, self.encoded_size, name='encoded',
                                        independent=False)

        self.latent_model = Model(inputs=latent_input, outputs=latent_z)

        # Create the full encoder network
        encoder_input = Input(self.input_shape, name='encoder_input')
        encoder_label_input = Input(self.label_shape, name='encoder_label_input')
        precursor_output = self.encoder_precursor([encoder_input, encoder_label_input])
        encoder_output = self.latent_model(precursor_output)

        # Apply the KL divergence from the prior
        self.prior = self.latent_prior(self.encoded_size)  # Latent prior distribution
        prior_dist = self.prior(Lambda(lambda x: K.sum(x))(encoder_input))  # Make prior depend on input

        self._encoder = Model(inputs=[encoder_input, encoder_label_input], outputs=encoder_output, name='encoder')
        self._encoder_with_prior = Model(inputs=[encoder_input, encoder_label_input],
                                         outputs=[encoder_output, prior_dist, precursor_output],
                                         name='encoder_with_prior')

    def build_decoder(self):
        # Create the decoder
        decoder_input = Input(self.encoded_size, name='decoder_precursor_input')
        label_input = Input(self.label_shape, name='decoder_precursor_label_input')
        x = Concatenate()([decoder_input, label_input])
        output_dist = self.to_distribution(x, self.reconstruction_dist, self.input_shape, self.decoder_network,
                                           True, transform=self.consistency_augmentations,
                                           sample=self.sampled_reconstructions, name='reconstruction')
        self._decoder = Model(inputs=[decoder_input, label_input], outputs=output_dist, name='decoder')

    def build_predictor(self):
        # Create a model to predict labels from encoded space
        input = Input(self.input_shape, name='predictor_input')
        x = self.encoder_network(input)
        output_dist = self.to_distribution(x, self.predictor_dist, self.label_shape, self.predictor_network, True,
                                           sample=False, name='prediction')

        self.label_prior = categorical_prior(self.label_shape)  # Latent prior distribution
        prior_dist = self.label_prior(Lambda(lambda x: K.sum(x))(input))  # Make prior depend on input
        self._predictor = Model(inputs=input, outputs=output_dist, name='predictor')
        self._predictor_with_prior = Model(inputs=input,
                                           outputs=[output_dist, prior_dist],
                                           name='predictor_with_prior')

    def build_model(self):

        image_input = Input(self.input_shape, name='model_input')
        label_input = Input(self.label_shape, name='label_input')

        with tf.GradientTape() as g:
            varbl = [l.kernel for l in self._predictor.layers if isinstance(l, Dense)][0]
            g.watch(varbl)

            s_ind = Lambda(lambda x: tf.where(tf.math.is_nan(tf.reduce_sum(x, axis=-1, keepdims=False)), 0.,
                                              tf.reduce_sum(x, axis=-1, keepdims=False)))(label_input)
            label_input_clean = Lambda(lambda x: tf.where(tf.math.is_nan(x), 0., x))(label_input)
            latent_z, prior_dist, _ = self._encoder_with_prior([image_input, label_input_clean])
            latent_z_sample = TrainableKLLoss(weight=0)([latent_z, prior_dist])
            recon = self._decoder([latent_z_sample, label_input_clean])
            prediction, label_prior_dist = self._predictor_with_prior(image_input)

            supervised_klloss = Lambda(lambda x: x[0] * tfd.kl_divergence(x[1], x[2]))([s_ind, latent_z, prior_dist])
            supervised_recon_loss = Lambda(lambda x: x[0] * nll()(x[1], x[2]))(
                [s_ind, image_input, recon])
            supervised_y_loss = Lambda(lambda x: x[0] * nll()(x[1], x[2]))(
                [s_ind, label_input, label_prior_dist])
            supervised_prediction_loss = Lambda(lambda x: x[0] * nll()(x[1], x[2]))(
                [s_ind, label_input, prediction])

            u_ind = Lambda(lambda x: 1. - x)(s_ind)

            unsupervised_klloss = []
            unsupervised_recon_loss = []
            q_y = Lambda(lambda x: tf.reshape(x[0], (-1, 1)) * x[1].mean())([u_ind, prediction])
            for c in range(self.label_shape):
                numpy_c = np.zeros((1, self.label_shape), dtype=np.float32)
                numpy_c[0, c] = 1
                c_in = Lambda(lambda x: 0 * x + K.constant(numpy_c, name='c_constant_' + str(c)))(label_input_clean)
                u_latent_z, u_prior_dist, _ = self._encoder_with_prior([image_input, c_in])
                u_latent_z_sample = TrainableKLLoss(weight=0)([u_latent_z, u_prior_dist])
                u_recon = self._decoder([u_latent_z_sample, c_in])

                unsupervised_klloss.append(Lambda(lambda x: tfd.kl_divergence(x[0], x[1]))([u_latent_z, u_prior_dist]))
                unsupervised_recon_loss.append(Lambda(lambda x: nll()(x[0], x[1]))([image_input, u_recon]))

            unsupervised_klloss = Lambda(lambda x: tf.reduce_sum(tf.stack(x[0], axis=1) * x[1], axis=1))(
                [unsupervised_klloss, q_y])
            unsupervised_recon_loss = Lambda(lambda x: tf.reduce_sum(tf.stack(x[0], axis=1) * x[1], axis=1))(
                [unsupervised_recon_loss, q_y])
            unsupervised_y_loss = Lambda(lambda x: x[0] * x[1].cross_entropy(x[2]))(
                [u_ind, prediction, label_prior_dist])
            unsupervised_prediction_loss = Lambda(lambda x: x[0] * -x[1].entropy())([u_ind, prediction])

            loss = Lambda(lambda x: x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7])([supervised_klloss, supervised_recon_loss, supervised_y_loss, supervised_prediction_loss, unsupervised_klloss, unsupervised_recon_loss, unsupervised_y_loss, unsupervised_prediction_loss])

            # Create the training model
            self.model = Model(inputs=[image_input, label_input], outputs=[recon, prediction, loss],
                                          name='training_model')

            supervised_losses = [(self.lam, supervised_prediction_loss),
                                 (self.beta, supervised_klloss),
                                 (self.recon_weight, supervised_recon_loss),
                                 (1, supervised_y_loss)]

            unsupervised_losses = [(1, unsupervised_prediction_loss),
                                   (self.beta, unsupervised_klloss),
                                   (self.recon_weight, unsupervised_recon_loss),
                                   (1, unsupervised_y_loss)]

        for weight, loss in supervised_losses + unsupervised_losses:
            self.model.add_loss(weight * K.mean(loss))

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss=[nll(0), nll(0), None],
                           run_eagerly=self.debug,
                           metrics=[['mse'], self.metric, []], experimental_run_tf_function=False)


    def conditional_sample(self, Y, threshold=0.9):
        Y = np.squeeze(Y)
        Y = Y if Y.ndim == 1 else Y.argmax(axis=1)
        ynew = np.zeros((Y.shape[0], self.label_shape))
        for c in range(self.label_shape):
            ynew[Y == c, c] = 1
        Y = ynew
        samples = self.sample_prior(Y.shape[0])
        return self._decoder.predict([samples, Y]).mean()

    def build(self, data=None):
        self.setup(data)
        self.build_encoder()
        self.build_decoder()
        self.build_predictor()
        self.build_model()

        # Build the networks used elsewhere
        encoder_input = Input(self.input_shape)
        predicted_label = self._predictor(encoder_input)
        encoded = self._encoder([encoder_input, predicted_label])
        encoded = Lambda(lambda x: [x.mean(), K.log(x.variance()), x.sample()])(encoded)
        self.encoder = tf.keras.Model(inputs=encoder_input, outputs=encoded)
        self.mean_encoder = tf.keras.Model(inputs=encoder_input, outputs=encoded[0])
        self.sample_encoder = tf.keras.Model(inputs=encoder_input, outputs=encoded[2])

        # self.decoder = tf.keras.Sequential([self._decoder, Lambda(lambda x: x.mean())])
        # self.sample_decoder = tf.keras.Sequential([self._decoder, Lambda(lambda x: x.sample())])

        decoded = self._decoder([encoded[0], predicted_label])
        self.autoencoder = tf.keras.Model(inputs=encoder_input, outputs=Lambda(lambda x: x.mean())(decoded))
        decoded = self._decoder([encoded[2], predicted_label])
        self.sample_autoencoder = tf.keras.Model(inputs=encoder_input, outputs=Lambda(lambda x: x.sample())(decoded))

        self.predictor = tf.keras.Sequential([self._predictor, Lambda(lambda x: x.mean())])
        self.sample_prior = lambda n: self.prior(0.).sample(n)


