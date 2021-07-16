from tensorflow.keras.layers import Lambda, Add, Input, Dense, Concatenate, Flatten, Reshape, Conv2D, LeakyReLU, \
    Activation
from tensorflow.keras.models import Model
from .core import *


class TwoLayerCPC(CPC):
    def __init__(self, bridge_network=None, predict_from_z1=False, cst_from_prior=False,
                 z1_encoded_size=10, z1_shuffle=False, **kwargs):
        super(TwoLayerCPC, self).__init__(**kwargs)
        self.bridge_network = bridge_network
        self.predict_from_z1 = predict_from_z1
        self.cst_from_prior = cst_from_prior
        self.z1_encoded_size = z1_encoded_size
        self.z1_shuffle = z1_shuffle
        if self.predict_from_z1 and self.cst_from_prior:
            self.cst_from_prior = False
            self.custom_sampler = 'prior'

    def setup(self, data=None):
        if self.is_setup:
            return

        super(TwoLayerCPC, self).setup(data)
        self.gen_bridge = get_bridge_network(self.bridge_network, **self.kwargs)
        self.var_bridge = get_bridge_network(self.bridge_network, **self.kwargs)

    def build_decoder(self):
        # Create the base decoder network
        z1_input = Input(self.z1_encoded_size, name='decoder_input')
        z2_input = Input(self.encoded_size, name='decoder_input_z2')
        rdist = Normal if self.ar_model else self.reconstruction_dist
        input = Concatenate()([z1_input, z2_input])
        if self.ar_model:
            output_dist = self.decoder_network(input)
            output_dist = LinearReshape(self.ar_model.decoder_shape)(output_dist)
        else:
            output_dist = self.to_distribution(input, rdist, self.input_shape, self.decoder_network,
                                               True, transform=self.consistency_augmentations,
                                               sample=self.sampled_reconstructions, name='reconstruction')
        self._decoder = Model(inputs=[z1_input, z2_input], outputs=output_dist, name='decoder')

    def add_consistency_loss(self, latent_z1, latent_z2, latent_z1_sample, latent_z2_sample, prediction, input_x,
                             input_y):
        # Allow for custom sampling of latent space (e.g. spherical, increased variance or VAT)
        if self.custom_sampler or self.cst_from_prior:
            if self.custom_sampler:
                latent_z2 = Lambda(lambda x: self.custom_sampler(x))(latent_z2)
            if self.cst_from_prior:
                latent_z1 = self.z1_decoder(latent_z2)
                if self.z1_shuffle:
                    latent_z1 = Lambda(lambda x: tf.stop_gradient(tf.random.shuffle(x)))(latent_z1)

            latent_z1_sample = Lambda(lambda z: z[0] + 0 * z[1])([latent_z1_sample, latent_z1])
            latent_z2_sample = Lambda(lambda z: z[0] + 0 * z[1])([latent_z2_sample, latent_z2])

            latent_z1 = Lambda(lambda x: tf.stop_gradient(x))(latent_z1)
            latent_z2 = Lambda(lambda x: tf.stop_gradient(x))(latent_z2)
            cst_reconstruction = self._decoder([latent_z1, latent_z2])
            if self.ar_model and self.sampled_reconstructions:
                cst_reconstruction = self.ar_sampler(cst_reconstruction)
            elif self.ar_model:
                cst_reconstruction = self.ar_mean(cst_reconstruction)

        vae_reconstruction_0 = self._decoder([latent_z1_sample, latent_z2_sample])
        vae_reconstruction = vae_reconstruction_0
        if self.ar_model:  # PixelCNN Special Case
            vae_reconstruction = self.ar_net([input_x, vae_reconstruction])

        if not self.consistency_loss:
            return None, vae_reconstruction

        if not self.custom_sampler:
            cst_reconstruction = vae_reconstruction

        if self.clip_consistency:
            cst_reconstruction = Lambda(lambda x: tf.stop_gradient(tf.clip_by_value(x.sample(), -1, 1)))(cst_reconstruction)

        sample_weight = 1.
        if self.output_iaf:
            cst_reconstruction = ConvolutionalIAFNetwork(cst_reconstruction)
            sample_weight = cst_reconstruction.sample_weight()

        # Re-run through encoder and predictor with z2
        consis_z = self._latent_z1_encoder_full(cst_reconstruction) if self.predict_from_z1 else self._encoder(cst_reconstruction)
        prediction_2 = self._predictor(consis_z)

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

        # Run the encoder for z2 and add KL-divergence
        latent_z2, prior_dist_z2, precursor_output = self._encoder_with_prior(input_x)
        latent_z2_sample = TrainableKLLoss(weight=self.beta)([latent_z2, prior_dist_z2, balancer])

        # Create the q(z1|z2,x) network
        latent_z1_precursor = Concatenate()([Flatten()(precursor_output), Flatten()(latent_z2)])
        q_z1_input = Input((latent_z1_precursor.shape[-1],))
        q_z1_output = self.var_bridge(q_z1_input)
        q_z1_output = self.to_distribution(q_z1_output, self.variational_dist, self.z1_encoded_size, name='z1_encoded',
                                           independent=False)
        self.z1_encoder = Model(inputs=q_z1_input, outputs=q_z1_output)

        # Create the p(z1|z2) network
        p_z1_input = Input((latent_z2_sample.shape[-1],))
        p_z1_output = self.gen_bridge(p_z1_input)
        p_z1_output = self.to_distribution(p_z1_output, self.variational_dist, self.z1_encoded_size, name='z1_gen',
                                           independent=False)
        self.z1_decoder = Model(inputs=p_z1_input, outputs=p_z1_output)

        # Apply the networks to get the z1 output
        latent_z1 = self.z1_encoder(latent_z1_precursor)
        self._latent_z1_encoder_full = Model(inputs=input_x, outputs=latent_z1)
        prior_dist_z1 = self.z1_decoder(latent_z2_sample)
        latent_z1_sample = TrainableKLLoss(weight=self.beta)([latent_z1, prior_dist_z1, balancer])

        # Run the decoder and predictor
        prediction = self._predictor(latent_z1 if self.predict_from_z1 else latent_z2)
        cst_loss, reconstruction = self.add_consistency_loss(latent_z1, latent_z2, latent_z1_sample, latent_z2_sample,
                                                             prediction, input_x, input_y)

        self.model = Model(inputs=[input_x, input_y], outputs=[reconstruction, prediction],
                                      name='training_model')

        # Add the main losses
        self.model.add_loss(tf.reduce_mean(nll(self.recon_weight * balancer)(input_x, reconstruction)))
        self.model.add_loss(tf.reduce_mean(nll(self.lam)(input_y, prediction)))

        # Add additional losses
        if self.consistency_loss:  # Consistency constraint
            self.model.add_loss(self.alpha * self.lam * K.mean(cst_loss))
        if self.entropy_weight:  # Minimize prediction entropy
            ent_loss = minent(self.entropy_weight * self.lam)(prediction)
            self.model.add_loss(ent_loss)

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss=[nll(0), nll(0)],
                           run_eagerly=self.debug,
                           metrics=[['mse'], self.metric], experimental_run_tf_function=True)

        # Add additional metrics
        self.model.add_metric(nll()(input_x, reconstruction), aggregation='mean', name='recon_loss')
        self.model.add_metric(nll()(input_y, prediction), aggregation='mean', name='pred_loss')
        try:
            self.model.add_metric(latent_z2.kl_divergence(prior_dist_z2), aggregation='mean', name='kl_loss_z2')
        except:
            pass
        try:
            self.model.add_metric(latent_z1.kl_divergence(prior_dist_z1), aggregation='mean', name='kl_loss_z1')
        except:
            pass
        try:
            self.model.add_metric(cst_loss, aggregation='mean', name='cst_loss')
        except:
            pass
        try:
            self.model.add_metric(minent(1.)(prediction), aggregation='mean', name='ent_loss')
        except:
            pass

        # Make the ridiculous amount of auxiliary models for evaluation
        z_sample = Lambda(lambda x: tf.concat([x[0].sample(), x[1].sample()], axis=1))([latent_z1, latent_z2])
        z_mean = Lambda(lambda x: tf.concat([x[0].mean(), x[1].mean()], axis=1))([latent_z1, latent_z2])
        z_var = Lambda(lambda x: tf.math.log(tf.concat([x[0].variance(), x[1].variance()], axis=1)))(
            [latent_z1, latent_z2])

        self.encoder = tf.keras.Model(inputs=input_x, outputs=[z_mean, z_var, z_sample])
        self.mean_encoder = tf.keras.Model(inputs=input_x, outputs=z_mean)
        self.sample_encoder = tf.keras.Model(inputs=input_x, outputs=z_sample)
        self.secondary_encoder = self.encoder

        decoder_input = Input((self.z1_encoded_size + self.encoded_size))
        split_latent = Lambda(lambda x: [x[:, :self.z1_encoded_size], x[:, self.z1_encoded_size:]])(decoder_input)
        self.dist_decoder = Model(inputs=decoder_input, outputs=self._decoder(split_latent))
        self.decoder = tf.keras.Sequential([self.dist_decoder, Lambda(lambda x: x.mean())])
        self.sample_decoder = tf.keras.Sequential([self.dist_decoder, Lambda(lambda x: x.sample())])

        self.predictor = Model(inputs=input_x, outputs=Lambda(lambda x: x.mean())(prediction))
        latent_predictor_output = self._predictor(split_latent[0] if self.predict_from_z1 else split_latent[1])
        self.latent_predictor = Model(inputs=decoder_input, outputs=Lambda(lambda x: x.mean())(latent_predictor_output))

        self.autoencoder = Model(inputs=input_x, outputs=self.decoder(z_sample))
        self.sample_autoencoder = Model(inputs=input_x, outputs=self.sample_decoder(z_sample))
        if not hasattr(self, 'const_autoencoder'):
            self.const_autoencoder = self.autoencoder

        if self.ar_model:
            lamlayer = Lambda(lambda x: x.mean())
            self.decoder = tf.keras.Sequential([self.dist_decoder, lamlayer, self.ar_sampler])
            self.sample_decoder = tf.keras.Sequential([self.dist_decoder, lamlayer, self.ar_sampler])

            self.autoencoder = Model(inputs=input_x, outputs=self.ar_sampler(lamlayer(self.dist_decoder(z_sample))))
            self.sample_autoencoder = Model(inputs=input_x,
                                            outputs=self.ar_sampler(lamlayer(self.dist_decoder(z_sample))))

    def sample_prior(self, n):
        z_2 = self.prior(0.).sample(n)
        z_1 = self.z1_decoder.predict(z_2)
        return np.concatenate([z_1, z_2], axis=-1)

    def llik(self, data, split=None, samples=0):
        return 0.

    def build(self, data=None):
        self.setup(data)
        if self.ar_model:
            self.build_ar_model()
        self.build_encoder()
        self.build_decoder()
        self.build_predictor()
        self.build_model()
