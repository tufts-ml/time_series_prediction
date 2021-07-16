import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Dropout, Add, Input, Dense, Concatenate, Flatten, Reshape, Conv2D, \
    LeakyReLU, Activation
from pcvae.third_party.keras_patch import ConsistantKModel
import numpy as np
from ..util.distributions import *
from ..util.optimizers import get_optimizer
from ..networks.networks import get_decoder_network, get_encoder_network, get_predictor_network, get_bridge_network
from ..util.lda_distribution import InstantiatedLDA
from .hmm import HMM


class LDA(HMM):
    def __init__(self, input_shape=None, label_shape=None, alpha=0, beta=1, lam=1, recon_weight=1, entropy_weight=0.,
                 states=10, predictor_dist=None, predictor_network=None, topics=10, lda_alpha=1., T=100, nu=0.005,
                 optimizer=None, lda_prior_args=None, consistency_loss=None, debug=False, class_entropy_loss=0,
                 topic_init_var=1e-3,
                 *args, **kwargs):
        super(LDA, self).__init__(*args, **kwargs)

        self.input_shape = input_shape
        self.topics = topics
        self.label_shape = label_shape
        self.alpha = alpha
        self.beta = beta
        self.lam = lam * np.prod(self.input_shape) if self.input_shape else lam
        self.recon_weight = recon_weight
        self.entropy_weight = entropy_weight
        self.states = states
        self.predictor_dist = predictor_dist
        self.predictor_network = predictor_network
        self.kwargs = kwargs
        self.is_setup = False
        self.lda_alpha = lda_alpha
        self.T = T
        self.nu = nu
        self.lda_prior_args = lda_prior_args
        self.consistency_loss = consistency_loss
        self.optimizer = optimizer
        self.debug = debug
        self.metric = []
        self.topic_init_var = topic_init_var
        self.class_entropy_loss = class_entropy_loss


    def setup(self, data=None):
        if self.is_setup:
            return

        if data is None:
            self.reconstruction_dist = self.reconstruction_dist if self.reconstruction_dist else 'Normal'
            self.predictor_dist = self.predictor_dist if self.predictor_dist else 'Categorical'
        else:
            self.input_shape = data.shape()
            self.label_shape = data.dim()
            self.metric = data.get_metrics()
            if hasattr(data, 'predictor_dist') and not self.predictor_dist:
                self.predictor_dist = data.predictor_dist
            else:
                self.predictor_dist = self.predictor_dist if self.predictor_dist else 'Categorical'

        self.dictionary_size = self.input_shape[-1]
        self.predictor_network = get_predictor_network(self.predictor_network, **self.kwargs)
        self.predictor_dist = get_tfd_distribution(self.predictor_dist, **self.kwargs)
        self.optimizer = get_optimizer(self.optimizer, **self.kwargs)
        self.observation_dist = get_tfd_distribution(self.observation_dist)
        self.consistency_loss = get_consistency_loss(self.consistency_loss, **self.kwargs)

    def build_predictor(self):
        # Create a model to predict labels from encoded space
        input = Input((self.topics), name='predictor_input')
        x = input
        output_dist = self.to_distribution(x, self.predictor_dist, self.label_shape, self.predictor_network, True,
                                           sample=False, name='prediction')
        self._predictor = tf.keras.Model(inputs=input, outputs=output_dist, name='predictor')

    def build_lda_model(self):
        # Model inputs
        input_x = Input(self.input_shape, name='lda_input')
        dummy_input = Lambda(lambda x: 0 * tf.reduce_sum(x))(input_x)
        x = Lambda(lambda x: tf.squeeze(x, [1, 2]))(input_x)
        logit_layer = tfpl.VariableLayer((self.topics, self.dictionary_size - 1), trainable=True, dtype=tf.float32, initializer=tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=self.topic_init_var, seed=None
            )
        )
        logits = logit_layer(dummy_input)
        lda = tfpl.DistributionLambda(lambda l: InstantiatedLDA(l, alpha=self.lda_alpha, T=self.T, nu=self.nu))(logits)
        posterior = Lambda(lambda h: h[0].posterior_marginals(h[1]))([lda, x])
        self.topicprobs = tf.keras.Sequential([logit_layer, Lambda(lambda a: tfb.SoftmaxCentered().forward(a))])
        self.lda_model = tf.keras.Model(inputs=input_x, outputs=lda, name='lda_model')
        self.posterior_model = tf.keras.Model(inputs=input_x, outputs=posterior, name='posterior_model')
        self.topics_model = tf.keras.Model(inputs=input_x,
                                           outputs=Lambda(lambda a: tf.expand_dims(tf.nn.softmax(a, axis=-1), 0))(
                                               logits), name='posterior_model')

    def get_topics(self):
        x = np.ones((1,) + tuple(self.input_shape))
        return np.squeeze(self.topics_model.predict(x))

    def build_model(self):
        # Model inputs
        input_x = Input(self.input_shape, name='model_input')
        input_y = Input(self.label_shape, name='label_input')

        x = input_x
        lda = self.lda_model(x)
        log_prob, posterior = Lambda(lambda h: h[0].log_prob_and_marginals(tf.squeeze(h[1], [1, 2])))([lda, x])
        prediction = self._predictor(posterior)

        if self.consistency_loss:
            recon = lda.posterior_samples(x, posterior)
            lda2 = self.lda_model(x)
            posterior2 = Lambda(lambda h: h[0].posterior_marginals(h[1]))([lda2, recon])
            prediction2 = self._predictor(posterior2)

            # Get of mask of labeled examples
            mask = Flatten()(input_y)
            mask = Lambda(lambda x: tf.where(tf.math.is_nan(tf.reduce_sum(x, axis=-1, keepdims=True)), 0., 1.))(mask)

            labeled_loss = mask * nll()(input_y, prediction2)
            unlabeled_loss = (1. - mask) * self.consistency_loss(prediction, prediction2)
            cst_loss = labeled_loss + unlabeled_loss

            if np.any(self.class_entropy_loss):
                dist = np.ones((input_y.shape[-1])) * self.class_entropy_loss
                dist = dist / np.sum(dist)
                dist = tfd.Categorical(probs=tf.convert_to_tensor(dist.astype(np.float32)))
                predicted_dist = tf.reduce_sum((1. - mask) * prediction.distribution.probs_parameter(), axis=0)
                predicted_dist = predicted_dist / tf.reduce_sum(predicted_dist)
                predicted_dist = tfd.Categorical(probs=predicted_dist)
                cst_loss = cst_loss + self.class_entropy_weight * self.consistency_loss(dist, predicted_dist)

        self.model = Model(inputs=[input_x, input_y], outputs=[log_prob, prediction],
                                      name='training_model')

        loss_x = tf.reduce_mean(-(self.recon_weight) * log_prob)
        loss_y = tf.reduce_mean(nll(self.lam)(input_y, prediction))

        self.model.add_loss(loss_x)
        self.model.add_loss(loss_y)
        if self.consistency_loss:
            self.model.add_loss(self.alpha * self.lam * K.mean(cst_loss))

        if hasattr(lda, 'prior_loss') and self.lda_prior_args:
            self.model.add_loss(lda.prior_loss(**self.lda_prior_args))

        def dummy_loss(a, b):
            return 0 * tf.reduce_mean(a) * tf.reduce_mean(b)
        self.model.compile(optimizer=self.optimizer, loss=[dummy_loss, dummy_loss],
                           run_eagerly=self.debug,
                           metrics=[[], self.metric], experimental_run_tf_function=True)

        self.model.add_metric(loss_x, aggregation='mean', name='recon_loss')
        self.model.add_metric(loss_y, aggregation='mean', name='pred_loss')

    def build(self, data=None):
        self.setup(data)
        self.build_lda_model()
        self.build_predictor()
        self.build_model()

        self.predictor = tf.keras.Sequential(
            [self.posterior_model, self._predictor, Lambda(lambda x: x.mean())])
        self.autoencoder = tf.keras.Sequential([Lambda(lambda x: x)])