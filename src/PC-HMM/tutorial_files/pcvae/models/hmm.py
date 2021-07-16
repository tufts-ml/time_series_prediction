import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Dropout, Add, Input, Dense, Concatenate, Flatten, Reshape, Conv2D, \
    LeakyReLU, Activation
from tensorflow.keras.models import Model
from .base import BaseVAE
from pcvae.third_party.keras_patch import ConsistantKModel
import numpy as np
from ..util.distributions import *
from ..util.optimizers import get_optimizer
from ..networks.networks import get_decoder_network, get_encoder_network, get_predictor_network, get_bridge_network
from ..util.hmm_layer import HMMLayer


class HMM(BaseVAE):
    def __init__(self, input_shape=None, label_shape=None, lam=1, recon_weight=1,
                 optimizer=None,
                 states=10, observation_dist=None, predictor_dist=None, predictor_network=None,
                 initial_state_initializer=None, transition_initializer=None, observation_initializer=None,
                 initial_state_alpha=1., transition_alpha=1., observation_prior_loss=None, prior_weight=1.0,
                 debug=False,
                 *args, **kwargs):
        super(HMM, self).__init__(*args, **kwargs)

        self.input_shape = input_shape
        self.label_shape = label_shape
        self.lam = lam * np.prod(self.input_shape) if self.input_shape else lam
        self.recon_weight = recon_weight
        self.states = states
        self.observation_dist = observation_dist
        self.predictor_dist = predictor_dist
        self.predictor_network = predictor_network
        self.kwargs = kwargs
        self.optimizer = optimizer
        self.is_setup = False
        self.observation_dist = observation_dist
        self.initial_state_initializer = initial_state_initializer
        self.transition_initializer = transition_initializer
        self.observation_initializer = observation_initializer
        self.initial_state_alpha = initial_state_alpha
        self.transition_alpha = transition_alpha
        self.observation_prior_loss = observation_prior_loss
        self.metric = []
        self.debug = debug
        self.prior_weight = prior_weight

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

        self.steps = self.input_shape[-2]
        self.predictor_network = get_predictor_network(self.predictor_network, **self.kwargs)
        self.predictor_dist = get_tfd_distribution(self.predictor_dist, **self.kwargs)
        self.optimizer = get_optimizer(self.optimizer, **self.kwargs)
        self.observation_dist = get_tfd_distribution(self.observation_dist)

    def to_distribution(self, input, distribution, shape, network_builder=lambda x: x, independent=True,
                        sample=True, name='dist'):
        # Get the parameter shapes for the target distribution
        output_shapes = distribution.param_static_shapes(shape) if distribution else None

        try:  # Case where network builder can take specified output shapes
            output = network_builder(input, output_shapes=output_shapes)
        except Exception as e:
            output = network_builder(input)

        # Get list of parameters for the distribution
        preconstrained = True
        params, tensors = [], []
        for (param, shape) in output_shapes.items():
            params.append(param)
            if type(output) is dict:
                tensor = output[param]
            else:
                tensor = Reshape(shape)(Dense(shape.num_elements())(output))

            try:
                tensor = distribution.param_constrain(param, tensor)
            except:
                preconstrained = False
            tensors.append(tensor)

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

    def build_predictor(self):
        # Create a model to predict labels from encoded space
        input = Input((self.steps, self.states), name='predictor_input')
        x = input
        output_dist = self.to_distribution(x, self.predictor_dist, self.label_shape, self.predictor_network, True,
                                           sample=False, name='prediction')
        self._predictor = Model(inputs=input, outputs=output_dist, name='predictor')

    def build_hmm_model(self):
        # Model inputs
        input_x = Input(self.input_shape, name='model_input')
        hmm = HMMLayer(self.observation_dist, self.states, initial_state_initializer=self.initial_state_initializer,
                       transition_initializer=self.transition_initializer,
                       observation_initializer=self.observation_initializer,
                       initial_state_alpha=self.initial_state_alpha, transition_alpha=self.transition_alpha,
                       observation_prior_loss=self.observation_prior_loss)(input_x)
        self.hmm_model = tf.keras.Model(inputs=input_x, outputs=hmm, name='hmm_model')

    def build_model(self):
        # Model inputs
        input_x = Input(self.input_shape, name='model_input')
        input_y = Input(self.label_shape, name='label_input')

        hmm = self.hmm_model(input_x)
        prediction = self._predictor(hmm)
        self.model = Model(inputs=[input_x, input_y], outputs=[hmm, prediction],
                           name='training_model')

        self.model.compile(optimizer=self.optimizer, loss=[nll(prior_weight=self.prior_weight), nll(self.lam)],
                           run_eagerly=self.debug,
                           metrics=[[None], self.metric], experimental_run_tf_function=True)

    def build(self, data=None):
        self.setup(data)
        self.build_hmm_model()
        self.build_predictor()
        self.build_model()

        self.predictor = tf.keras.Sequential(
            [self.hmm_model, self._predictor, Lambda(lambda x: x.mean())])
        self.autoencoder = tf.keras.Sequential([Lambda(lambda x: x)])