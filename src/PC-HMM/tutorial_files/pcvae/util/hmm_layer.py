import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import prefer_static as ps
from ..networks.wrn import residual_block
import math
from tensorflow_probability import distributions as tfd
from .distributions import IndexedCategorical


class HiddenMarkovModel(tfd.HiddenMarkovModel):
    '''
    Essentially the same as the tensorflow-probability HMM, but encapsulates a set of observations.
    Implicit conversion to tensor will use these observations to compute posterior marginals and
    return as a tensor.

    Also adds a 'prior_loss' function to compute regularizers for the component parameters.
    '''
    def __init__(self, initial_distribution, transition_distribution, observation_distribution, num_steps, observations,
                 initial_state_alpha=1., transition_alpha=1., observation_prior_loss=None,
                 **kwargs):
        self.observations = observations
        self.initial_state_alpha = initial_state_alpha
        self.transition_alpha = transition_alpha
        self.observation_prior_loss = observation_prior_loss
        super(HiddenMarkovModel, self).__init__(initial_distribution, transition_distribution, observation_distribution, num_steps, **kwargs)

    def posterior_marginals(self, observations=None, mask=None, name='posterior_marginals'):
        if observations is None:
            observations = self.observations
        return super(HiddenMarkovModel, self).posterior_marginals(observations, mask=mask, name=name).probs_parameter()

    def prior_loss(self):
        loss = self.initial_distribution.prior_loss(self.initial_state_alpha)
        loss = loss + self.transition_distribution.prior_loss(self.transition_alpha)
        if self.observation_prior_loss:
            loss = loss + self.observation_prior_loss(self.observation_distribution)
        return loss


class HMMLayer(tfp.layers.DistributionLambda):
    '''
    A trainable Keras layer that encapsulates all the parameters for an HMM and returns an HMM distribution.

    TODO: Make this support masking for variable length sequences
    '''
    def __init__(self, observation_distribution, states,
                 initial_state_initializer=None, transition_initializer=None, observation_initializer=None,
                 initial_state_alpha=1., transition_alpha=1., observation_prior_loss=None, **kwargs):
        self.observation_distribution = observation_distribution
        self.states = states
        self.steps = 0
        self.initial_state_initializer = initial_state_initializer
        self.transition_initializer = transition_initializer
        self.observation_initializer = observation_initializer
        self.initial_state_alpha = initial_state_alpha
        self.transition_alpha = transition_alpha
        self.observation_prior_loss = observation_prior_loss
        super(HMMLayer, self).__init__(self.get_distribution, HiddenMarkovModel.posterior_marginals)

    def get_initializer(self, initializer, name=None):
        # Check if a tensor, function or keras initializer name was given for a parameter
        try:
          if name in initializer.keys():
            initializer = initializer[name]
          else:
            initializer = None
        except:
          pass

        if initializer is None:
            return 'normal'
        try:
            initial_value = tf.convert_to_tensor(initializer)
            def init(shape, dtype=tf.float32):
                return tf.cast(initial_value, dtype=dtype)
            return init
        except:
            return initializer


    def build(self, input_shape):
        # Set number of timesteps in chain
        self.steps = input_shape[-2]

        # Initialize the initial state distribution variables
        self.initial_distribution_param = self.add_weight(shape=(self.states,),
                                                          initializer=self.get_initializer(self.initial_state_initializer),
                                                          name='hmm_initial_state',
                                                          trainable=True)

        # Initialize the transition distribution variables
        self.transition_distribution_param = self.add_weight(shape=(self.states, self.states),
                                                          initializer=self.get_initializer(
                                                              self.transition_initializer),
                                                          name='hmm_transition',
                                                          trainable=True)

        # Get observation distribution parameters
        observation_param_spec = self.observation_distribution.param_static_shapes((self.states, input_shape[-1]))
        self.observation_params = []
        self.observation_param_names = []
        for name, shape in observation_param_spec.items():
            param = self.add_weight(shape=shape,
                                                          initializer=self.get_initializer(
                                                              self.observation_initializer, name),
                                                          name='hmm_' + name,
                                                          trainable=True)
            self.observation_params.append(param)
            self.observation_param_names.append(name)

    def get_distribution(self, inputs):
        # Make the HMM distribution
        inputs = tf.squeeze(inputs, axis=[1])
        initial_distribution = IndexedCategorical(logits=self.initial_distribution_param)
        transition_distibution = IndexedCategorical(logits=self.transition_distribution_param)
        observation_distribution = self.observation_distribution(**dict(zip(self.observation_param_names, self.observation_params)))
        observation_distribution = tfd.Independent(observation_distribution)
        return HiddenMarkovModel(initial_distribution, transition_distibution, observation_distribution, self.steps, inputs,
                                 initial_state_alpha=self.initial_state_alpha, transition_alpha=self.transition_alpha,
                                 observation_prior_loss=self.observation_prior_loss)