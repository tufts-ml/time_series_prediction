from tensorflow_probability import layers as tfpl
from tensorflow.keras.layers import Input
from ..util.distributions import *
import tensorflow.keras.backend as K
from tensorflow_probability import bijectors as tfb
import numpy as np

'''
Standard Gaussian prior
'''
def standard_gaussian_prior(encoded_size):
    # Standarad isotropic gaussian prior (not trainable)
    return tf.keras.Sequential([
        tfpl.VariableLayer(encoded_size, trainable=False, dtype=tf.float32),
        tfpl.DistributionLambda(lambda t:
            tfd.MultivariateNormalDiag(loc=t)),
    ])


'''
Categorical prior (for M2)
'''
def categorical_prior(label_size):
    # Standarad isotropic gaussian prior (not trainable)
    return tf.keras.Sequential([
        tfpl.VariableLayer(label_size, trainable=True, dtype=tf.float32),
        tfpl.DistributionLambda(lambda t:
            tfd.Independent(tfd.OneHotCategorical(logits=t))),
    ])

'''
Full-covariance Gaussian mixture prior
'''
def get_regularized_categorical_parameter(input, components, dirichlet_alpha=1., categorical_prior_weight=1., mixture_proportions_trainable=True, **kwargs):
    # Get a categorical probability distribution parameter with a dirichlet prior on weights
    probs = tfpl.VariableLayer(components, dtype=tf.float32, trainable=mixture_proportions_trainable, activation=tf.nn.softmax)(input)
    probs = tf.keras.layers.Lambda(lambda x: x, activity_regularizer=lambda x:
        -categorical_prior_weight * K.sum(tfd.Dirichlet(np.ones(components, dtype=np.float32) * dirichlet_alpha).log_prob(x)))(probs)
    return probs

def get_regularized_mean_parameter(input, encoded_size, components, gauss_scale=None, precision=1., mean_prior_weight=1., **kwargs):
    # Get a multivariate mean parameter with an isotropic gaussian prior
    prior_mean = np.zeros((components, encoded_size), dtype=np.float32)
    prior_scale = np.broadcast_to(np.eye(encoded_size, dtype=np.float32), (components, encoded_size, encoded_size)) #if gauss_scale is None else gauss_scale
    means = tfpl.VariableLayer((components, encoded_size), initializer=tf.keras.initializers.RandomNormal(stddev=5),
                               dtype=tf.float32)(input)
    means = tf.keras.layers.Lambda(lambda x: x, activity_regularizer=lambda x:
        -mean_prior_weight * K.sum(tfd.MultivariateNormalTriL(loc=prior_mean, scale_tril=prior_scale / precision).log_prob(x)))(means)
    return means

def get_regularized_covariance_parameter(input, encoded_size, components, wishart_df=2., wishart_scale=1, covariance_prior_weight=1., **kwargs):
    # Get a multivariate mean parameter with a wishart prior
    covs = tfpl.VariableLayer((components, encoded_size, encoded_size), dtype=tf.float32)(input)
    covs = tf.keras.layers.Lambda(lambda x: tf.linalg.set_diag(x, tf.exp(tf.linalg.diag_part(x)) + 1e-5))(covs)
    covs = tf.keras.layers.Lambda(lambda x: x, activity_regularizer=lambda x:
        -covariance_prior_weight * K.sum(tfd.WishartTriL(df=wishart_df + encoded_size, scale_tril=np.eye(encoded_size, dtype=np.float32) * wishart_scale,
                          input_output_cholesky=True).log_prob(tfb.CholeskyToInvCholesky()(x))))(covs)
    return covs

def mixture_gaussian_prior(encoded_size, components=10, aligned=False, **kwargs):
    # Trainable mixture of gaussians prior
    model_in = Input(tuple())
    vars = get_regularized_covariance_parameter(model_in, encoded_size, components, **kwargs)
    means = get_regularized_mean_parameter(model_in, encoded_size, components, gauss_scale=vars, **kwargs)
    probs = get_regularized_categorical_parameter(model_in, components, **kwargs)

    mixture = AlignedMixture if aligned else tfd.MixtureSameFamily
    dist = tfpl.DistributionLambda(lambda t:
            mixture(mixture_distribution=tfd.Categorical(probs=t[0]),
                                  components_distribution=tfd.MultivariateNormalTriL(loc=t[1], scale_tril=t[2])))([probs, means, vars])
    return tf.keras.Model(inputs=model_in, outputs=dist, name='mixture_prior')

'''
Diagonal-covariance Gaussian mixture prior
'''
def get_regularized_diag_mean_parameter(input, encoded_size, components, gauss_scale=None, precision=1., mean_prior_weight=1., **kwargs):
    # Get a multivariate mean parameter with an isotropic gaussian prior
    prior_mean = np.zeros((components, encoded_size), dtype=np.float32)
    prior_scale = np.broadcast_to(np.ones(encoded_size, dtype=np.float32), (components, encoded_size)) #if gauss_scale is None else gauss_scale
    means = tfpl.VariableLayer((components, encoded_size), initializer=tf.keras.initializers.RandomNormal(stddev=5),
                               dtype=tf.float32)(input)
    means = tf.keras.layers.Lambda(lambda x: x, activity_regularizer=lambda x:
        -mean_prior_weight * K.sum(tfd.Normal(loc=prior_mean, scale=prior_scale / precision).log_prob(x)))(means)
    return means

def get_regularized_diag_covariance_parameter(input, encoded_size, components, log_normal_mean=0., log_normal_scale=1, covariance_prior_weight=1., **kwargs):
    # Get a multivariate mean parameter with a wishart prior
    vars = tfpl.VariableLayer((components, encoded_size), dtype=tf.float32)(input)
    vars = tf.keras.layers.Lambda(lambda x: tf.exp(x) + 1e-5)(vars)
    vars = tf.keras.layers.Lambda(lambda x: x, activity_regularizer=lambda x:
        -covariance_prior_weight * K.sum(tfd.LogNormal(loc=log_normal_mean, scale=log_normal_scale).log_prob(x)))(vars)
    return vars

def mixture_diag_gaussian_prior(encoded_size, components=10, aligned=False, **kwargs):
    # Trainable mixture of gaussians prior
    model_in = Input(tuple())
    vars = get_regularized_diag_covariance_parameter(model_in, encoded_size, components, **kwargs)
    means = get_regularized_diag_mean_parameter(model_in, encoded_size, components, gauss_scale=vars, **kwargs)
    probs = get_regularized_categorical_parameter(model_in, components, **kwargs)

    mixture = AlignedMixture if aligned else tfd.MixtureSameFamily
    dist = tfpl.DistributionLambda(lambda t:
            mixture(mixture_distribution=tfd.Categorical(probs=t[0]),
                                  components_distribution=tfd.Independent(tfd.Normal(loc=t[1], scale=t[2]))))([probs, means, vars])
    return tf.keras.Model(inputs=model_in, outputs=dist, name='mixture_prior')

def get_prior(latent_prior, aligned=False, **kwargs):
    #if callable(latent_prior):
    #    return network
    if latent_prior == 'GaussianMixture':
        return lambda encoded_size: mixture_gaussian_prior(encoded_size, **kwargs)
    elif latent_prior == 'AlignedGaussianMixture':
        return lambda encoded_size: mixture_gaussian_prior(encoded_size, aligned=True, **kwargs)
    elif latent_prior == 'GaussianMixtureDiag':
        return lambda encoded_size: mixture_diag_gaussian_prior(encoded_size, **kwargs)
    elif latent_prior == 'AlignedGaussianMixtureDiag':
        return lambda encoded_size: mixture_diag_gaussian_prior(encoded_size, aligned=True, **kwargs)
    return standard_gaussian_prior