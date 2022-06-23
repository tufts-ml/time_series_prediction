import tensorflow.keras.backend as K
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability import layers as tfpl
import tensorflow as tf
import warnings
# from ..util.util import as_tuple
from tensorflow.keras.layers import Lambda, Dense
import numpy as np
# from ..third_party.convar import ConvolutionalAutoregressiveNetwork
import tensorflow as tf

'''
Functions for computing between-distribution losses
'''
def nll(weight=1., noise=0., include_base_dist_loss=False, prior_weight=False):
    def loglik(x, p_x):
#         mask = tf.reduce_sum(x, axis=tf.range(1, tf.rank(x)))
#         x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
#         x = x + noise * tf.random.normal(tf.shape(x))
        llik = -weight * p_x.log_prob(x)
        if isinstance(p_x, tfd.TransformedDistribution) and include_base_dist_loss:
            llik = llik - weight * p_x.distribution.log_prob(x)
        llik = tf.where(tf.math.is_nan(llik), tf.zeros_like(llik), llik)
        if prior_weight:
            llik = llik + prior_weight * p_x.prior_loss()
        return llik
    return loglik

def nll_data(weight=1., noise=0., include_base_dist_loss=False, prior_weight=False):
    def loglik(x, p_x):
        llik = -weight * p_x.log_prob(x)
        if isinstance(p_x, tfd.TransformedDistribution) and include_base_dist_loss:
            llik = llik - weight * p_x.distribution.log_prob(x)
        if prior_weight:
            llik = llik + prior_weight * p_x.prior_loss()
        return llik/tf.cast(len(x), tf.float32)
    return loglik

def nll_labels(weight=1., noise=0., include_base_dist_loss=False, predictor_weight=False):
    def loglik(x, p_x):
        mask = tf.reduce_sum(x, axis=tf.range(1, tf.rank(x)))# get mask indices
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        llik = -weight * p_x.log_prob(x)
        if isinstance(p_x, tfd.TransformedDistribution) and include_base_dist_loss:
            llik = llik - weight * p_x.distribution.log_prob(x)
        llik = tf.where(tf.math.is_nan(mask*llik), tf.zeros_like(llik), llik)
        if predictor_weight:
            llik = llik + predictor_weight * p_x.prior_loss()
        return llik
    return loglik

def minent(weight=0.):
    return lambda p_x: K.mean(weight * p_x.entropy())

def kl_loss(p1, p2):
    return p1.kl_divergence(p2)

def reverse_kl_loss(p1, p2):
    return p2.kl_divergence(p1)

def reverse_negative_cross_entropy_loss(p1, p2):
    return tf.reduce_sum(p2.probs() - tf.log(1 - p1.probs()), axis=-1)

def cross_entropy_loss(p1, p2):
    return p1.cross_entropy(p2)

def negative_cross_entropy_loss(p1, p2):
    return tf.reduce_sum(p1.probs() - tf.log(1 - p2.probs()), axis=-1)

def reverse_cross_entropy_loss(p1, p2):
    return p2.cross_entropy(p1)

def l2_loss(p1, p2):
    return (p1.mean() - p2.mean()) ** 2

def get_consistency_loss(loss, **kwargs):
    if type(loss) is str:
        return dict(kl=kl_loss, kl_r=reverse_kl_loss,
                    cross_ent=cross_entropy_loss, cross_ent_r=reverse_cross_entropy_loss,
                    neg_cross_ent=negative_cross_entropy_loss, neg_cross_ent_r=reverse_negative_cross_entropy_loss,
                    l2=l2_loss)[loss]
    return loss

'''
Adaptations of distributions from tensorflow_probability. All distributions are modified to have 
unconstrained parameterizations and to have defined and unique parameterizations return from param_static_shapes 
'''

class Normal(tfd.Normal):
    def __init__(self, loc, scale):
        super(Normal, self).__init__(loc=loc,
scale=tf.math.softplus(scale + 1.) + 1e-5)
        tfd.Normal.__init__(self, loc=loc, scale=tf.math.softplus(scale + 1.) + 1e-5)

class NormalWithMissing(tfd.Normal):
    def __init__(self, loc, scale):
        super(NormalWithMissing, self).__init__(loc=loc, scale=tf.math.softplus(scale + 1.) + 1e-5)
#         tfd.Normal.__init__(self, loc=loc, scale=tf.math.softplus(scale + 1.) + 1e-5)        
    
    def log_prob(self, x):
        scale = tf.convert_to_tensor(self.scale)
        
        log_normalization = tf.constant(
            0.5 * np.log(2. * np.pi), dtype=self.dtype) + tf.math.log(scale)
        
        # create mask
        mask = tf.cast(~tf.math.is_nan(x), dtype=self.dtype)
        
        # replace nans in data with 0 
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        
        log_unnormalized = -0.5 * tf.math.squared_difference(
            x / scale, self.loc / scale)
        
        # compute log probability 
        log_probs = log_unnormalized - log_normalization
        
        # return 0 in locations where the observations are missing 
        return mask*log_probs
    
    def prob(self, x):
        scale = tf.convert_to_tensor(self.scale)
        
        log_normalization = tf.constant(
            0.5 * np.log(2. * np.pi), dtype=self.dtype) + tf.math.log(scale)
        
        # create mask
        mask = tf.cast(~tf.math.is_nan(x), dtype=self.dtype)
        
        # replace nans in data with 0 
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        
        log_unnormalized = -0.5 * tf.math.squared_difference(
            x / scale, self.loc / scale)
        
        # compute log probability 
        probs = tf.exp(log_unnormalized - log_normalization) 
        
        return mask*probs
    

    
class FixedNormal(tfd.Normal):
    def __init__(self, loc):
        tfd.Normal.__init__(self, loc=loc, scale=1e-5)

    @staticmethod
    def _param_shapes(sample_shape):
        return {'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

class StandardNormal(tfd.Normal):
    def __init__(self, loc):
        tfd.Normal.__init__(self, loc=loc * 0., scale=1.)

    @staticmethod
    def _param_shapes(sample_shape):
        return {'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

class StudentT(tfd.StudentT):
    def __init__(self, df, loc, scale):
        tfd.StudentT.__init__(self, df=tf.math.softplus(df), loc=loc, scale=tf.math.softplus(scale + 1.) + 1e-5)

class ConstrainedNormal(tfd.Normal):
    def __init__(self, loc, scale):
        tfd.Normal.__init__(self, loc=tf.math.tanh(loc), scale=tf.math.sigmoid(scale + 2) + 1e-3)

class ConstrainedStudentT(tfd.StudentT):
    def __init__(self, df, loc, scale):
        tfd.StudentT.__init__(self, df=tf.math.softplus(df) + 2, loc=tf.math.tanh(loc), scale=tf.math.sigmoid(scale + 2) + 1e-3)

class NormalTanH(tfd.Normal):
    def __init__(self, loc, scale):
        tfd.Normal.__init__(self, loc=loc, scale=tf.math.softplus(scale + 1.) + 1e-5,
                                             bijector=tfb.Affine(-1., 2.))

    def mean(self, name='mean', **kwargs):
        return self.mean_approx()

    @staticmethod
    def _param_shapes(sample_shape):
        return Normal._param_shapes(sample_shape)

class StudentTTanH(tfd.TransformedDistribution):
    def __init__(self, df, loc, scale):
        tfd.TransformedDistribution.__init__(self, Normal(loc, scale), tfb.Tanh())

    @staticmethod
    def _param_shapes(sample_shape):
        return StudentT._param_shapes(sample_shape)

class MultivariateNormalTriL(tfd.MultivariateNormalTriL):
    def __init__(self, loc, scale_tril):
        tfd.MultivariateNormalTriL.__init__(self, loc=loc, scale_tril=tf.linalg.set_diag(scale_tril, tf.exp(tf.linalg.diag_part(scale_tril))))

    @staticmethod
    def _param_shapes(sample_shape):
        return {'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'scale_tril': tf.convert_to_tensor(tuple(sample_shape) + (sample_shape[-1],), dtype=tf.int32)}

class MultivariateNormalDiag(tfd.MultivariateNormalDiag):
    def __init__(self, loc, scale_diag):
        tfd.MultivariateNormalDiag.__init__(self, loc=loc, scale_diag=tf.exp(scale_diag))

    @staticmethod
    def _param_shapes(sample_shape):
        return {'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'scale_diag': tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

class ConstrainedMultivariateNormal(tfd.MultivariateNormalLinearOperator):
    def __init__(self, loc, scale):
        loc = tf.math.tanh(loc)
        scale = tf.linalg.LinearOperatorLowerTriangular(tfb.FillScaleTriL().forward(scale))
        tfd.MultivariateNormalLinearOperator.__init__(self, loc=loc, scale=scale)

    @staticmethod
    def _param_shapes(sample_shape):
        return {'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'scale': tf.convert_to_tensor(tuple(sample_shape[:-1]) + ((sample_shape[-1] * (sample_shape[-1] + 1)) // 2,), dtype=tf.int32)}

class ConstrainedMultivariateStudentT(tfd.MultivariateStudentTLinearOperator):
    def __init__(self, df, loc, scale):
        loc = tf.math.tanh(loc)
        df = tf.math.softplus(df) + 2
        scale = tf.linalg.LinearOperatorLowerTriangular(tfb.FillScaleTriL().forward(scale))
        tfd.MultivariateStudentTLinearOperator.__init__(self, df=df, loc=loc, scale=scale)

    @staticmethod
    def _param_shapes(sample_shape):
        return {'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'df': tf.convert_to_tensor(sample_shape[:-1], dtype=tf.int32),
                'scale': tf.convert_to_tensor(tuple(sample_shape[:-1]) + ((sample_shape[-1] * (sample_shape[-1] + 1)) // 2,), dtype=tf.int32)}


class Bernoulli(tfd.Bernoulli):
    def __init__(self, logits):
        tfd.Bernoulli.__init__(self, logits=logits, dtype=logits.dtype)

    def _mean(self):
        return self.probs_parameter()

    @staticmethod
    def _param_shapes(sample_shape):
        return {'logits': tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

class Categorical(tfd.OneHotCategorical):
    def __init__(self, logits=None, probs=None):
        if not (probs is None):
            probs = probs + 1e-3
            probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
        if not (logits is None):
            logits = tf.where(tf.math.is_nan(logits), 0., logits)
        tfd.OneHotCategorical.__init__(self, logits=logits, probs=probs, dtype=tf.float32)

    def prior_loss(self, alpha=1., weight=1.):
        probs = self.probs_parameter()
        alpha = alpha + 0. * probs
        return tf.reduce_mean(nll(weight)(probs, tfd.Dirichlet(alpha)))

    def _mean(self):
        return self.probs_parameter()

    @staticmethod
    def _param_shapes(sample_shape):
        return {'logits': tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

class IndexedCategorical(tfd.Categorical):
    def __init__(self, logits):
        tfd.Categorical.__init__(self, logits=logits)

    def _mean(self):
        return self.probs_parameter()

    def prior_loss(self, alpha=1., weight=1.):
        probs = self.probs_parameter()
        alpha = alpha + 0. * probs
        return tf.reduce_mean(nll(weight)(probs, tfd.Dirichlet(alpha)))

    @staticmethod
    def _param_shapes(sample_shape):
        return {'logits': tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

class Multinomial(tfd.Multinomial):
    def __init__(self, logits):
        tfd.Multinomial.__init__(self, logits=logits, dtype=logits.dtype)

    def prior_loss(self, alpha=1., weight=1.):
        probs = self.probs_parameter()
        alpha = alpha + 0. * probs
        return tf.reduce_mean(nll(weight)(probs, tfd.Dirichlet(alpha)))

    @staticmethod
    def _param_shapes(sample_shape):
        return {'logits': tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

class PixelCNN(tfd.Distribution):
    def __init__(self, components=10):
        self.components = components
        self.pixcnn = None

    def __call__(self, conditional=None):
        self.conditional = conditional
        return self

    def log_prob(self, value):
        return self.pixcnn.log_prob(value, conditional_input=self.conditional)

    def sample(self, sample_shape=()):
        return self.pixcnn.sample(sample_shape, conditional_input=self.conditional)

    def mean(self):
        return self.pixcnn.mean(conditional_input=self.conditional)

    def param_static_shapes(self, sample_shape):
        if self.pixcnn is None:
            self.pixcnn = tfd.PixelCNN(sample_shape, conditional_shape=(self.components,), high=1., low=-1.)
        return dict(conditional=tf.TensorShape((self.components,)))


class NoiseNormal(tfd.Normal):
    def __init__(self, loc=None, scale=None, mix=None, reparameterize=True, preconstrained=False, dtype=tf.int32,
                 validate_args=False,
                 allow_nan_stats=True, name='NoiseNormal'):
        self.reparameterize = reparameterize
        if not preconstrained:
            loc, scale, mix = NoiseNormal.param_constrain('loc', loc), NoiseNormal.param_constrain('scale',
                                                                                                   scale), NoiseNormal.param_constrain(
                'mix', mix)
        # scale = tf.maximum(tf.math.sqrt(scale), 1e-4)
        self.probs = mix
        self.dist = tfd.TruncatedNormal(loc=loc, scale=scale, low=-1., high=1.)
        tfd.Normal.__init__(self, loc=loc, scale=scale)

    def log_prob(self, value):
        log_prob = tf.math.log(self.probs) + self.dist.log_prob(value)
        log_prob = tf.stack([log_prob, tf.math.log((1. - self.probs) * 0.5) + 0. * log_prob])
        return tf.reduce_logsumexp(log_prob, axis=0)

    def prob(self, value):
        prob = self.dist.prob(value)
        prob = self.probs * prob + (1. - self.probs) * 0.5
        return prob

    def cdf(self, value):
        return self.probs * self.dist.cdf(value) + (1. - self.probs) * tfd.Uniform(low=-1., high=1.).cdf(value)

    def sample(self, **kwargs):
        norm_sample = self.dist.sample(**kwargs)
        uni_sample = tf.random.uniform(tf.shape(norm_sample))
        choice_sample = tf.random.uniform(tf.shape(norm_sample))
        z = tf.stop_gradient(tf.where(choice_sample < self.probs, norm_sample, uni_sample))
        if self.reparameterize:
            z = self._reparameterize_sample(z)
        return tf.clip_by_value(z, -0.999, 0.999)

    def _reparameterize_sample(self, z):
        # Reparameterize sample with implicit reparameterization gradients
        z = tf.stop_gradient(z)
        gradient_val = -self.cdf(z) / tf.stop_gradient(self.prob(z))
        return z + (gradient_val - tf.stop_gradient(gradient_val))

    def mean(self, **kwargs):
        return self.probs * self.dist.mean(**kwargs)

    @staticmethod
    def _param_shapes(sample_shape):
        return {'mix': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'scale': tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

    @classmethod
    def param_constrain(cls, param, tensor):
        if param == 'mix':
            return tf.keras.layers.Lambda(lambda t: tf.clip_by_value(tf.sigmoid(t), 0.0001, 0.9999))(tensor)
        elif param == 'loc':
            return tf.keras.layers.Lambda(lambda t: tf.math.tanh(t))(tensor)
        elif param == 'scale':
            return tf.keras.layers.Lambda(lambda t: (tf.math.sigmoid(t) + 1e-2))(tensor)


class NoiseZCA(tfd.Normal):
    def __init__(self, loc=None, scale=None, mix=None, reparameterize=True, preconstrained=False, dtype=tf.int32,
                 validate_args=False,
                 allow_nan_stats=True, name='NoiseNormal'):
        self.reparameterize = reparameterize
        if not preconstrained:
            loc, scale, mix = NoiseZCA.param_constrain('loc', loc), NoiseZCA.param_constrain('scale', scale), NoiseZCA.param_constrain(
                'mix', mix)
        # scale = tf.maximum(tf.math.sqrt(scale), 1e-4)
        self.probs = mix
        self.dist = tfd.Laplace(loc=loc, scale=scale)
        self.background = tfd.Laplace(0., 0.4)
        tfd.Normal.__init__(self, loc=loc, scale=scale)

    def log_prob(self, value):
        log_prob = tf.math.log(self.probs) + self.dist.log_prob(value)
        log_prob = tf.stack([log_prob, tf.math.log(1. - self.probs) + self.background.log_prob(value) + 0. * log_prob])
        return tf.reduce_logsumexp(log_prob, axis=0)

    def prob(self, value):
        prob = self.dist.prob(value)
        prob = self.probs * prob + (1. - self.probs) * self.background.prob(value)
        return prob

    def cdf(self, value):
        return self.probs * self.dist.cdf(value) + (1. - self.probs) * self.background.cdf(value)

    def sample(self, **kwargs):
        norm_sample = self.dist.sample(**kwargs)
        uni_sample = tf.random.uniform(tf.shape(norm_sample))
        choice_sample = tf.random.uniform(tf.shape(norm_sample))
        z = tf.stop_gradient(tf.where(choice_sample < self.probs, norm_sample, uni_sample))
        if self.reparameterize:
            z = self._reparameterize_sample(z)
        return z

    def _reparameterize_sample(self, z):
        # Reparameterize sample with implicit reparameterization gradients
        z = tf.stop_gradient(z)
        gradient_val = -self.cdf(z) / tf.stop_gradient(self.prob(z))
        return z + (gradient_val - tf.stop_gradient(gradient_val))

    def mean(self, **kwargs):
        return self.probs * self.dist.mean(**kwargs)

    @staticmethod
    def _param_shapes(sample_shape):
        return {'mix': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'scale': tf.convert_to_tensor(sample_shape, dtype=tf.int32),}

    @classmethod
    def param_constrain(cls, param, tensor):
        if param == 'mix':
            return tf.keras.layers.Lambda(lambda t: tf.clip_by_value(tf.sigmoid(t), 0.0001, 0.9999))(tensor)
        elif param == 'loc':
            return tf.keras.layers.Lambda(lambda t: tf.clip_by_value(t, -6, 6))(tensor)
        elif param == 'scale':
            return tf.keras.layers.Lambda(lambda t: (tf.math.sigmoid(t) + 1e-2))(tensor)

class DifferentiableNoiseNormal(object):
    '''
        Convenience class to construct a diagonal-covariance Gaussian mixture distribution with a fixed number of components
        '''

    def __init__(self, components=10, aligned=False):
        pass

    def __call__(self, logits, loc, scale, reference_dist=None):
        probs = tf.clip_by_value(tf.sigmoid(logits), 0.0001, 0.9999)
        probs = tf.stack([probs, 1. - probs], axis=-1)
        loc = tf.stack([tf.math.tanh(loc), 0. * loc], axis=-1)
        scale = tf.stack([tf.math.sigmoid(scale) + 1e-2, 0. * scale + 100], axis=-1)
        return tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),
                       components_distribution=tfd.TruncatedNormal(loc=loc, scale=scale, low=-1., high=1.),
                       reparameterize=True)

    def param_static_shapes(self, sample_shape):
        sample_shape = as_tuple(sample_shape)
        return {'logits': tf.TensorShape(sample_shape),
                'loc': tf.TensorShape(tuple(sample_shape)),
                'scale': tf.TensorShape(tuple(sample_shape))
                }


class NoiseStudentT(tfd.StudentT):
    def __init__(self, loc=None, scale=None, df=None, mix=None):
        self.mix = tf.math.sigmoid(mix)
        tfd.StudentT.__init__(self, loc=tf.math.tanh(loc), df=tf.math.softplus(df) + 1e-3, scale=tf.math.sigmoid(scale) + 1e-3)

    def log_prob(self, value):
        log_prob = tf.math.log(self.mix) + tfd.StudentT.log_prob(self, value)
        log_prob = tf.stack([log_prob, tf.math.log((1. - self.mix) * 0.5)])
        return tf.reduce_logsumexp(log_prob, axis=0)

    def prob(self, value):
        prob = tfd.StudentT.prob(self, value)
        prob = self.mix * prob + (1. - self.mix) * 0.5
        return prob

    def sample(self, **kwargs):
        norm_sample = tfd.StudentT.sample(self, **kwargs)
        uni_sample = tf.random.uniform(tf.shape(norm_sample))
        choice_sample = tf.random.uniform(tf.shape(norm_sample)) < self.mix
        return tf.stop_gradient(tf.where(choice_sample, norm_sample, uni_sample))


    def mean(self):
        return self.mix * tfd.StudentT.mean(self)

    @staticmethod
    def _param_shapes(sample_shape):
        return {'mix': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'loc': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'scale': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                'df': tf.convert_to_tensor(sample_shape, dtype=tf.int32),
                }

class GaussianMixture(object):
    '''
    Convenience class to construct a full-covariance Gaussian mixture distribution with a fixed number of components
    '''
    def __init__(self, components=10, aligned=False):
        self.components = components
        self.aligned = aligned

    def __call__(self, logits, loc, scale_tril):
        mixture = AlignedMixture if self.aligned else tfd.MixtureSameFamily
        return mixture(mixture_distribution=tfd.Categorical(logits=logits),
                                              components_distribution=MultivariateNormalTriL(loc=loc, scale_tril=scale_tril),
                                              reparameterize=True)

    def param_static_shapes(self, sample_shape):
        sample_shape = as_tuple(sample_shape)
        return {'logits': tf.TensorShape(tuple(sample_shape)[:-1] + (self.components,)),
                'loc': tf.convert_to_tensor(tuple(sample_shape)[:-1] + (self.components,) + (sample_shape[-1],)),
                'scale_tril': tf.convert_to_tensor(tuple(sample_shape)[:-1] + (self.components,) + (sample_shape[-1], sample_shape[-1]))
                }

class ConstrainedGaussianMixture(object):
    '''
    Convenience class to construct a full-covariance Gaussian mixture distribution with a fixed number of components
    '''
    def __init__(self, components=10, aligned=False):
        self.components = components
        self.aligned = aligned

    def __call__(self, logits, loc, scale):
        mixture = AlignedMixture if self.aligned else tfd.MixtureSameFamily
        return mixture(mixture_distribution=tfd.Categorical(logits=logits),
                                              components_distribution=ConstrainedMultivariateNormal(loc=loc, scale_tril=scale),
                                              reparameterize=True)

    def param_static_shapes(self, sample_shape):
        sample_shape = as_tuple(sample_shape)
        return {'logits': tf.TensorShape(tuple(sample_shape)[:-1] + (self.components,)),
                'loc': tf.convert_to_tensor(tuple(sample_shape)[:-1] + (self.components,) + (sample_shape[-1],)),
                'scale': tf.convert_to_tensor(tuple(sample_shape)[:-1] + (self.components,) + ((sample_shape[-1] * (sample_shape[-1] - 1)) // 2,))
                }

class ConstrainedNormalMixtureWithUniform(object):
    '''
    Convenience class to construct a full-covariance Gaussian mixture distribution with a fixed number of components
    '''
    def __init__(self):
        pass

    def __call__(self, mix, loc, scale):
        mix = tf.math.sigmoid(mix)
        mix = tf.concat([mix, 1. - mix], axis=-1)
        return tfd.Mixture(cat=tfd.Categorical(probs=mix),
                           components=[ConstrainedNormal(loc=loc, scale_tril=scale), tfd.Uniform(-1., 1.)])

    def param_static_shapes(self, sample_shape):
        sample_shape = as_tuple(sample_shape)
        return {'mix': tf.TensorShape(tuple(sample_shape)[:-1] + (1,)),
                'loc': tf.convert_to_tensor(tuple(sample_shape)[:-1] + (sample_shape[-1],)),
                'scale': tf.convert_to_tensor(tuple(sample_shape)[:-1] + (sample_shape[-1],))
                }

class ConstrainedStudentTMixture(object):
    '''
    Convenience class to construct a full-covariance Gaussian mixture distribution with a fixed number of components
    '''
    def __init__(self, components=10, aligned=False):
        self.components = components
        self.aligned = aligned

    def __call__(self, logits, loc, scale):
        mixture = AlignedMixture if self.aligned else tfd.MixtureSameFamily
        return mixture(mixture_distribution=tfd.Categorical(logits=logits),
                                              components_distribution=ConstrainedMultivariateStudentT(loc=loc, scale_tril=scale),
                                              reparameterize=True)

    def param_static_shapes(self, sample_shape):
        sample_shape = as_tuple(sample_shape)
        return {'logits': tf.TensorShape(tuple(sample_shape)[:-1] + (self.components,)),
                'loc': tf.convert_to_tensor(tuple(sample_shape)[:-1] + (self.components,) + (sample_shape[-1],)),
                'scale': tf.convert_to_tensor(tuple(sample_shape)[:-1] + (self.components,) + ((sample_shape[-1] * (sample_shape[-1] - 1)) // 2,))
                }

class GaussianMixtureDiag(object):
    '''
    Convenience class to construct a diagonal-covariance Gaussian mixture distribution with a fixed number of components
    '''
    def __init__(self, components=10, aligned=False):
        self.components = components
        self.aligned = aligned

    @classmethod
    def align(cls, posterior, prior):
        if (not isinstance(posterior, AlignedMixture)) or (not isinstance(posterior, AlignedMixture)):
            return posterior
        output = GaussianMixtureDiag(aligned=True)(logits=posterior.mixture_distribution.logits,
                                                 loc=posterior.components_distribution.distribution.mean() +
                                                     prior.components_distribution.distribution.mean(),
                                                 scale=tf.math.log(posterior.components_distribution.distribution.stddev()) +
                                                       tf.math.log(prior.components_distribution.distribution.stddev())
                                                 )
        return output

    def __call__(self, logits, loc, scale, reference_dist=None):
        mixture = AlignedMixture if self.aligned else tfd.MixtureSameFamily
        return mixture(mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.Independent(Normal(loc=loc, scale=scale), reinterpreted_batch_ndims=1),
                                  reparameterize = False)

    def param_static_shapes(self, sample_shape):
        sample_shape = as_tuple(sample_shape)
        return {'logits': tf.TensorShape(tuple(sample_shape)[:-1] + (self.components,)),
                'loc': tf.TensorShape(tuple(sample_shape)[:-1] + (self.components,) + (sample_shape[-1],)),
                'scale': tf.TensorShape(tuple(sample_shape)[:-1] + (self.components,) + (sample_shape[-1],))
                }

class QuantizedDistribution(tfd.QuantizedDistribution):
    def mean(self):
        return self.distribution.mean()

class QuantizedMixture(tfd.MixtureSameFamily):
    def log_prob(self, value):
      value = tf.math.floor((value + 1.) * 128.)
      return tfd.MixtureSameFamily.log_prob(self, value)

    def prob(self, value):
      value = tf.math.floor((value + 1.) * 128.)
      return tfd.MixtureSameFamily.prob(self, value)

    def sample(self, **kwargs):
        return tfd.MixtureSameFamily.sample(self, **kwargs) / 128. - 1.

    def mean(self):
        return tfd.MixtureSameFamily.mean(self) / 128. - 1.

class LogisticMixtureQuantized(object):
    '''
    Convenience class to construct a diagonal-covariance Gaussian mixture distribution with a fixed number of components
    '''
    def __init__(self, components=10, aligned=False):
        self.components = components

    def __call__(self, logits, loc, scale, reference_dist=None):
        logits = tf.reshape(logits, tf.shape(loc)[:-1])
        discretized_logistic_dist = QuantizedDistribution(
            distribution=tfd.TransformedDistribution(
                distribution=tfd.Logistic(loc=128 * tf.math.tanh(loc) + 128, scale=10* tf.math.softplus(scale + 1.) + 1e-5),
                bijector=tfb.AffineScalar(shift=-0.5)),
            low=-1.,
            high=257.)
        discretized_logistic_dist = tfd.Independent(discretized_logistic_dist, reinterpreted_batch_ndims=1)
        mixture_dist = QuantizedMixture(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=discretized_logistic_dist,
            reparameterize=False
        )

        return mixture_dist

    def param_static_shapes(self, sample_shape):
        sample_shape = as_tuple(sample_shape)
        return {'logits': tf.TensorShape(tuple(sample_shape)[:-1] + (self.components,)),
                'loc': tf.TensorShape(tuple(sample_shape)[:-1] + (self.components,) + (sample_shape[-1],)),
                'scale': tf.TensorShape(tuple(sample_shape)[:-1] + (self.components,) + (sample_shape[-1],))
                }

class AlignedMixture(tfd.MixtureSameFamily):
    '''
    Subclass of MixtureSameFamily that allows for analytical computation of KL divergence, under the assumption
    that there is a one-to-one correspondence between components.
    '''
    def __init__(self, *args, **kwargs):
        super(AlignedMixture, self).__init__(*args, **kwargs)


@kullback_leibler.RegisterKL(AlignedMixture, AlignedMixture)
def _kl_aligned_mixture(d1, d2, name=None):
    with tf.name_scope(name or 'kl_aligned_mixture'):
        entropy_term = d1.mixture_distribution.entropy() + \
             tf.reduce_sum(d1.mixture_distribution.probs_parameter() * d1.components_distribution.entropy(), axis=-1)
        mixture_cross_entropy = d1.mixture_distribution.cross_entropy(d2.mixture_distribution)
        component_cross_entropy = d1.components_distribution.cross_entropy(d2.components_distribution)
        component_cross_entropy = tf.reduce_sum(d1.mixture_distribution.probs_parameter() * component_cross_entropy, axis=-1)
        return -entropy_term + mixture_cross_entropy + component_cross_entropy


def get_tfd_distribution(dist, components=1, aligned=False, **kwargs):
    if callable(dist):
        return dist
    dist_dict = dict(MultivariateNormalTriL=MultivariateNormalTriL, MultivariateNormalDiag=MultivariateNormalDiag,
                     Categorical=Categorical, Multinomial=Multinomial, Bernoulli=Bernoulli, NoiseNormal=NoiseNormal, NoiseZCA=NoiseZCA,
                     Normal=Normal, StandardNormal=StandardNormal, StudentT=StudentT, NoiseStudentT=NoiseStudentT, NormalTanH=NormalTanH, StudentTTanH=StudentTTanH, NormalWithMissing=NormalWithMissing,
                     ConstrainedMultivariateStudentT=ConstrainedMultivariateStudentT, ConstrainedMultivariateNormal=ConstrainedMultivariateNormal,
                     ConstrainedNormal=ConstrainedNormal, ConstrainedStudentT=ConstrainedStudentT, ConstrainedNormalMixtureWithUniform=ConstrainedNormalMixtureWithUniform)
    mixture_dist_dict = dict(GaussianMixture=GaussianMixture, GaussianMixtureDiag=GaussianMixtureDiag,
                             AlignedGaussianMixture=GaussianMixture, AlignedGaussianMixtureDiag=GaussianMixtureDiag,
                             LogisticMixtureQuantized=LogisticMixtureQuantized, ConstrainedStudentTMixture=ConstrainedStudentTMixture,
                             ConstrainedGaussianMixture=ConstrainedGaussianMixture, DifferentiableNoiseNormal=DifferentiableNoiseNormal,
                             )
    aligned = aligned or (type(dist) is str and 'Aligned' in dist)
    if type(dist) is str:
        if dist in dist_dict:
            return dist_dict[dist]
        elif dist in mixture_dist_dict:
            return mixture_dist_dict[dist](components=components, aligned=aligned)
        elif dist == 'pixelcnn':
            return PixelCNN(components=components)
        return getattr(tfd, dist)
    return dist

class TrainableKLLoss(tf.keras.layers.Layer):
    def __init__(self, weight=1., **kwargs):
        super(TrainableKLLoss, self).__init__(**kwargs)
        self.weight = weight

    def call(self, inputs):
        if len(inputs) == 3: # Case with balancer b/w labeled and unlabeled
            distribution_a, distribution_b, balancer = tuple(inputs)
        else:
            distribution_a, distribution_b, balancer = tuple(inputs) + (1.,)

        z = distribution_a.sample()
        try:
            loss = tfd.kl_divergence(distribution_a, distribution_b)
        except NotImplementedError:
            warnings.warn(
                'Exact KL not supported between distribution types: (%s, %s). Using sampled KL-divergence...' %
                (str(type(distribution_a)), str(type(distribution_b))))
            loss = distribution_a.log_prob(z) - distribution_b.log_prob(z)
        self.add_loss(tf.reduce_mean(self.weight * balancer * loss))
        return z

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def stop_grad_distribution(distribution):
    output_args = {}
    for param, value in distribution.parameters.items():
        if isinstance(value, tfd.Distribution):
            output_args[param] = stop_grad_distribution(value)
        elif param in ['validate_args', 'name', 'allow_nan_stats', 'dtype', 'probs']:
            pass
        else:
            try:
                output_args[param] = tf.stop_gradient(value)
            except:
                output_args[param] = value
    return type(distribution)(**output_args)

class FixedSample(tfd.Normal):
    def __init__(self, sample=None, lp=None, dtype=tf.float32,
                 validate_args=False,
                 allow_nan_stats=True, name='Fixed'):
        self.smpl = sample
        self.lp = lp
        tfd.Normal.__init__(self, sample, 0)

    def log_prob(self, x):
        return self.lp + 0. * tf.reduce_sum(x),

def IAFNetwork(dist, condition, iaf_layers=2, netargs=dict()):
    '''
    Implementation of "Improved Variational Inference with Inverse Autoregressive Flow"

    See algorithm (1), computes a sample and its log-probability given a (diagonal-Gaussian) distribution.
    Returns a fake distribution, caching the sample and probability
    '''

    args = dict(activation='elu', hidden_units=[200, 200])
    args.update(netargs)

    sample = Lambda(lambda d: d.sample())(dist)
    log_prob = Lambda(lambda dx: dx[0].log_prob(dx[1]))((dist, sample))

    ndims = sample.shape[-1]
    condition = Dense(ndims)(condition)

    for l in range(iaf_layers):
        order = ['left-to-right', 'right-to-left'][l % 2]
        network = tfb.AutoregressiveNetwork(2, event_shape=ndims, conditional=True, input_order=order,
                                            conditional_event_shape=ndims, name='IAF_l%d' % l, **args)
        zl = network(sample, condition)
        m = Lambda(lambda x: x[:, :, 0])(zl)
        s = Lambda(lambda x: tf.math.sigmoid(x[:, :, 1] + 2.))(zl)
        sample = Lambda(lambda x: x[2] * x[0] + (1. - x[2]) * x[1])([sample, m, s])

        if len(log_prob.shape) == 1:
            log_prob = Lambda(lambda x: x[0] - tf.reduce_sum(tf.math.log(x[1]), axis=-1))([log_prob, s])
        else:
            log_prob = Lambda(lambda x: x[0] - tf.math.log(x[1]))([log_prob, s])

    return tfpl.DistributionLambda(lambda s_pl: FixedSample(*s_pl))([sample, log_prob])


class IAFWrappedSample(tfd.Normal):
    def __init__(self, sample=None, lp=None, base_dist=None, dtype=tf.float32,
                 validate_args=False,
                 allow_nan_stats=True, name='Fixed'):
        self.smpl = sample
        self._sample_weight = tf.exp(base_dist.log_prob(sample) - lp)
        self.base_dist = base_dist
        tfd.Normal.__init__(self, sample, 0)

    def log_prob(self, x):
        return self.base_dist.log_prob(x)

    def mean(self):
        return self.base_dist.mean()

    def sample_weight(self):
        return self._sample_weight

def ConvolutionalIAFNetwork(dist, iaf_layers=4, netargs=dict()):
    '''
    Implementation of "Improved Variational Inference with Inverse Autoregressive Flow"

    See algorithm (1), computes a sample and its log-probability given a (diagonal-Gaussian) distribution.
    Returns a fake distribution, caching the sample and probability
    '''

    args = dict(activation='elu', hidden_units=[50, 50])
    args.update(netargs)

    sample = Lambda(lambda d: d.sample())(dist)
    log_prob = Lambda(lambda dx: dx[0].log_prob(dx[1]))((dist, sample))

    for l in range(iaf_layers):
        order = ['left-to-right', 'right-to-left'][l % 2]
        network = ConvolutionalAutoregressiveNetwork(2, event_shape=sample.shape[-3:], kernel_shape=(7, 7), input_order=order **args)
        zl = network(sample)
        m = Lambda(lambda x: x[:, :, :, :, 0])(zl)
        s = Lambda(lambda x: tf.math.sigmoid(x[:, :, :, :, 1] + 2.))(zl)
        sample = Lambda(lambda x: x[2] * x[0] + (1. - x[2]) * x[1])([sample, m, s])

        if len(log_prob.shape) == 1:
            log_prob = Lambda(lambda x: x[0] - tf.reduce_sum(tf.math.log(x[1]), axis=[-1, -2, -3]))([log_prob, s])
        elif len(log_prob.shape) == 3:
            log_prob = Lambda(lambda x: x[0] - tf.reduce_sum(tf.math.log(x[1]), axis=-1))([log_prob, s])
        else:
            log_prob = Lambda(lambda x: x[0] - tf.math.log(x[1]))([log_prob, s])

    return tfpl.DistributionLambda(lambda s_pl: IAFWrappedSample(*s_pl))([sample, log_prob, dist])

class ConvolutionalIAF(object):
    def __init__(self, base, iaf_layers=4, netargs=dict()):
        self.base = base
        self.iaf_layers = iaf_layers
        self.netargs = netargs

    def __call__(self, **kwargs):
        params, inputs = [], []
        for (param, tensor) in kwargs.items():
            params.append(param)
            inputs.append(tensor)
        return self.iaf_model(inputs)

    def get_trainable_weights(self):
        return self.iaf_model.trainable_weights

    def param_static_shapes(self, sample_shape):
        shapes = self.base.param_static_shapes(sample_shape)
        params, inputs, test_input = [], [], []
        for (param, outshape) in shapes.items():
            params.append(param)
            inputs.append(tf.keras.layers.Input(outshape))
            test_input.append(np.random.random((1,) + outshape, dtype=np.float32))

        # Create the distribution object
        def distribution_lambda(tensors_in):
            return tfd.Independent(self.base(
                **{p: vt for (p, vt) in zip(params, tensors_in)}))

        output_dist = tfpl.DistributionLambda(distribution_lambda)(inputs)
        output_dist = ConvolutionalIAFNetwork(output_dist, self.iaf_layers, self.netargs)
        self.iaf_model = tf.keras.Model(inputs, output_dist)
        assert self.iaf_model(test_input).shape[1:] == sample_shape
        return shapes






