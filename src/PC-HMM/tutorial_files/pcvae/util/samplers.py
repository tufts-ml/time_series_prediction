import tensorflow as tf
from ..third_party.vat_utils import generate_virtual_adversarial_perturbation

def spherical_sample(latent_z, latent_scale=1, **kwargs):
    z_m = latent_z.mean()
    epsilon = tf.random.normal(tf.shape(z_m))
    Y = tf.random.gamma((tf.shape(z_m)[0], 1), 1, 1)
    epsilon = epsilon / tf.sqrt(tf.reduce_sum(epsilon ** 2, axis=1, keepdims=True) + Y)
    return z_m + epsilon * latent_scale

def wide_normal_sample(latent_z, latent_scale=1, **kwargs):
    z_m = latent_z.mean()
    epsilon = tf.random.normal(tf.shape(z_m))
    return z_m + epsilon * latent_scale * latent_z.stddev()

def vat_sample(latent_z, latent_scale=1, predictor_model=None, const_loss=None, **kwargs):
    z_m = latent_z.mean()
    return generate_virtual_adversarial_perturbation(z_m, predictor_model, const_loss, vat_epsilon=latent_scale)

def prior_sample(latent_z, **kwargs):
    z_m = latent_z.mean()
    epsilon = tf.random.normal(tf.shape(z_m))
    return z_m * 0. + epsilon + 0 * latent_z.stddev()

def augment_sample(latent_z, latent_scale=0, **kwargs):
    z_sample = latent_z.sample()
    tdims, qdims = z_sample[:, :6], z_sample[:, 6:]
    tdims = tf.random.normal(tf.shape(tdims)) + 0. * tdims
    if latent_scale:
        epsilon = tf.random.normal(tf.shape(qdims))
        Y = tf.random.gamma((tf.shape(qdims)[0], 1), 1, 1)
        epsilon = epsilon / tf.sqrt(tf.reduce_sum(epsilon ** 2, axis=1, keepdims=True) + Y)
        qdims = qdims + epsilon * latent_scale
    return tf.concat([tdims, qdims], axis=-1)

def get_sampler(custom_sampler, **kwargs):
    if custom_sampler == 'spherical':
        def spherical_func(x, **extra_args):
            kwargs.update(extra_args)
            return spherical_sample(x, **kwargs)
        return spherical_func
    elif custom_sampler == 'wide_normal':
        def wn_func(x, **extra_args):
            kwargs.update(extra_args)
            return wide_normal_sample(x, **kwargs)
        return wn_func
    elif custom_sampler == 'prior':
        def prior_func(x, **extra_args):
            kwargs.update(extra_args)
            return prior_sample(x, **kwargs)
        return prior_func
    elif custom_sampler == 'augment':
        def prior_func(x, **extra_args):
            kwargs.update(extra_args)
            return augment_sample(x, **kwargs)
        return prior_func
    elif custom_sampler == 'vat':
        def vat_func(x, **extra_args):
            kwargs.update(extra_args)
            return vat_sample(x, **kwargs)
        return vat_func

    return None