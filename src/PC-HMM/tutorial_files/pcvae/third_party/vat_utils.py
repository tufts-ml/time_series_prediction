import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Lambda


def generate_virtual_adversarial_perturbation(x, forward, loss, logit=None, vat_xi=1e-6, vat_epsilon=1e-3, scale=1.):
    """Generate an adversarial perturbation.
    Args:
        x: Model inputs.
        logit: Original model output without perturbation.
        forward: Callable which computs logits given input.
        hps: Model hyperparameters.
    Returns:
        Adversarial perturbation to be applied to x.
    """
    if logit is None:
        logit = forward(x)
    d = K.random_normal(shape=K.shape(x))
    for _ in range(1):
        d = vat_xi * get_normalized_vector(d)
        logit_p = logit
        if tf.executing_eagerly():
            with tf.GradientTape() as g:
                g.watch(d)
                logit_m = forward(x + scale  * d)
                dist = loss(logit_p, logit_m)
                dist = K.mean(dist)
            grad = g.gradient(dist, d)
        else:
            logit_m = forward(x + scale * d)
            dist = loss(logit_p, logit_m)
            dist = K.mean(dist)
            grad = K.gradients(dist, [d])[0]
        d = K.stop_gradient(grad)
    return vat_epsilon * get_normalized_vector(scale * d) + x

def get_normalized_vector(d):
    """Normalize d by infinity and L2 norms."""
    d /= 1e-12 + K.max(
        K.abs(d), list(range(1, len(d.get_shape()))), keepdims=True
    )
    d /= K.sqrt(
        1e-6
        + K.sum(
            K.pow(d, 2.0), list(range(1, len(d.get_shape()))), keepdims=True
        )
    )
    return d
