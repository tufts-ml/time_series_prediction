from .distributions import *
import tensorflow.keras.backend as K
from tensorflow_probability.python.internal import reparameterization


class InstantiatedLDA(tfd.Multinomial):

    def __init__(self, phi, alpha=1.1, T=100, nu=0.005, dtype=tf.float32, validate_args=False, allow_nan_stats=True, words_per_sample=200,
                 sample_temp=0.1, nef_topics=True,
                 name='LDA'):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            self.phi_logits = phi
            self.phi = tfb.SoftmaxCentered().forward(phi)
            self.alpha = alpha
            self.topics = self.phi.shape[0]
            self.T = T
            self.nu = nu
            self.check_grad = False
            self.pi_min_mass_preserved = 0.25
            self.pi_decay_rate = 0.75
            self.words_per_sample = words_per_sample
            self.sample_temp = sample_temp
            self.nef_topics = nef_topics
            super(InstantiatedLDA, self).__init__(
                10, logits=tf.ones(2))

    def infer_pi(self, x):
        pi_init = 0. * tf.matmul(x, self.phi, transpose_b=True) + 1
        pi_init = pi_init / tf.reduce_sum(pi_init, axis=-1, keepdims=True)
        counts = tf.reduce_sum(x, axis=-1)
        concentration = tf.ones_like(pi_init) * self.alpha
        step_size = self.nu * tf.ones_like(counts)

        def forward(pi):
            mult_param = tf.matmul(pi, self.phi)
            loss = tfd.Multinomial(total_count=counts, probs=mult_param).log_prob(x)
            loss = loss + tfd.Dirichlet(concentration).log_prob(pi)
            return -tf.reduce_sum(loss)

        def compute_grad(pi):
            mult_param = tf.matmul(pi, self.phi)  # N x D
            return tf.matmul(x / mult_param, self.phi, transpose_b=True) + (self.alpha / (1e-9 + pi))

        def update(inputs, *args):
            pi, nu = inputs
            grad = compute_grad(pi)
            if self.check_grad:
                if tf.executing_eagerly():
                    with tf.GradientTape() as g:
                        g.watch(pi)
                        loss = forward(pi)
                    cgrad = g.gradient(loss, pi)
                else:
                    loss = forward(pi)
                    cgrad = K.gradients(loss, [pi])[0]

            grad = tf.expand_dims(nu, 1) * grad
            grad = grad - tf.reduce_max(grad, axis=-1, keepdims=True)

            pi_new = pi * tf.exp(grad)
            pi_sum = tf.reduce_sum(pi_new, axis=-1, keepdims=True)

            pi_check = pi_sum <= self.pi_min_mass_preserved
            pi_new = pi_new / pi_sum
            nu = tf.where(tf.squeeze(pi_check, axis=1), self.pi_decay_rate * nu, nu)
            pi = tf.where(pi_check, pi, pi_new)
            return (pi, nu)

        return tf.foldl(update, tf.zeros(self.T), (pi_init, step_size))[0]

    def infer_pi2(self, x):
        pi_init = 0. * tf.matmul(x, self.phi, transpose_b=True) + 1
        pi_init = pi_init / tf.reduce_sum(pi_init, axis=-1, keepdims=True)
        counts = tf.reduce_sum(x, axis=-1)
        concentration = tf.ones_like(pi_init) * self.alpha

        def forward(pi):
            mult_param = tf.matmul(pi, self.phi)
            loss = tfd.Multinomial(total_count=counts, probs=mult_param).log_prob(x)
            loss = loss + tfd.Dirichlet(concentration).log_prob(pi)
            return tf.reduce_mean(loss)

        def update(pi, *args):
            if tf.executing_eagerly():
                with tf.GradientTape() as g:
                    g.watch(pi)
                    loss = forward(pi)
                grad = g.gradient(loss, pi)
            else:
                loss = forward(pi)
                grad = K.gradients(loss, [pi])[0]

            grad = self.nu * grad
            grad = grad - tf.reduce_max(grad, axis=-1, keepdims=True)
            pi = pi * tf.exp(grad)
            return pi / tf.reduce_sum(pi, axis=-1, keepdims=True)

        return tf.foldl(update, tf.zeros(self.T), pi_init)

    def posterior_marginals(self, x):
        return self.infer_pi(x)

    def log_prob(self, x, pi=None):
        if pi is None:
            pi = self.infer_pi(x)
        counts = tf.reduce_sum(x, axis=-1)
        concentration = tf.ones_like(pi) * self.alpha
        mult_param = tf.matmul(pi, self.phi)
        loss = tfd.Multinomial(total_count=counts, probs=mult_param).log_prob(x)
        loss = loss + tfd.Dirichlet(concentration).log_prob(pi)
        return loss

    def log_prob_and_marginals(self, x):
        pi = self.infer_pi(x)
        return self.log_prob(x, pi), pi

    def prior_loss(self, alpha=1., weight=1.):
        if self.nef_topics:
            phi_dist = tfd.TransformedDistribution(tfd.Dirichlet(alpha + 0. * self.phi), tfb.Invert(tfb.SoftmaxCentered()))
            return tf.reduce_mean(nll()(self.phi_logits, phi_dist))
        else:
            phi_dist = tfd.Dirichlet(alpha + 0. * self.phi)
            return tf.reduce_mean(nll()(self.phi, phi_dist))

    def posterior_samples(self, x, pi=None):
        if pi is None:
            pi = self.infer_pi(x)
        counts = tf.round(tf.reduce_sum(x, axis=-1))
        mult_param = tf.matmul(pi, self.phi)
        return tf.reduce_sum(tfd.RelaxedOneHotCategorical(self.sample_temp, probs=mult_param).sample(self.words_per_sample), axis=0)
        #return tfd.Multinomial(total_count=counts, probs=mult_param).sample()
