import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf

def loglik(x, p_x):
    # get index of sequences that have any missing observtions
    # mask = tf.reduce_sum(x, axis=tf.range(1, tf.rank(x)))

    # replace the nans with zeros
    x_imp = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    # predictor loss multiplied by lambda
    weight=1
    llik = -weight * p_x.log_prob(x_imp)

    # replace likelihoods of the missing observations at a specific dimension with 0
    if llik.shape[-1]==x.shape[-1]:
        llik = tf.where(tf.math.is_nan(x), tf.zeros_like(llik), llik)

        # sum across all dimensions (Note : Change this if the covariance matrix is not diagonal)
        llik = tf.reduce_sum(llik, axis=-1)

    # Uncomment this line if we want to completely ignore sequences with any missing observation or labels
    # llik = tf.where(tf.math.is_nan(mask * llik), tf.zeros_like(llik), llik)

    return llik 


if __name__ == '__main__':
    
    # create some synthetic data
    N = 100
    T = 50
    F = 2
    
    x = np.random.randn(N, T, F)
    
    # add missing values at random
    n_missing = 1000
    miss_inds = np.random.choice(x.size, n_missing, replace=False)
    x.ravel()[miss_inds] = np.nan
    
    # instantiate tensorflow normal distribution 
    p_x = tfp.distributions.Normal(loc=np.zeros(2), scale =np.ones(F))
    
    # print log likelihood
    print('Log likelihoods : ')
    print(loglik(x, p_x))
    print('Number of nan terms : %s'%(np.sum(np.isnan(loglik(x, p_x)))))
    
    