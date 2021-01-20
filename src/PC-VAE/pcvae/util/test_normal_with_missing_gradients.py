import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class NormalWithMissing(tfp.distributions.Normal):
    def __init__(self, loc, scale):
        tfp.distributions.Normal.__init__(self, loc=loc, scale=tf.math.softplus(scale + 1.) + 1e-5)        
    
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
        prob_with_nans = log_unnormalized - log_normalization        
        
#         # replace nans with zeros
#         prob = tf.where(tf.math.is_nan(prob_with_nans), tf.zeros_like(prob_with_nans), prob_with_nans)
        
        # return 0 in locations where the observations are missing 
        return mask*prob_with_nans
    
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
        prob_with_nans = tf.exp(log_unnormalized - log_normalization) 
        
        return mask*prob


@tf.function
def compute_log_prob(x, loc, scale):
    '''
    Function to compute the log probability of data x given univariate gaussians with loc and scale as means and stds 
    '''
    p_x_missing = NormalWithMissing(loc=loc, scale = scale)
    
    return -p_x_missing.log_prob(x)


if __name__ == '__main__':
    
    print('Optimizing log probability function with NO missing values...')
    # initialize parameters of 2 univariate gaussians
    means = tf.Variable(np.zeros(2))
    stds = tf.Variable(np.array([1., 1.]))
    
    # choose 2 non-missing data points
    x = tf.constant(np.array([[10., 5.], [6., 5.]]))
    
    # Create an optimizer.
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Compute the gradients for a list of variables.
    for iteration in range(0, 10000):
        with tf.GradientTape() as tape:
            log_probs = compute_log_prob(x, means, stds)
            if (iteration%1000)==0:
                print('epoch : %s'%iteration)
                print('log probabilities : ')
                print(log_probs)
            
        # apply gradient
        grads = tape.gradient(log_probs, [means, stds])
        opt.apply_gradients(zip(grads, [means, stds]))

    print('optimal means : ')
    print(means.numpy())
    
    print('optimal stds : ')
    print(stds.numpy())
    
    print('Optimizing log probability function with missing values...')
    # choose 2 new data points with missing observations
    x = tf.constant(np.array([[10., np.nan], [6., 5.]]))

    # Create an optimizer.
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # initialize parameters of 2 univariate gaussians
    means = tf.Variable(np.zeros(2))
    stds = tf.Variable(np.array([1., 1.]))
        
    # Compute the gradients for a list of variables.
    for iteration in range(0, 10000):
        with tf.GradientTape() as tape:
            log_probs = compute_log_prob(x, means, stds)
            if (iteration%1000)==0:
                print('epoch : %s'%iteration)
                print('log probabilities : ')
                print(log_probs)
            
        # apply gradient
        grads = tape.gradient(log_probs, [means, stds])
        opt.apply_gradients(zip(grads, [means, stds]))

    print('optimal means : ')
    print(means)
    
    print('optimal stds : ')
    print(stds)
    
    from IPython import embed; embed()