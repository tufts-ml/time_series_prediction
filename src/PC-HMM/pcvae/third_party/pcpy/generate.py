import numpy as np
import scipy.stats as stats

def sigmoid(x):
    """Computes the sigmoid function element-wise."""
    return 1 / (1.0 + np.exp(-x))

def calc_zbar_index(z, nstates):
    """
    Computes a vector of the fraction of time spend in each state from a
    given state sequence.
    Parameters
    ----------
    z : vector (T)
        Hidden state sequence.
    nstates : int
        Number of possible hidden states (K).
    Returns
    -------
    zbar : vector (K)
        The vector of state fractions.
    Notes
    -----
    This is NOT differentiable with Autograd.
    """
    zbar = np.zeros((z.size, nstates))
    zbar[np.arange(z.size), z] = 1
    return zbar.sum(axis=0) / zbar.sum()

def create_pi(nstates=5, dir_param=1, self_bias=0, rand_bias=True):
    """
    Sample inital state and transition probabilities for an HMM from
    a Dirichlet distribution.
    Parameters
    ----------
    nstates : int, optional
        Number of possible hidden states (K).
    dir_param : float or vector (K), optional
        Parameters of the Dirichlet distribution. If a scalar then treat as a
        symmetric Dirichlet of dimension K.
    self_bias : float or vector (K), optional
        If nozero, value added to self transition probabilities. Probabilities
        are then renormalized.
    rand_bias : bool, optional
        If true, multiply self_bias by a random vector for randomized biases.
    Returns
    -------
    pi_0 : vector (K)
        Initial state probabilities.
    pi : matrix (K x K)
        Transition probabilities.
    """
    lam = dir_param * np.ones(nstates)
    pi_0 = stats.dirichlet.rvs(lam).flatten()
    pi = stats.dirichlet.rvs(lam, nstates)

    self_bias *= np.random.random(nstates) if rand_bias else 1
    pi += self_bias * np.eye(nstates)
    pi = pi / pi.sum(axis=1).reshape((-1, 1))
    return pi_0, pi


def create_eta(nstates=5, mean=0, dev=1, regress_joint=False, classes=1):
    """
    Sample the regression weights for an sHMM model from a diagonal Gaussian.
    Parameters
    ----------
    nstates : int, optional
        Number of possible hidden states (K).
    mean : float, optional
        Mean of the distribution for each weight.
    dev : float, optional
        Std dev of distribution for each weight.
    regress_joint : boolean, optional
        If true, create states^2 regression coefficients.
    Returns
    -------
    eta : vector (K)
        Regression weights.
    """
    nstates = nstates * nstates if regress_joint else nstates
    size = nstates if classes <= 2 else (nstates, classes)
    eta = stats.norm.rvs(mean, dev, size)
    return eta


def create_std_gaussian_phi(nstates=5, xdim=2, mean=0, dev=20):
    """
    Sample means for Gaussian emmision distributions with fixed variances.
    Parameters
    ----------
    nstates : int, optional
        Number of possible hidden states (K).
    xdim : int, optional
        Dimension of observations (D).
    mean : float, optional
        Mean of the distribution for emission means.
    dev : float, optional
        Std dev of distribution for emission means.
    Returns
    -------
    phi : matrix (K x D)
        Emission distribution parameters.
    """
    mu = mean * np.ones(xdim)
    phi = stats.multivariate_normal.rvs(mu, dev, nstates)
    return phi.reshape((-1, xdim))


def create_full_gaussian_phi(nstates=5, xdim=2, mean=0, dev=20):
    """
    Sample means for Gaussian emmision distributions with indentity covariances.
    Parameters
    ----------
    nstates : int, optional
        Number of possible hidden states (K).
    xdim : int, optional
        Dimension of observations (D).
    mean : float, optional
        Mean of the distribution for emission means.
    dev : float, optional
        Std dev of distribution for emission means.
    Returns
    -------
    phi : matrix (K x D)
        Emission distribution parameters.
    """
    mu = mean * np.ones(xdim)
    phi = stats.multivariate_normal.rvs(mu, dev, nstates).reshape((-1, xdim))
    cov = np.stack([np.eye(xdim).flatten() for i in range(nstates)])
    phi = np.hstack([phi, cov])
    return phi


def create_categorical_phi(nstates=5, xdim=2, dir_param=1):
    """
    Sample means for Gaussian emmision distributions with fix variances.
    Parameters
    ----------
    nstates : int, optional
        Number of possible hidden states (K).
    xdim : int, optional
        Dimension of observations (D).
    dir_param : float or vector (D), optional
        Parameters of the Dirichlet distribution. If a scalar then treat as a
        symmetric Dirichlet of dimension K.
    Returns
    -------
    phi : matrix (K x D)
        Emission distribution parameters.
    """
    lam = dir_param * np.ones(xdim)
    phi = stats.dirichlet.rvs(lam, nstates)
    return np.log(phi)


def generate_state_sequence(pi_0, pi, length=25):
    """
    Sample a sequence of hidden states.
    Parameters
    ----------
    pi_0 : vector (K)
        Initial state probabilities.
    pi : matrix (K x K)
        Transition probabilities.
    length : int, optional
        Length of the sequence (T).
    Returns
    -------
    z : vector (T)
        Hidden state sequence.
    """
    z = [np.random.multinomial(1, pi_0, 1).argmax()]
    for i in range(1, length):
        z.append(np.random.multinomial(1, pi[z[-1]], 1).argmax())
    return np.array(z, dtype=int)


def generate_label(z, eta, multiplier=10):
    """
    Sample a label for a sequence.
    Parameters
    ----------
    z : vector (T)
        Hidden state sequence.
    eta : vector (K)
        Regression weights.
    multiplier: scalar, optional
        Factor to mulitply the results of zbar^T eta
    Returns
    -------
    y : int
        Label for a sequence.
    """
    zbar = calc_zbar_index(z, eta.size)
    p = sigmoid(multiplier * np.dot(zbar, eta))
    return 1 if np.random.random() < p else 0


def generate_obs_sequence_std_gauss(z, phi):
    """
    Generate observations for a sequence using Gaussian emission distributions
    with fixed (identity) variances.
    Parameters
    ----------
    phi : matrix (K x D)
        Emission distribution parameters.
    z : vector (T)
        Hidden state sequence.
    Returns
    -------
    x : matrix (T x D)
        Observations for a sequence.
    """
    x = stats.multivariate_normal.rvs(np.zeros(phi.shape[1]), 1, z.size)
    x = x.reshape((-1, phi.shape[1]))
    x += phi[z, :]
    return x


def generate_obs_sequence_full_gauss(z, phi, xdim=2):
    """
    Generate observations for a sequence using Gaussian emission distributions
    with full covariances.
    Parameters
    ----------
    phi : matrix (K x (D + D^2))
        Emission distribution parameters.
    z : vector (T)
        Hidden state sequence.
    Returns
    -------
    x : matrix (T x D)
        Observations for a sequence.
    """
    xdim = int((np.sqrt(1 + 4 * phi.shape[1]) - 1) / 2)

    x = []

    mu = phi[:, :xdim]
    L = phi[:, xdim:].reshape((-1, xdim, xdim))

    cov = []
    for Lk in L:
        cov.append(np.dot(Lk, Lk.T))

    for zk in z:
        x.append(stats.multivariate_normal.rvs(mu[zk], cov[zk]))

    x = np.stack(x)
    return x


def generate_obs_sequence_categorical(z, phi):
    """
    Generate observations for a sequence using categorical emission distributions
    with fixed (identity) variances.
    Parameters
    ----------
    phi : matrix (K x D)
        Emission distribution parameters.
    z : vector (T)
        Hidden state sequence.
    Returns
    -------
    x : matrix (T x 1)
        Observations for a sequence (int).
    """
    phi = np.exp(phi)
    phi = phi / phi.sum(axis=1).reshape((-1, 1))

    x = []
    for zk in z:
        x.append(np.random.choice(phi.shape[1], p=phi[zk]))
    x = np.array(x, dtype=np.int64).reshape((-1, 1))
    return x


def generate_sequence(
    pi_0,
    pi,
    eta,
    phi,
    length=25,
    obs_generator=generate_obs_sequence_std_gauss,
    multiplier=10,
):
    """
    Generate a full sequence of data from the sHMM model.
    Parameters
    ----------
    pi_0 : vector (K)
        Initial state probabilities.
    pi : matrix (K x K)
        Transition probabilities.
    eta : vector (K)
        Regression weights.
    phi : matrix (K x D)
        Emission distribution parameters.
    length : int, optional
        Length of the sequence (T).
    obs_generator : function, optional
        Function that generates x from phi for a particular emission distribution.
    multiplier: scalar, optional
        Factor to mulitply the results of zbar^T eta
    Returns
    -------
    x : matrix (T x D)
        Observations for a sequence.
    y : int
        Label for a sequence.
    z : vector (T)
        Hidden state sequence.
    """
    z = generate_state_sequence(pi_0, pi, length)
    y = generate_label(z, eta, multiplier=multiplier)
    x = obs_generator(z, phi)
    return x, y, z


def generate_dataset(
    pi_0,
    pi,
    eta,
    phi,
    nseq=3,
    mean_length=100,
    obs_generator=generate_obs_sequence_std_gauss,
    multiplier=10,
):
    """
    Generate a dataset of sequences from the sHMM model.
    Parameters
    ----------
    pi_0 : vector (K)
        Initial state probabilities.
    pi : matrix (K x K)
        Transition probabilities.
    eta : vector (K)
        Regression weights.
    phi : matrix (K x D)
        Emission distribution parameters.
    nseq : int, optional
        Number of sequences to generate (N).
    mean_length : int, optional
        Mean of a Poisson distribution on the length of sequences (T).
    obs_generator : function, optional
        Function that generates x from phi for a particular emission distribution.
    multiplier: scalar, optional
        Factor to mulitply the results of zbar^T eta
    Returns
    -------
    x : list of matrix (T x D)
        Observations for each sequence.
    y : list of int
        Label for each sequence.
    z : list of vector (T)
        Hidden state sequences.
    """
    x, y, z = [], [], []
    for i in range(nseq):
        length = stats.poisson.rvs(mean_length) if mean_length > 0 else -mean_length
        x_t, y_t, z_t = generate_sequence(
            pi_0,
            pi,
            eta,
            phi,
            length,
            obs_generator=obs_generator,
            multiplier=multiplier,
        )
        x.append(x_t)
        y.append(y_t)
        z.append(z_t)
    return x, y, z