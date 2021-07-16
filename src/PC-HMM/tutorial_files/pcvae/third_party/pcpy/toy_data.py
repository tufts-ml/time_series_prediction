from .generate import *
import random

def split_dataset(x, y, z, split=0.7, shuffle=True, seed=543, preserve_classes=True):
    """
    Split a dataset into training and test data.

    Parameters
    ----------
    x : list of matrix (T x D)
        Observations for each sequence.
    y : list of int
        Label for each sequence.
    z : list of vector (T)
        Hidden state sequences.
    split : float, optional
        Fraction of data to use for training.

    Returns
    -------
    data : tuple
        (x, y, z) for both training and test data.

    """
    random.seed(seed)

    if not (type(split) is list) and not (type(split) is tuple):
        split = [split, 1.0 - split]

    if preserve_classes:
        sets = [[] for s in split]
        for c in range(np.array(y).astype(int).max() + 1):
            sdata = [(xi, yi, zi) for xi, yi, zi in list(zip(x, y, z)) if int(yi) == c]

            if shuffle:
                random.shuffle(sdata)

            splits, ind = [int(round(len(sdata) * s)) for s in split], 0
            for set_ind, s in enumerate(splits):
                sets[set_ind] += sdata[ind : (ind + s)]
                ind += s

        for s in sets:
            random.shuffle(s)

        datasets = []
        for s in sets:
            x, y, z = list(zip(*s))
            x, y, z = list(x), list(y), list(z)
            datasets.append((x, y, z))

        return datasets

    if shuffle:
        sdata = list(zip(x, y, z))
        random.shuffle(sdata)
        x, y, z = list(zip(*sdata))
        x, y, z = list(x), list(y), list(z)

    # If split is just a float, split once otherwise split into the number of sets in the split list
    sets, splits, ind = [], [int(round(len(x) * s)) for s in split], 0
    for s in splits:
        sets.append((x[ind : (ind + s)], y[ind : (ind + s)], z[ind : (ind + s)]))
        ind += s
    return sets

def create_line_dataset(
    nstates=5, xdim=2, nseq=500, mean_length=20, seed=326732, viz=False
):
    np.random.seed(seed)
    pi_0, pi = create_pi(nstates, dir_param=0.1, self_bias=0.75)
    eta = create_eta(nstates, dev=3)
    phi = create_full_gaussian_phi(nstates, xdim=xdim)
    phi[:, 1] = 0.0
    phi[:, 0] = np.arange(nstates) * 8
    x, y, z = generate_dataset(
        pi_0,
        pi,
        eta,
        phi,
        obs_generator=generate_obs_sequence_full_gauss,
        nseq=nseq,
        mean_length=mean_length,
        multiplier=2,
    )
    y = np.round(np.random.random(len(y))).tolist()
    x = [
        np.hstack([xn[:, :1], np.abs(xn[:, 1:]) * (2 * yn - 1)]) for xn, yn in zip(x, y)
    ]
    (x, y, z), (x_test, y_test, z_test) = split_dataset(x, y, z, split=0.7)
    return (x, y, z), (x_test, y_test, z_test), (pi_0, pi, phi, eta)


def create_soft_line_dataset(
    nstates=5, xdim=2, nseq=500, mean_length=12, thresh=0.0, seed=326732, viz=False,
        self_bias=0.75, pi=None,
):
    np.random.seed(seed)
    if not (pi is None):
        pi_0, pi = pi
    else:
        pi_0, pi = create_pi(nstates, dir_param=0.1, self_bias=self_bias)
    eta = create_eta(nstates, dev=3)
    phi = create_full_gaussian_phi(nstates, xdim=xdim)
    phi[:, 1] = 0.0
    phi[:, 0] = np.arange(nstates) * 8
    x, y, z = generate_dataset(
        pi_0,
        pi,
        eta,
        phi,
        obs_generator=generate_obs_sequence_full_gauss,
        nseq=nseq,
        mean_length=mean_length,
        multiplier=2,
    )
    phi = phi[:, :xdim], phi[:, xdim:]
    y = [np.round(np.sign(np.sum(np.sign(xd[:, 1] - thresh)))) for xd in x]
    y = [1 if yd == 0 else (yd + 1) / 2 for yd in y]
    (x, y, z), (x_test, y_test, z_test) = split_dataset(x, y, z, split=0.7)
    return (x, y, z), (x_test, y_test, z_test), (pi_0, pi, phi, eta)

def convertToSS(data):
    data = tuple(zip(*[(xi[zi != 2], yi, zi[zi != 2]) for (xi, yi, zi) in zip(*data)]))
    ndata = []
    for (xi, yi, zi) in zip(*data):
        ind = int(np.random.random() * len(zi))
        xn = stats.multivariate_normal.rvs(mean=np.array([15., 0.]))
        xi = np.concatenate([xi[:ind], np.atleast_2d(xn), xi[ind:]])
        zi = np.concatenate([zi[:ind], np.atleast_1d(2), zi[ind:]])
        yi = int(xn[1] > 0.)
        ndata.append((xi, yi, zi))
    D = tuple(zip(*ndata))
    D = list(D[0]), list(D[1]), list(D[2])
    return D

def create_soft_split_dataset():
    data, test_data, params = create_soft_line_dataset(
        nstates=5, xdim=2, nseq=500, mean_length=12, seed=326732, viz=False
    )

    nlabeled = 10
    data, test_data = convertToSS(data), convertToSS(test_data)
    dataSS = data[0], data[1][:nlabeled] + (len(data[1][nlabeled:]) * [-1]), data[2]
    dataSM = data[0][:nlabeled], data[1][:nlabeled], data[2][:nlabeled]
    return (data, test_data), (dataSS, test_data), (dataSM, test_data)

def create_soft_line_dataset_per_step_labels(
    nstates=5,
    xdim=2,
    nseq=500,
    mean_length=30,
    thresh=0.0,
    window=5,
    majority=True,
    per_ts=True,
    cdrop=False,
    seed=326732,
    viz=False,
):
    np.random.seed(seed)
    pi_0, pi = create_pi(nstates, dir_param=0.1, self_bias=1.75)
    eta = create_eta(nstates, dev=3)
    phi = create_full_gaussian_phi(nstates, xdim=xdim)
    phi[:, 1] = 0.0
    phi[:, 0] = np.arange(nstates) * 8
    x, y, z = generate_dataset(
        pi_0,
        pi,
        eta,
        phi,
        obs_generator=generate_obs_sequence_full_gauss,
        nseq=nseq,
        mean_length=mean_length,
        multiplier=2,
    )
    phi = phi[:, :xdim], phi[:, xdim:]

    y = []
    for xi, zi in zip(x, z):
        # Label obs above and below line
        signs = np.sign(xi[:, 1] - thresh)

        # Label each window based on if more are positive or negative
        wthresh = 0 if majority else (window - 0.5)
        labels = np.sign(np.convolve(signs, np.ones(window), "same") - wthresh)

        # Convert to 0-1 and add to list
        labels = np.round((labels + 1) / 2).astype(int)
        if not per_ts:
            labels = np.array(labels, dtype=int).max()

        if cdrop:
            print(signs[1:] < signs[:-1])
            print(zi[1:] == z[:-1])
            labels = np.array(
                [0] + ((signs[1:] < signs[:-1]) & (zi[1:] == zi[:-1])).tolist()
            )

        y.append(labels)
    (x, y, z), (x_test, y_test, z_test) = split_dataset(
        x, y, z, split=0.7, preserve_classes=False
    )
    return (x, y, z), (x_test, y_test, z_test), (pi_0, pi, phi, eta)


def create_pluses_dataset(
    nstates=9, xdim=2, nseq=500, mean_length=20, seed=326732, viz=False
):
    np.random.seed(seed)
    pi_0, pi = create_pi(nstates, dir_param=0.1, self_bias=0.25)
    eta = create_eta(nstates, dev=3)
    phi = create_full_gaussian_phi(nstates, xdim=xdim)
    phi[:, 1] = 0.0
    phi[:, 0] = 0.0
    x, y, z = generate_dataset(
        pi_0,
        pi,
        eta,
        phi,
        obs_generator=generate_obs_sequence_full_gauss,
        nseq=nseq,
        mean_length=mean_length,
        multiplier=2,
    )
    y = np.round(np.random.random(len(y))).tolist()

    for i in range(len(x)):
        lab = y[i]
        for j in range(x[i].shape[0]):
            obs = x[i][j]
            state = z[i][j]
            cx = 10 * (state % 3 - 1)
            cy = 10 * (state / 3 - 1)

            obs[0] = obs[0] * 2
            obs[1] = obs[1] / 2

            if lab == 1:
                obs[0], obs[1] = obs[1], obs[0]

            obs[0] += cx
            obs[1] += cy
            x[i][j] = obs

    for state in range(nstates):
        cx = 10 * (state % 3 - 1)
        cy = 10 * (state / 3 - 1)
        phi[state, 0] = cx
        phi[state, 1] = cy

    (x, y, z), (x_test, y_test, z_test) = split_dataset(x, y, z, split=0.7)
    return (x, y, z), (x_test, y_test, z_test), (pi_0, pi, phi, eta)


def create_rotation_dataset(
    nstates=3, xdim=2, nseq=500, mean_length=50, seed=326732, soft=False, viz=False
):
    np.random.seed(seed)
    pi_0 = np.ones(3) / 3.0

    pi_A = np.zeros((3, 3))
    pi_A[0, 1] = 1.0
    pi_A[1, 2] = 1.0
    pi_A[2, 0] = 1.0

    pi_B = np.zeros((3, 3))
    pi_B[1, 0] = 1.0
    pi_B[2, 1] = 1.0
    pi_B[0, 2] = 1.0

    if soft:
        pi_A = normalize(pi_A + np.ones_like(pi_A), 1)
        pi_B = normalize(pi_B + np.ones_like(pi_B), 1)

    phi = np.array([[3, 0], [-3, 0], [0, 6]])

    eta = create_eta(nstates, dev=3)

    x_A, y_A, z_A = generate_dataset(
        pi_0,
        pi_A,
        eta,
        phi,
        obs_generator=generate_obs_sequence_std_gauss,
        nseq=nseq,
        mean_length=mean_length,
        multiplier=2,
    )
    x_B, y_B, z_B = generate_dataset(
        pi_0,
        pi_B,
        eta,
        phi,
        obs_generator=generate_obs_sequence_std_gauss,
        nseq=nseq,
        mean_length=mean_length,
        multiplier=2,
    )

    y_A = np.zeros(len(y_A))
    y_B = np.ones(len(y_B))

    x = x_A + x_B
    y = np.hstack([y_A, y_B]).tolist()
    z = z_A + z_B

    order = np.random.permutation(len(x))
    x = [x[i] for i in order]
    y = [y[i] for i in order]
    z = [z[i] for i in order]

    (x, y, z), (x_test, y_test, z_test) = split_dataset(x, y, z, split=0.7)
    return (x, y, z), (x_test, y_test, z_test), (pi_0, np.ones((3, 3)) / 3.0, phi, eta)


def create_flip_flop_dataset(
    nstates=2, xdim=2, nseq=500, mean_length=50, seed=326732, viz=False
):
    np.random.seed(seed)
    pi_0 = np.ones(2) / 2.0

    pi_A = np.ones((2, 2)) / 2.0
    pi_B = np.array([[0.95, 0.05], [0.05, 0.95]])

    phi = np.array([[5, 0], [-5, 0]])

    eta = create_eta(nstates, dev=3)

    x_A, y_A, z_A = generate_dataset(
        pi_0,
        pi_A,
        eta,
        phi,
        obs_generator=generate_obs_sequence_std_gauss,
        nseq=nseq,
        mean_length=mean_length,
        multiplier=2,
    )
    x_B, y_B, z_B = generate_dataset(
        pi_0,
        pi_B,
        eta,
        phi,
        obs_generator=generate_obs_sequence_std_gauss,
        nseq=nseq,
        mean_length=mean_length,
        multiplier=2,
    )

    y_A = np.zeros(len(y_A))
    y_B = np.ones(len(y_B))

    x = x_A + x_B
    y = np.hstack([y_A, y_B]).tolist()
    z = z_A + z_B

    order = np.random.permutation(len(x))
    x = [x[i] for i in order]
    y = [y[i] for i in order]
    z = [z[i] for i in order]

    (x, y, z), (x_test, y_test, z_test) = split_dataset(x, y, z, split=0.7)
    return (x, y, z), (x_test, y_test, z_test), (pi_0, np.ones((2, 2)) / 2.0, phi, eta)


def create_mult_dataset(
    nstates=8, xdim=10, nseq=500, mean_length=20, dir_param=1, seed=326732
):
    np.random.seed(seed)
    pi_0_A, pi_A = create_pi(
        nstates=nstates, dir_param=dir_param, self_bias=0, rand_bias=True
    )
    pi_0_B, pi_B = create_pi(
        nstates=nstates, dir_param=dir_param, self_bias=0, rand_bias=True
    )

    eta = create_eta(nstates, dev=3)
    phi = create_categorical_phi(nstates=nstates, xdim=xdim, dir_param=1)

    x_A, y_A, z_A = generate_dataset(
        pi_0_A,
        pi_A,
        eta,
        phi,
        obs_generator=generate_obs_sequence_categorical,
        nseq=nseq,
        mean_length=mean_length,
        multiplier=2,
    )
    x_B, y_B, z_B = generate_dataset(
        pi_0_B,
        pi_B,
        eta,
        phi,
        obs_generator=generate_obs_sequence_categorical,
        nseq=nseq,
        mean_length=mean_length,
        multiplier=2,
    )

    y_A = np.zeros(len(y_A))
    y_B = np.ones(len(y_B))

    x = x_A + x_B
    y = np.hstack([y_A, y_B]).tolist()
    z = z_A + z_B

    order = np.random.permutation(len(x))
    x = [x[i] for i in order]
    y = [y[i] for i in order]
    z = [z[i] for i in order]

    (x, y, z), (x_test, y_test, z_test) = split_dataset(x, y, z, split=0.7)
    return (
        (x, y, z),
        (x_test, y_test, z_test),
        (pi_0_A, np.ones((2, 2)) / 2.0, phi, eta),
    )