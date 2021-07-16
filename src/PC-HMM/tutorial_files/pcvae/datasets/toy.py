from .base import dataset, input_dataset, quantized_dataset, classification_dataset, binary_classification_dataset, \
    binary_dataset, \
    real_dataset, remote_dataset, tensorflow_dataset, uniform_dataset
from sklearn.datasets import make_moons
import requests
import io
import numpy as np
from scipy.sparse import csr_matrix
from ..third_party.pcpy.toy_data import create_soft_line_dataset


def load_bar_split(split='train'):
    response = requests.get('https://github.com/dtak/prediction-constrained-topic-models/raw/master/datasets/toy_bars_3x3/X_csr_%s.npz' % split)
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    X = csr_matrix((data['data'], data['indices'], data['indptr'])).todense()
    X = np.expand_dims(X, [1, 2])
    response = requests.get('https://github.com/dtak/prediction-constrained-topic-models/raw/master/datasets/toy_bars_3x3/Y_%s.npy' % split)
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    y = data.astype(int).flatten()
    return X, y

def load_reviews_split(split='train'):
    response = requests.get('https://github.com/dtak/prediction-constrained-topic-models/raw/master/datasets/movie_reviews_pang_lee/X_csr_%s.npz' % split)
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    X = csr_matrix((data['data'], data['indices'], data['indptr'])).todense()
    X = np.expand_dims(X, [1, 2])
    response = requests.get('https://github.com/dtak/prediction-constrained-topic-models/raw/master/datasets/movie_reviews_pang_lee/Y_%s.npy' % split)
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    y = data.astype(int).flatten()
    return X, y

class toy_bars(dataset, real_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'TOY_BARS'
        self._noutputs = 2
        self._labels = [str(i) for i in range(2)]
        self.rescale_images = False
        self._shape = (1, 1, 9)

    def fetch_data(self, download_dir=None):
        self.data = dict(train=load_bar_split('train'), valid=load_bar_split('valid'), test=load_bar_split('test'))

class movie_reviews(dataset, real_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'MOVIE_REVIEWS'
        self._noutputs = 2
        self._labels = [str(i) for i in range(2)]
        self.rescale_images = False
        self._shape = (1, 1, 5338)

    def fetch_data(self, download_dir=None):
        self.data = dict(train=load_reviews_split('train'), valid=load_reviews_split('valid'), test=load_reviews_split('test'))
        response = requests.get(
            'https://github.com/dtak/prediction-constrained-topic-models/raw/master/datasets/movie_reviews_pang_lee/X_colnames.txt')
        response.raise_for_status()
        self.words = str(response.content).split('\\n')


class half_moons(dataset, real_dataset, classification_dataset):
    def __init__(self, n_samples=1000, noise=0.1, seed=543, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'MOONS'
        self._noutputs = 2
        self._labels = [str(i) for i in range(2)]
        self.n_samples = n_samples
        self.noise = noise
        self.use_rescale = True
        self.rescale_images = False
        self.seed = seed
        self.fix_order = True

    def fetch_data(self, download_dir=None):
        # Create data here
        x, y = make_moons(n_samples=self.n_samples - self.nlabels, noise=self.noise, random_state=self.seed)
        xu, yu = make_moons(n_samples=self.n_samples - self.nlabels, noise=self.noise, random_state=self.seed)
        x1, y1 = make_moons(n_samples=self.nlabels, noise=0.05, random_state=self.seed)
        xtr = np.concatenate([xu, x1], axis=0)
        ytr = np.concatenate([yu.astype(float) * np.nan, y1.astype(float)])
        self.data = dict(train=(xtr.reshape(-1, 2, 1, 1), ytr),
                         valid=(x.reshape(-1, 2, 1, 1), y), test=(x.reshape(-1, 2, 1, 1), y))

    def mask_labels(self, y, nlb, seed=543):
        return np.isfinite(y)




class toy_line(dataset, real_dataset, classification_dataset):
    def __init__(self, seed=543, t_length=8, **kwargs):
        dataset.__init__(self,  **kwargs)
        self._name = 'LINE'
        self._noutputs = 2
        self._labels = [str(i) for i in range(2)]
        self._t_length = t_length
        self._shape = (1, self._t_length, 2)
        self.use_rescale = False
        self.rescale_images = False
        self.seed = seed
        self.fix_order = True

    def fetch_data(self, download_dir=None):
        pi_0 = np.array([1., 0., 0., 0.])
#         pi = np.array([[.5, .5, 0., 0.],
#                        [0., .5, .5, 0.],
#                        [0., 0., 0., 1.],
#                        [0., 0., 0., 1.]])

        # 3 state
        pi = np.array([[0., 1., 0.],
                       [0., 0., 1.],
                       [0., 0., 1.]])
        
#         # 4 state
#         pi = np.array([[0., 1., 0., 0.],
#                        [0., 0., 1., 0.],
#                        [0., 0., 0., 1.],
#                        [0., 0., 0., 1.]])
        
#         # 6 state
#         pi = np.array([[.5, .5, 0., 0., 0., 0.],
#                        [0., .5, .5, 0., 0., 0.],
#                        [0., 0., .5, .5, 0., 0.],
#                        [0., 0., 0., .5, .5, 0.],
#                        [0., 0., 0., 0., .5, .5],
#                        [0., 0., 0., 0., 0., 1.]])

        np.random.seed(543)
        data = create_soft_line_dataset(
            nstates=3, xdim=2, nseq=500, mean_length=-self._t_length, seed=int(np.random.random() * 1000), viz=False, self_bias=3,
            pi=(pi_0, pi)
        )

        def convertLabels(data):
            ndata = []
            for (xi, yi, zi) in zip(*data):
                if np.sum(zi == 2) != 1:
                    continue
                yi = int(xi[zi == 2, 1] > 0.)
                ndata.append((xi, yi, zi))
            D = tuple(zip(*ndata))
            D = list(D[0]), list(D[1]), list(D[2])
            return D

        data, test_data = convertLabels(data[0]), convertLabels(data[1])
        train_data = np.expand_dims(np.stack(data[0]), 1), np.expand_dims(data[1], 1)
        test_data = np.expand_dims(np.stack(test_data[0]), 1), np.expand_dims(test_data[1], 1)
        self.data = dict(train=train_data, valid=test_data, test=test_data)

