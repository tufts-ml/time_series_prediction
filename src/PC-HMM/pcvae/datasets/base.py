from pcvae.util import get_function_spec, print_options
from sklearn.model_selection import train_test_split
import numpy as np
from pcvae.util import as_np
import tensorflow as tf
from copy import copy
from pcvae.util import select_labeled_binary, random_shift, random_flip, random_invert, cutout, get_gcn, \
    get_img_rescale, get_zca_transformer, get_img_standardize, get_img_clustered, get_img_instance_standardize, \
    get_img_instance_greyscale
import gc, pickle
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr, scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h
#     farid_v, farid_h
import skimage
from ..visualizations import plot_images
from ..third_party.VolumetricDataGenerator import customImageDataGenerator
import tensorflow_datasets as tfds

# Classes that handle how data is loaded
class remote_dataset(object):
    def fetch_data(self, download_dir=None):
        raise NotImplementedError()


class tensorflow_dataset(remote_dataset):
    def fetch_data(self, download_dir=None):
        data = self.tf_dataset()
        download_dir = download_dir if download_dir else self.download_dir
        data.download_and_prepare(download_dir=download_dir, download_config=tfds.download.DownloadConfig(register_checksums=self.register_checksums))
        as_supervised = self.xkey == 0 and self.ykey == 1
        self.data = data.as_dataset(as_supervised=as_supervised, batch_size=self.chunk_size)


# Classes that handle different input types
class input_dataset(object):
    def preprocessing_func(self, x):
        return x


class binary_dataset(input_dataset):
    def preprocessing_func(self, x):
        if np.max(x) > 1:
            x = x / 256.
        r = np.random.random(x.shape)
        return (x > r).astype(self.dtype)


class uniform_dataset(input_dataset):
    def preprocessing_func(self, x):
        if np.max(x) > 1:
            x = x / 256.
        return x.astype(self.dtype)


class real_dataset(input_dataset):
    def preprocessing_func(self, x):
        x = x.astype(self.dtype)
        if self.make_continuous_images:
            x = x + np.random.random(x.shape) - 0.5
        if self.rescale_images:
            x = x / 256.
        if self.autoregressive_transform:
            x = np.concatenate([x[:, :, :-1], x[:, :, 1:]], axis=-1)
        return x


class quantized_dataset(input_dataset):
    def preprocessing_func(self, x):
        x = x.astype(self.dtype)
        if self.rescale_images:
            x = x / 256.
        return x


# Classes that handle different label types
class labeled_dataset(object):
    def mask_labels(self, y, nlb, seed=543):
        raise NotImplementedError()

    def encode_labels(self, y, unsupervised=False):
        raise NotImplementedError()

    def default_predictor(self):
        raise NotImplementedError()


class classification_dataset(labeled_dataset):
    def mask_labels(self, y, nlb, seed=543):
        if -1 in list(y):
            return np.array(y, dtype=int).flatten() >= 0
        classes = self.classes()
        y_train_in = np.zeros(len(y), dtype=int)
        np.random.seed(seed)

        nlb = int(nlb / classes)
        inds = np.arange(len(y))
        inds_by_label = []

        for c in range(classes):
            clbs = inds[np.squeeze(y) == c].copy()
            np.random.shuffle(clbs)
            inds_by_label.append(clbs[:nlb])
        inds = np.concatenate(inds_by_label, axis=0)
        y_train_in[inds] = 1
        return y_train_in.astype(bool)

    def encode_labels(self, y, unsupervised=False):
        ynew = np.zeros((y.shape[0], self.classes()))
        if unsupervised:
            return ynew
        for c in range(self.classes()):
            ynew[np.squeeze(y) == c, c] = 1
        
        # leave the unlabelled data untouched
        ynew[np.isnan(y)] = np.nan
        return ynew

    def get_metrics(self):
        return [
#             tf.keras.metrics.CategoricalAccuracy(name='accuracy'), 
                tf.keras.metrics.AUC(curve='PR', name='AUPRC', multi_label=True, label_weights=[0, 1]),
                tf.keras.metrics.AUC(name='AUC', multi_label=True),
        ]


class regression_dataset(labeled_dataset):
    def mask_labels(self, y, nlb, seed=543):
        if not np.all(np.isfinite(y)):
            return np.ones(y.shape[0])
        y_train_in = np.zeros(len(y), dtype=int)
        y_train_in[:nlb] = 1
        np.random.seed(seed)
        np.random.shuffle(y_train_in)
        return y_train_in.astype(bool)

    def encode_labels(self, y, unsupervised=False):
        return y

    def get_metrics(self):
        return [tf.keras.metrics.RootMeanSquaredError(name='accuracy')]


class binary_classification_dataset(labeled_dataset):
    def mask_labels(self, y, nlb, seed=543):
        # Remove labels in a class-balanced way and encode
        y = np.expand_dims(y, axis=-1)[:2]
        y_train_in = np.zeros_like(y)
        np.random.seed(seed)
        inds = select_labeled_binary(y, nlb)
        y_train_in[inds] = 1
        return y_train_in.astype(bool)

    def encode_labels(self, y, unsupervised=False):
        y = np.atleast_2d(y).astype(float)
        if unsupervised:
            y = -1 * np.ones_like(y)
        return y

    def get_metrics(self):
        return [tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='AUC')]


class dataset(labeled_dataset, remote_dataset, input_dataset):
    '''
    Generic class that encapsulates downloading, preprocessing and feeding data.
    '''

    def __init__(self, download_dir=None, augment_args=None, zca_whiten=False, batch_size=32, chunk_size=20000,
                 nlabels=-1, valid_size=10000, random_shift=0, shift_channels=False, random_flip=False,
                 random_invert=False, random_noise=False, cutout=False,
                 seed=543, use_extra=False, make_continuous_images=False, rescale_images=True, balance=True,
                 dtype=np.float32, labeled_only=False, show_inputs=False, quantize=0, greyscale=False,
                 half_size=False, resize_images=None,
                 **kwargs):
        augment_args = {} if augment_args is None else augment_args
        if zca_whiten:
            augment_args.update(dict(zca_whitening=True, featurewise_center=True))
        augment_args['dtype'] = dtype
        self.dtype = dtype
        self.download_dir = download_dir
        self.batch_size = batch_size
        self.balance = balance
        self.chunk_size = chunk_size
        self.nlabels = nlabels
        self.valid_size = valid_size
        self.seed = seed
        self.use_extra = use_extra
        self.labeled_only = labeled_only
        self.make_continuous_images = make_continuous_images
        self.rescale_images = rescale_images
        self.data_loaded = False
        self.data_fetched = False
        self.semisupervised = self.nlabels > 0
        augment_args['preprocessing_function'] = self.preprocessing_func
        self.idg = tf.keras.preprocessing.image.ImageDataGenerator(**augment_args)
        if hasattr(self, '_shape') and len(self._shape) > 3:
            self.idg = customImageDataGenerator(**augment_args)
        self.split = 'test'
        self.xkey = 0
        self.ykey = 1
        self.steps_per_epoch = 500
        self.random_shift = random_shift
        self.random_flip = random_flip
        self.random_invert = random_invert
        self.random_noise = random_noise
        self.cutout = cutout
        self.use_gcn = False
        self.shift_channels = shift_channels
        self.norm = lambda x: x
        self.norm_inv = lambda x: x
        self.use_rescale = False
        self.use_standardize = False
        self.instance_standardize = False
        self.use_zca = False
        self.zca = lambda x: x
        self.show_inputs = show_inputs
        self.draw_samples = False
        self.quantize = quantize
        self.greyscale = greyscale
        self.edges = False
        self._shape = None if not hasattr(self, '_shape') else self._shape
        self.imagedata = True
        self.trim = None
        self.register_checksums = False
        self.half_size = half_size
        self.resize_images = resize_images
        self.label_group = None
        self.autoregressive_transform = False

    def prepare_data(self):
        # Extract the relevant features
        as_supervised = self.xkey == 0 and self.ykey == 1
        train, test, labeled = self.data['train'], self.data['test'], None
        
        
        if not as_supervised:
            def mf(a):
                x = a[self.xkey]
                if self.label_group:
                    a = a[self.label_group]

                ykey = self.ykey if (type(self.ykey) is list) else [self.ykey]
                y = []
                for yk in ykey:
                    yi = tf.convert_to_tensor(a[yk])
                    yi = tf.cond(tf.rank(yi) == 1, lambda: tf.reshape(yi, (-1, 1)), lambda: yi)
                    y.append(yi)
                y = tf.concat(y, axis=1)
                return x, y

            train, test = train.map(mf), test.map(mf)
            valid = self.data['valid'].map(mf) if 'valid' in self.data else None
            extra = self.data['extra'].map(mf) if 'extra' in self.data else None
        else:
            valid = self.data['valid'] if 'valid' in self.data else None
            extra = self.data['extra'] if 'extra' in self.data else None
        
        # Get training data as numpy arrays
        (x_train, y_train) = (
            np.concatenate([as_np(d[0]) for d in train], axis=0),
            np.concatenate([as_np(d[1]) for d in train], axis=0))

        # Shuffle
        inds = np.arange(x_train.shape[0])
        np.random.shuffle(inds)
        x_train, y_train = x_train[inds], y_train[inds]
        print('Class dist', np.nanmean(y_train, axis=0))

        # Get validation data if needed
        if valid is None:
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=self.valid_size,
                                                                  random_state=42, stratify=y_train)
            valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(self.chunk_size)
        else:
            valid = valid.unbatch().take(self.valid_size).batch(self.chunk_size)

        self.idg.fit(x_train[:10000].astype(self.dtype))
        tx = np.stack([self.idg.standardize(xi) for xi in np.copy(x_train[:10000])])


        # Fit the generator parameters\
        if self.use_gcn:
            self.norm, self.norm_inv = get_gcn(tx)
        elif self.quantize:
            n_colors = 16 if int(self.quantize) < 2 else int(self.quantize)
            self.norm, self.norm_inv = get_img_clustered(tx, n_colors)
        elif self.use_rescale:
            self.norm, self.norm_inv = get_img_rescale()
        elif self.use_standardize:
            self.norm, self.norm_inv = get_img_standardize(tx)
        elif self.instance_standardize:
            self.norm, self.norm_inv = get_img_instance_standardize()
        elif self.greyscale:
            self.norm, self.norm_inv = get_img_instance_greyscale()
        elif self.use_zca:
            self.norm, self.norm_inv = get_zca_transformer(tx)
        elif self.edges:
            self.norm, self.norm_inv = (lambda x: np.stack([self.filt(xi, sobel) for xi in x])), (lambda x: x)
        elif self.half_size:
            normf, norminv = get_img_rescale()
            self.norm, self.norm_inv = (lambda x: normf(x[:, ::2, ::2])), norminv


        self._shape = self.norm(x_train)[0].shape if self._shape is None else self._shape
        print('Training data shape:', x_train.shape)

        # Get the labeled subset as a separate dataset
        if self.semisupervised:
            mask = self.mask_labels(y_train, self.nlabels, seed=self.seed)
            x_labeled, y_labeled = x_train[mask], y_train[mask]
            x_train, y_train = x_train[np.logical_not(mask)], y_train[np.logical_not(mask)]
            labeled = tf.data.Dataset.from_tensor_slices((x_labeled, y_labeled)).batch(self.chunk_size)
        train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.chunk_size)

        # If we have extra data (e.g. for svhn), add it in here
        if not (extra is None) and self.use_extra:
            train = train.concatenate(extra)

        self._train, self._valid, self._test, self._labeled = train, valid, test, labeled

    def load_data(self, download_dir=None):
        # Fetch and prepare the data, allowing overrides
        if not self.data_fetched:
            self.fetch_data(download_dir=download_dir)
            for k in self.data.keys():
                if not isinstance(self.data[k], tf.data.Dataset):
                    self.data[k] = tf.data.Dataset.from_tensor_slices(self.data[k]).batch(self.chunk_size)
            self.data_fetched = True

        if not self.data_loaded:
            self.prepare_data()
            self.data_loaded = True
        return self

    def fourd(self, x):
        # Make at least 4-d
        if len(x.shape) >= 4:
            return x
        return np.reshape(x, (list(x.shape) + [1, 1])[:4])

    def filt(self, img, f):
        fx = np.concatenate([f(img[:, :, i]).reshape(img[:, :, i].shape + (1,)) for i in range(3)], axis=-1)
        return np.sqrt(np.sum(fx ** 2, axis=-1, keepdims=True))

    def process_batch(self, labeled_datasets, unlabeled_datasets, x_input=True, y_input=True, x_target=True,
                      y_target=True,
                      optimizing=False):
        # Run garbage collection to clean up all the big expensive objects
        gc.collect()

        # Get each next chunk
        updated_labeled, updated_unlabeled = [], []
        for ld in labeled_datasets:
            try:
                updated_labeled.append(next(ld))
            except StopIteration:
                raise RuntimeError()
        for uld in unlabeled_datasets:
            try:
                updated_unlabeled.append(next(uld))
            except StopIteration:
                raise RuntimeError()
        labeled_datasets, unlabeled_datasets = updated_labeled, updated_unlabeled

        # Encode the labels and make unsupervised
        labeled_datasets = [(x, self.encode_labels(y, unsupervised=False)) for (x, y) in labeled_datasets]
        unlabeled_datasets = [(x, np.nan * self.encode_labels(y, unsupervised=True)) for (x, y) in unlabeled_datasets]

        # If we're not balancing just combine everything into one batch
        if not self.balance or not optimizing:
            labeled_datasets = labeled_datasets + unlabeled_datasets
            unlabeled_datasets = []

        # Get the full arrays of labeled data and batch size
        labeled_x, labeled_y = np.concatenate([x for x, y in labeled_datasets], axis=0), np.concatenate(
            [y for x, y in labeled_datasets], axis=0)
        labeled_x = self.fourd(labeled_x)
        batch_size = (self.batch_size // 2, self.batch_size // 2) if len(unlabeled_datasets) and not type(
            self.batch_size) is tuple else self.batch_size
        if not type(batch_size) is tuple:
            batch_size = (batch_size,)
        nbatches = max(labeled_x.shape[0] // batch_size[0], 1)

        # Get the generators
        generators = []
        if optimizing:
            generators.append(self.idg.flow(labeled_x, labeled_y, batch_size=batch_size[0], shuffle=True))
            if len(unlabeled_datasets):
                unlabeled_x, unlabeled_y = np.concatenate([x for x, y in unlabeled_datasets], axis=0), np.concatenate(
                    [y for x, y in unlabeled_datasets], axis=0)
                unlabeled_x = self.fourd(unlabeled_x)
                generators.append(self.idg.flow(unlabeled_x, unlabeled_y, batch_size=batch_size[1], shuffle=True))
                nbatches = max(nbatches, unlabeled_x.shape[0] // batch_size[1])
        else:
            labeled_x = np.stack([self.idg.standardize(xi) for xi in np.copy(labeled_x)])
            generators.append(zip(np.array_split(labeled_x, nbatches), np.array_split(labeled_y, nbatches)))

        # Yield each batch from the generators
        for batch in range(nbatches):
            x, y = [], []
            for datagen in generators:
                xi, yi = next(datagen)
                x.append(xi)
                y.append(yi)

            x, y = [np.concatenate(x, axis=0), np.concatenate(y, axis=0)]

            # Preprocessing functions
            x = self.zca(self.norm(x))
            x1 = x.copy()
            if optimizing and self.random_shift:
                x = random_shift(x, self.random_shift, self.shift_channels)
            if optimizing and self.random_invert:
                x = random_invert(x)
            if optimizing and self.random_flip:
                x = random_flip(x)
            if optimizing and self.random_noise:
                x = x + np.random.randn(*x.shape) * self.random_noise
            if optimizing and self.cutout:
                x = cutout(x, self.cutout)
            if optimizing and self.draw_samples:
                x = x[:, :1] + np.exp(x[:, 1:]) * np.random.randn(*(x[:, 1:].shape))
            elif self.draw_samples:
                x = x[:, :1]
            if self.trim:
                trim = tuple(self.trim)
                if len(trim) == 1:
                    x = x[:, trim[0]:-trim[0]]
                elif len(trim) == 2:
                    x = x[:, trim[0]:-trim[0], trim[1]:-trim[1]]
                elif len(self.trim) == 3:
                    x = x[:, trim[0]:-trim[0], trim[1]:-trim[1], trim[2]:-trim[2]]
            if self.resize_images:
                x = np.stack([skimage.transform.resize(xi, self.resize_images) for xi in x])

            if self.show_inputs and optimizing and np.random.random() < 0.02:
                plot_images(x1)
                plot_images(x)

            input = ([x] if x_input else []) + ([y] if y_input else [])
            target = ([x] if x_target else []) + ([y] if y_target else [])

            input = input[0] if len(input) == 1 else tuple(input)
            target = target[0] if len(target) == 1 else tuple(target)
            yield input, target

    def generator(self, split, semisupervised=True, labeled_only=False, **kwargs):
        # Make sure we have data
        self.load_data()

        semisupervised = self.semisupervised and semisupervised
        labeled_only = self.labeled_only or labeled_only
        labeled = ([self._labeled] if not (self._labeled is None) else [])

        # Get the right combination of labeled and unlabeled data
        labeled_datasets, unlabeled_datasets = [], []
        if split == 'train' or split == 'optimize':
            labeled_datasets = labeled
            if not semisupervised or split == 'train':
                labeled_datasets += [self._train]
        if split == 'valid':
            labeled_datasets = [self._valid]
        if split == 'test':
            labeled_datasets = [self._test]
        if semisupervised and not labeled_only and split == 'optimize':
            unlabeled_datasets += [self._train]

        if split == 'labeled':
            labeled_datasets = labeled
        if split == 'unlabeled':
            labeled_datasets = [self._train]

        # During optimization, batches should be generated indefinitely
        if split == 'optimize':
            labeled_datasets = [ld.repeat() for ld in labeled_datasets]
            unlabeled_datasets = [uld.repeat() for uld in unlabeled_datasets]

        # Setup as iterators
        labeled_datasets = [iter(ld) for ld in labeled_datasets]
        unlabeled_datasets = [iter(uld) for uld in unlabeled_datasets]

        if split == 'optimize':
            while True:
                yield from self.process_batch(labeled_datasets, unlabeled_datasets, optimizing=True, **kwargs)
        else:
            for ld in labeled_datasets:
                while True:
                    try:
                        yield from self.process_batch([ld], [], **kwargs)
                    except StopIteration:
                        break
                    except RuntimeError:
                        break

    def __call__(self, download_dir=None, **kwargs):
        self.load_data(download_dir=download_dir)
        return self

    # Functions to get specific generators
    def optimize(self, labeled_only=False):
        return self.generator('optimize', labeled_only=labeled_only)

    def evaluate(self):
        xshape = tf.TensorShape([None] + list(self.shape()))
        yshape = tf.TensorShape([None, self.classes()])
        return tf.data.Dataset.from_generator(lambda: self.generator(self.split),
                                              output_types=((self.dtype, self.dtype), (self.dtype, self.dtype)),
                                              output_shapes=((xshape, yshape), (xshape, yshape)))

    def predict(self):
        xshape = tf.TensorShape([None] + list(self.shape()))
        yshape = tf.TensorShape([None, self.classes()])
        return tf.data.Dataset.from_generator(lambda: self.generator(self.split, y_input=False, x_target=False),
                                              output_types=(self.dtype, self.dtype),
                                              output_shapes=(xshape, yshape))

    def __iter__(self):
        return self.generator(self.split, y_input=False, x_target=False)

    # Get "splits" of the data by returning a shallow copy with the split flag set.

    def labeled(self):
        other = copy(self)
        other.split = 'labeled'
        return other

    def unlabeled(self):
        other = copy(self)
        other.split = 'unlabeled'
        return other

    def train(self):
        other = copy(self)
        other.split = 'train'
        return other

    def valid(self):
        other = copy(self)
        other.split = 'valid'
        return other

    def test(self):
        other = copy(self)
        other.split = 'test'
        return other

    def get(self, split='test'):
        if split == 'train':
            return self.train()
        elif split == 'valid':
            return self.valid()
        return self.test()

    def numpy(self):
        return np.concatenate([x for x, y in self], axis=0), np.concatenate(
            [y for x, y in self], axis=0)

    def numpy_labels(self):
        return np.concatenate([y for x, y in self], axis=0)

    # Return a copy without the data stored (e.g. for sending to/from a remote server)
    def clean(self):
        other = copy(self)
        other._train = None
        other._valid = None
        other._labeled = None
        other._test = None
        other.data = None
        other.data_fetched = False
        other.data_loaded = False
        return other

    def name(self):
        return self._name

    def shape(self):
        return self._shape

    def classes(self):
        return self._noutputs

    def dim(self):
        return self._noutputs

    def labels(self):
        if hasattr(self, '_labels'):
            return self._labels
        return [str(d) for d in range(self.classes())]

    @classmethod
    def get_all_args(self):
        d = {}
        d.update(get_function_spec(dataset.__init__))
        d.update(get_function_spec(self.__init__))
        return d

    @classmethod
    def help(cls):
        print('Dataset: %s' % cls.__name__)
        print_options(cls.get_all_args())

def make_dataset(x, y, xvalid=None, yvalid=None, xtest=None, ytest=None, name=''):
    class customdataset(dataset, real_dataset, regression_dataset):
        def __init__(self, **kwargs):
            self._shape = x.shape[1:]
            self._name = name
            self._noutputs = y.shape[-1]
            dataset.__init__(self, **kwargs)


        def fetch_data(self, download_dir=None):
            if xvalid is None:
                xv, yv = x, y
            else:
                xv, yv = xvalid, yvalid
            if xtest is None:
                xt, yt = x, y
            else:
                xt, yt = xtest, ytest
            self.data = dict(train=(x, y),
                             valid=(xv, yv), test=(xt, yt))

    return customdataset


class encoded(dataset, real_dataset, classification_dataset):
    def __init__(self, filename, classes=10, **kwargs):
        dataset.__init__(self, **kwargs)
        self._name = 'Encoded'
        self._noutputs = classes
        self._labels = list(range(classes))
        self._args = kwargs
        self._filename = filename
        self.rescale_images = False
        self.draw_samples = True

    def fetch_data(self, download_dir=None):
        data = pickle.load(open(self._filename, "rb"))
        train, valid, testd = data['train'], data['valid'], data['test']
        dim = train['mean'].shape[1]
        train = (
        np.concatenate([train['mean'].reshape((-1, 1, 1, dim)), train['var'].reshape((-1, 1, 1, dim))], axis=1),
        train['labels'])
        testd = (
        np.concatenate([testd['mean'].reshape((-1, 1, 1, dim)), testd['var'].reshape((-1, 1, 1, dim))], axis=1),
        testd['labels'])
        valid = (
        np.concatenate([valid['mean'].reshape((-1, 1, 1, dim)), valid['var'].reshape((-1, 1, 1, dim))], axis=1),
        valid['labels'])

        self.data = dict(train=(train[0], train[1]),
                         valid=(valid[0], valid[1]), test=(testd[0], testd[1]))


