import dill as pickle
import copy
import os
import numpy as np
import tensorflow as tf
import glob
from .base import dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(x, y):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {'data': _bytes_feature(tf.io.serialize_tensor(x)), 'label': _bytes_feature(tf.io.serialize_tensor(y))}
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(x, y):
    tf_string = tf.py_function(
        serialize_example,
        (x, y),  # pass these args to the above function.
        tf.string)  # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar


def serialize_dataset(dataset):
    dataset = tf.data.Dataset.from_generator(dataset, output_types=(tf.float32, tf.float32))
    return dataset.unbatch().map(tf_serialize_example)


# Create a dictionary describing the features.
image_feature_description = {
    'data': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    serialized = tf.io.parse_single_example(example_proto, image_feature_description)
    return tf.io.parse_tensor(serialized['data'], tf.float32), tf.io.parse_tensor(serialized['label'], tf.float32)


class CachedDataset(dataset):

    def __init__(self):
        self.split = 'test'
        self.dtype = np.float32
        self.batch_size = 32
        self.balance = True
        self.seed = 0
        self.nlabels = 0
        self._name = 'data'
        self.args = {}
        self.shuffle = True
        self.compress = 'ZLIB'

    @classmethod
    def from_dataset(cls, data):
        newdata = cls()
        data.load_data()
        newdata._shape = next(iter(data.test()))[0][0].shape[:]
        newdata.nlabels = data.nlabels
        print('Caching dataset: %s' % data._name)
        print('\tshape: %s' % str(newdata._shape))
        print('\tnlabels: %s' % str(data.nlabels))
        newdata._train = serialize_dataset(data.unlabeled())
        newdata._valid = serialize_dataset(data.valid())
        newdata._test = serialize_dataset(data.test())
        newdata._labeled = serialize_dataset(data.labeled()) if data.semisupervised else None
        newdata.norm = data.norm
        newdata.norm_inv = data.norm_inv
        newdata.semisupervised = data.semisupervised
        newdata.seed = data.seed
        newdata._noutputs = data._noutputs
        newdata.get_metrics = lambda: []  # data.get_metrics
        newdata.imagedata = data.imagedata
        if hasattr(data, '_name'):
            newdata._name = data._name
        if hasattr(data, '_labels'):
            newdata._labels = data._labels
        return newdata

    @classmethod
    def cache(cls, data, filepath=None, identifier='', compress=False):
        data = cls.from_dataset(data)
        data.compress = 'ZLIB' if compress else None
        filepath, filename = data.save(filepath=filepath, identifier=identifier)
        return cls.load(filepath=filepath, filename=filename)

    @classmethod
    def load(cls, name='data', identifier='*', nlabels='*', seed='*', filepath=None, filename=None):
        if filepath is None:
            filepath = os.path.join(os.path.expanduser('~'), 'pcvae_datasets', name)
        if filename is None:
            filename = '%s_%s_nlabels_%s_seed_%s.pkl' % (name, identifier, str(nlabels), str(seed))
        filename = os.path.join(filepath, filename)
        filename = glob.glob(filename)[0]

        with open(filename, 'rb') as reader:
            newdata = pickle.load(reader)

        newdata._train = newdata.load_data_file(newdata._train)
        newdata._test = newdata.load_data_file(newdata._test)
        if newdata._valid:
            newdata._valid = newdata.load_data_file(newdata._valid)
        if newdata._labeled:
            newdata._labeled = newdata.load_data_file(newdata._labeled)
        return newdata

    def __call__(self, balance=True, batch_size=32, shuffle=True, **kwargs):
        self.balance = balance
        self.batch_size = batch_size
        self.args = kwargs
        self.shuffle = shuffle
        return self

    def load_data(self, **kwargs):
        return self

    def load_data_file(self, filename):
        return tf.data.TFRecordDataset(filename, compression_type=self.compress)

    def save_data(self, split, filepath, identifier=''):
        data = \
        dict(train=self._train, valid=self._valid, test=self._test, labeled=self._labeled, unlabeled=self._train)[split]
        filename = '%s_%s_%s_nlabels_%d_seed_%d.tfrecord' % (
        self._name, identifier, split, max(self.nlabels, 0), self.seed)
        filename = os.path.join(filepath, filename)
        writer = tf.data.experimental.TFRecordWriter(filename, compression_type=self.compress)
        writer.write(data)
        return filename

    def save(self, filepath=None, identifier=''):
        if filepath is None:
            filepath = os.path.join(os.path.expanduser('~'), 'pcvae_datasets', self._name)
        try:
            os.makedirs(filepath)
        except:
            pass

        savedata = copy.copy(self)
        savedata._train = self.save_data('train', filepath, identifier)
        savedata._test = self.save_data('test', filepath, identifier)
        savedata._valid = self.save_data('valid', filepath, identifier)
        if self._labeled:
            savedata._labeled = self.save_data('labeled', filepath, identifier)
        filename = '%s_%s_nlabels_%d_seed_%d.pkl' % (self._name, identifier, max(self.nlabels, 0), self.seed)
        full_filename = os.path.join(filepath, filename)
        with open(full_filename, 'wb') as writer:
            pickle.dump(savedata, writer)
        print('Saved dataset to: %s' % filename)
        return filepath, filename

    def generator(self, split, y_input=True, x_target=True, y_target=True, **kwargs):
        data = \
        dict(train=self._train, valid=self._valid, test=self._test, labeled=self._labeled, unlabeled=self._train)[split]
        if self.semisupervised:
            data['train'] = self._labeled.concatenate(self._train)
        data = data.map(_parse_image_function)

        inputs, targets = data, data
        if not y_input:
            inputs = inputs.map(lambda x, y: x)
        if not x_target and y_target:
            targets = targets.map(lambda x, y: y)
        if not y_target and x_target:
            targets = targets.map(lambda x, y: x)
        if not y_target and not x_target:
            targets = targets.map(lambda x, y: tuple())

        for d in tf.data.Dataset.zip((inputs.batch(self.batch_size), targets.batch(self.batch_size))):
            inputs, targets = d[0], d[1]
            try:
                inputs = inputs.numpy()
            except:
                inputs = tuple([di.numpy for di in inputs])

            try:
                targets = targets.numpy()
            except:
                targets = tuple([di.numpy for di in targets])

            yield (inputs, targets)

    def evaluate(self):
        data = \
        dict(train=self._train, valid=self._valid, test=self._test, labeled=self._labeled, unlabeled=self._train)[
            self.split]
        data = data.map(_parse_image_function)
        return tf.data.Dataset.zip((data, data)).batch(32)

    def predict(self):
        data = \
        dict(train=self._train, valid=self._valid, test=self._test, labeled=self._labeled, unlabeled=self._train)[
            self.split]
        data = data.map(_parse_image_function)
        datax = data.map(lambda x, y: x)
        datay = data.map(lambda x, y: y)
        return tf.data.Dataset.zip((datax, datay)).batch(32)

    def augment(self, image, label):
        args = dict(resize=None, random_flip=None, random_crop=None, random_noise=None, random_brightness=None,
                    random_contrast=None, random_saturation=None, random_hue=None)
        args.update(self.args)
        shape = self._shape
        image = tf.reshape(image, self._shape)
        label = tf.reshape(label, (self._noutputs,))
        if args['resize']:
            shape = args['resize']
            image = tf.image.resize(image, [shape[0], shape[1]])
        if args['random_flip']:
            image = tf.image.random_flip_left_right(image)
        if args['random_brightness']:
            image = tf.image.random_brightness(image, max_delta=args['random_brightness'])
            image = tf.clip_by_value(image, 0, 1)
        if args['random_hue']:
            image = tf.image.random_hue(image, max_delta=args['random_hue'])
            image = tf.clip_by_value(image, 0, 1)
        if args['random_saturation']:
            image = tf.image.random_saturation(image, *args['random_saturation'])
            image = tf.clip_by_value(image, 0, 1)
        if args['random_contrast']:
            image = tf.image.random_contrast(image, *args['random_contrast'])
            image = tf.clip_by_value(image, 0, 1)
        if args['random_noise']:
            image = image + tf.random.normal(tf.shape(image), 0, args['random_noise'])
            image = tf.clip_by_value(image, 0, 1)
        if args['random_crop']:
            cs = args['random_crop']
            image = tf.image.resize_with_crop_or_pad(image, shape[0] + cs, shape[1] + cs)
            image = tf.image.random_crop(image, size=shape)
        return image, label

    def optimize(self, labeled_only=False):
        data = self._train
        if not self.semisupervised:
            data = data.map(_parse_image_function)
        elif labeled_only:
            data = self._labeled.map(_parse_image_function).repeat()
        elif self.semisupervised:
            data = data.map(_parse_image_function).map(lambda x, y: (x, y * np.nan))
            if self.balance:
                data = tf.data.experimental.sample_from_datasets([data, self._labeled.map(_parse_image_function)])
            else:
                data = data.concatenate(self._labeled.map(_parse_image_function)).repeat()

        if self.shuffle:
            data = data.shuffle(10000)
        data = data.map(self.augment)
        data = data.map(lambda x, y: ((x, y), (x, y)))
        data = data.batch(self.batch_size).repeat()
        return data.prefetch(AUTOTUNE)

