from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import inspect
import sys, os
from sklearn.cluster import KMeans as KMeans
from sklearn.metrics import pairwise_distances_argmin
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit
from skimage.color import rgb2gray
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt
from datetime import datetime

def current_time():
    now = datetime.now()
    return now.strftime("%Y_%m_%d__%H_%M")

def get_mask(input_dim, output_dim, kernel_size=3, mask_type=None):
    if mask_type is None:
        return lambda x: x

    mask_type, mask_n_channels = mask_type, 3

    mask = np.ones(
        (kernel_size, kernel_size, input_dim, output_dim),
        dtype='float32'
    )
    center = kernel_size // 2

    # Mask out future locations
    # filter shape is (height, width, input channels, output channels)
    mask[center + 1:, :, :, :] = 0.
    mask[center, center + 1:, :, :] = 0.

    # Mask out future channels
    for i in range(mask_n_channels):
        for j in range(mask_n_channels):
            if (mask_type == 'a' and i >= j) or (mask_type == 'b' and i > j):
                mask[
                center,
                center,
                i::mask_n_channels,
                j::mask_n_channels
                ] = 0.
    return lambda x: x * mask

class AddWeights(tf.keras.layers.Layer):
    def __init__(self):
        super(AddWeights, self).__init__()
        self.a = self.add_weight(shape=tuple(), initializer="zeros", trainable=True)

    def call(self, inputs):
        return inputs + 0. * self.a

class LinearReshape(tf.keras.layers.Layer):
    def __init__(self, output_shape=None, name='lreshape'):
        super(LinearReshape, self).__init__(name=name)
        self.output_reshape = output_shape

    def build(self, input_shape):
        input_shape = input_shape[1:]
        matched_dims = 0
        for di, do in zip(input_shape[:-1], self.output_reshape[:-1]):
            if di == do:
                matched_dims += 1
        self.flatdim_in = int(np.prod(input_shape[matched_dims:]))
        self.flatdim_out = int(np.prod(self.output_reshape[matched_dims:]))
        self.matched_dims = input_shape[:matched_dims]

        self.w = self.add_weight(
            shape=(self.flatdim_in, self.flatdim_out),
            initializer="glorot_normal",
            trainable=True, name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.flatdim_out,), initializer="glorot_normal", trainable=True, name='bias'
        )

    def call(self, inputs):
        inputs = tf.keras.layers.Reshape(self.matched_dims + (self.flatdim_in,)).call(inputs)
        output = tf.matmul(inputs, self.w) + self.b
        return tf.keras.layers.Reshape(self.output_reshape).call(output)

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def joinpath(*args, **kwargs):
    path = os.path.join(*args)
    try:
        os.makedirs(path)
    except:
        pass

    if 'file' in kwargs:
        path = os.path.join(path, kwargs['file'])
    return path

def as_tuple(x):
    try:
        return tuple(x)
    except:
        return (x,)

def as_np(x):
    if not isinstance(x, np.ndarray):
        try:
            return x.numpy()
        except:
            return np.array(x)
    return x

def interleave(a, b, axis=0):
    a_shape, b_shape = tuple([(-1 if d is None else d) for d in a.shape]), tuple(b.shape)
    out = tf.concat([tf.expand_dims(a, axis + 1), tf.expand_dims(b, axis + 1)], axis=axis+1)
    return tf.reshape(out, a_shape[:axis] + (a_shape[axis] + b_shape[axis],) + a_shape[(axis+1):])

def random_shift(x, shift_range, channels=False):
    # Randomly shift a batch of images by a few pixels
    ax = [1, 2, 3] if channels else [1, 2]
    for axis in ax:
        shifts = (np.random.random(x.shape[0]) * (shift_range + 1)).astype(int)
        for s in range(shift_range + 1):
            shift = s - (shift_range // 2)
            if shift != 0:
                x[shifts == shift] = np.roll(x[shifts == shift], shift, axis=axis)
    return x


def random_flip(x):
    toflip = np.random.randint(2, size=x.shape[0]).astype(bool)
    x[toflip] = np.flip(x[toflip], axis=2)
    return x

def random_invert(x):
    toflip = np.random.randint(2, size=x.shape[0]).astype(bool)
    x[toflip] = 1.0 - x[toflip]
    return x


def get_gcn(x, multiplier=55., eps=1e-8):
    def transform(x):
        mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
        denominator = np.sqrt(
            np.sum((x - mean) ** 2, axis=(1, 2, 3), keepdims=True)
        )
        denominator /= multiplier
        denominator[denominator < eps] = 1.
        return (x - mean) / denominator
    return transform, lambda a: a

def get_img_rescale():
    return (lambda x: (x * 2.) - 1.), (lambda x: (x + 1.) / 2.)

def get_img_clustered(x, n_colors=16):
    channels = x.shape[-1]
    x = x.reshape((-1, x.shape[-1]))
    rs = np.random.RandomState(543)
    x = x[rs.randint(0, x.shape[0], size=10000)]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(x)

    def quantize(img):
        shape = img.shape[:-1]
        img = img.reshape((-1, channels))
        img = kmeans.predict(img)
        imgnew = np.zeros((img.shape[0], n_colors))
        for c in range(n_colors):
            imgnew[img == c, c] = 1
        img = imgnew.reshape(shape + (n_colors,))
        return img

    def unquantize(img):
        shape = img.shape[:-1] + (channels,)
        img = img.reshape((-1, n_colors))
        img = np.dot(img, kmeans.cluster_centers_).reshape(shape)
        return img

    return quantize, unquantize

def get_img_standardize(x):
    mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
    dev = np.std(x, axis=(0, 1, 2), keepdims=True)
    return (lambda a: (a - mean) / dev), (lambda a: (a * dev) + mean)

def get_img_instance_standardize():
    def trans(x):
        mean = np.mean(x, axis=(-3, -2), keepdims=True)
        dev = np.std(x - mean, axis=(-3, -2), keepdims=True)
        return (x - mean) / dev

    def inv(x):
        return expit(x)
    return trans, inv

def get_img_instance_greyscale():
    def trans(x):
        if x.shape[-1] == 1:
            x = np.concatenate([x, x, x], axis=-1)
        mean = np.mean(x, axis=(-3, -2), keepdims=True)
        dev = np.std(x - mean, axis=(-3, -2), keepdims=True)
        return np.expand_dims(rgb2gray(expit((x - mean) / dev) * 2 - 1), -1)

    def inv(x):
        return np.clip((x + 1) / 2., 0. , 1.)
    return trans, inv

def get_constrained_zca_transformer(images, identity_scale=0.1, eps=1e-10, get_inv=True):
    shape = images.shape
    images = images.reshape((shape[0], -1))
    images = images - images.mean(axis=1, keepdims=True)
    per_image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    per_image_norm[per_image_norm < 1e-10] = 1
    images = 55 * images / per_image_norm
    image_covariance = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(
        image_covariance + identity_scale * np.eye(*image_covariance.shape)
    )
    zca_decomp = np.dot(U, np.dot(np.diag(1. / np.sqrt(S + eps)), U.T))
    zca_inv = np.linalg.inv(zca_decomp)
    image_mean = images.mean(axis=0)
    def apply_zca(x):
        x = x.reshape((x.shape[0], -1))
        x = x - x.mean(axis=1, keepdims=True)
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        x_norm[x_norm < 1e-10] = 1
        x = 55 * x / x_norm
        return np.tanh(np.dot(x - image_mean, zca_decomp).reshape((-1,) + shape[1:]) / 3)

    def undo(x):
        x = x.reshape((x.shape[0], -1))
        x = np.arctanh(np.clip(x, 0.001, 0.999)) * 3
        x = np.dot(x, zca_inv) + image_mean
        x = x - x.min(axis=1, keepdims=True)
        x = x / x.max(axis=1, keepdims=True)
        return x.reshape((-1,) + shape[1:])
    return apply_zca, undo


def get_zca_transformer(images, identity_scale=0.1, eps=1e-10, get_inv=True):
    shape = images.shape
    images = images.reshape((shape[0], -1))
    images = images - images.mean(axis=1, keepdims=True)
    per_image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    per_image_norm[per_image_norm < 1e-10] = 1
    images = 55 * images / per_image_norm
    image_mean = images.mean(axis=0, keepdims=True)
    images = images - image_mean
    image_covariance = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(
        image_covariance + identity_scale * np.eye(*image_covariance.shape)
    )
    zca_decomp = np.dot(U, np.dot(np.diag(1. / np.sqrt(S + eps)), U.T))
    zca_inv = np.linalg.inv(zca_decomp)

    def apply_zca(x):
        x = x.reshape((x.shape[0], -1))
        x = x - x.mean(axis=1, keepdims=True)
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        x_norm[x_norm < 1e-10] = 1
        x = 55 * x / x_norm
        return np.dot(x - image_mean, zca_decomp).reshape((-1,) + shape[1:])

    def undo(x):
        x = x.reshape((x.shape[0], -1))
        x = np.dot(x, zca_inv) + image_mean
        x = x - x.min(axis=1, keepdims=True)
        x = x / x.max(axis=1, keepdims=True)
        return x.reshape((-1,) + shape[1:])

    return apply_zca, undo

def cutout(img, size=16):
  n = img.shape[0]
  h = img.shape[1]
  w = img.shape[2]
  hstart = np.random.randint(h - size, size=n)
  wstart = np.random.randint(w - size, size=n)

  output = np.copy(img)
  for ni in range(n):
    hs = hstart[ni]
    ws = wstart[ni]
    output[ni, hs:(hs + size), ws:(ws+size), :] = np.nan
  return output

def list_models():
    import pcvae.models as models
    print('Available models:')
    for k, m in models.__dict__.items():
        try:
            if issubclass(m, models.BaseVAE) and m.name() != models.BaseVAE.name():
                print('\t' + m.__name__)
        except:
            pass
    print('\n')


def list_datasets():
    import pcvae.datasets as datasets
    print('Available datasets:')
    for m in datasets.__dict__.values():
        try:
            if issubclass(m, datasets.dataset) and m.__name__ != datasets.dataset.__name__:
                print('\t' + m.__name__)
        except:
            pass
    print('\n')


def list_options(dataset=None, model=None, constructor=None):
    from pcvae.experiments import base_constructor
    if not (model is None) and constructor is None:
        constructor = base_constructor
    print('PC-VAE: A package for Prediction-Constrained and Semi-Supervised VAE learning\n')

    if dataset is None and model is None and constructor is None:
        print('To see options, pass in a dataset, model and/or a network constructor')
        print('E.g.\n\tlist_options(dataset=cifar, model=PC, constructor=base_constructor)\n')
    else:
        if not (dataset is None):
            dataset.help()
        if not (model is None):
            model.help()
        if not (constructor is None):
            constructor.help()


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered.
    """

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')

        if loss is not None:
            try:
                if np.isnan(loss) or np.isinf(loss):
                    print('Batch %d: Invalid loss, terminating training' % (batch))
                    self.model.stop_training = True
            except:
                print(logs)
                raise


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def binary_crossentropy(target, output):
    output = tf.cast(target >= 0, tf.float32) * K.binary_crossentropy(target, output)
    output = tf.where(tf.math.is_nan(output), tf.zeros_like(output), output)
    return K.sum(output, axis=-1)

def categorical_crossentropy(target, output):
    output =  K.categorical_crossentropy(target, output)
    output = tf.where(tf.math.is_nan(output), tf.zeros_like(output), output)
    return K.sum(output, axis=-1)

def select_labeled_binary(y, nlb):
    inds = [np.random.randint(0, len(y))]
    all_inds = np.arange(len(y))
    for i in range(nlb):
        picked = 1 - y[inds].mean(axis=0, keepdims=True)
        dist = (y * picked * (y > 0)).sum(axis=1)
        dist[inds] = 1e-10
        dist = dist / dist.sum()
        inds.append(np.random.choice(all_inds, p=dist))
    return inds


def flatdict(d, out=None):
    if out is None:
        out = {}
    for k, v in d.items():
        if type(v) is dict:
            flatdict(v, out)
        else:
            out[k] = v
    return out


def l_or_t(x):
    return type(x) is list or type(x) is tuple


def rmap(f, inputs):
    return [(rmap(f, x) if l_or_t(x) else f(x)) for x in inputs]


def flatlist(inputs):
    outputs = []
    for x in inputs:
        if l_or_t(x):
            outputs.extend(flatlist(x))
        else:
            outputs.append(x)
    return outputs


class scipy_min:
    def __init__(
            self,
            fun=None,
            x=[],
            args=[],
            callback=None,
            method="L-BFGS-B",
            loss_callback=None,
            overwrite=True,
            options={},
    ):
        self.fun = fun
        self.sizes = [xi.size for xi in x]
        self.shapes = [xi.shape for xi in x]
        self.x = x if type(x) is list or type(x) is tuple else [x]
        self.args = args
        self.options = options
        self.callback = callback
        self.loss_callback = loss_callback
        self.overwrite = overwrite
        self.result = minimize(
            self.call_func,
            self.wrap(self.x),
            self.args,
            method,
            True,
            callback=self.callback,
            options=options,
        )
        self.unwrap(self.result.x)

    def wrap(self, x):
        return np.concatenate([xi.flatten() for xi in x])

    def unwrap(self, x):
        uw, start, end = [], 0, 0
        for size, shape in zip(self.sizes, self.shapes):
            end += size
            uw.append(x[start:end].reshape(shape))
            start = end

        if self.overwrite:
            for dst, src in zip(self.x, uw):
                np.copyto(dst, src)
        else:
            self.x = uw

    def call_func(self, x, *args):
        self.unwrap(x)
        out = self.fun(*(self.x + self.args))
        obj, grad = out[0], out[1:]
        if not (self.loss_callback is None):
            self.loss_callback(obj)
        return obj, self.wrap(grad)

    def cb(self, x):
        if not (self.callback is None):
            self.unwrap(x)
            self.callback(self.x)


class DummyFile(object):
    def write(self, x): pass

    def flush(self): pass


class progressbar:
    def __init__(
            self,
            maxval=1.0,
            print_vals=True,
            round_vals=False,
            width=30,
            digits=3,
            inplace=True,
            blanksym=".",
            barsym="=",
            endcap=">",
            stopsym="|",
            incompsym=" ",
    ):
        try:
            self.maxval = len(maxval)
            self.iterable = maxval
        except:
            self.maxval = maxval
        self.print_vals = print_vals
        self.width = width
        self.blanksym = blanksym
        self.barsym = barsym
        self.endcap = endcap
        self.fformat = "%." + str(digits) + "f"
        self.round_vals = round_vals
        self.count = 0
        self.val = 0
        self.lastnum = -1
        self.inplace = inplace
        self.save_stdout = sys.stdout
        self.stopsym = stopsym
        self.incompsym = incompsym

    def __enter__(self):
        self.save_stdout = sys.stdout
        sys.stdout = DummyFile()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.save_stdout
        self.end()

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.maxval:
            self.end()
            raise StopIteration()

        item = self.iterable[self.count]
        self.update()
        return item

    def update(self, val=None, extra="", end=False):
        if val is None:
            self.count += 1
            val = self.count
        self.val = val

        output = "\r" if self.inplace else "\n"
        if self.round_vals:
            v, mv = str(int(val)), str(int(self.maxval))
        else:
            v, mv = (self.fformat % float(val)), (self.fformat % float(self.maxval))

        v = " " * (len(mv) - len(v)) + v
        if self.print_vals:
            output += v + "/" + mv + " "

        num = int(round(min(float(val) / float(self.maxval), 1.0) * self.width))
        if end and num >= self.width:
            output += "[" + (self.width * self.barsym) + "] "
        elif num <= 0:
            output += "[" + (self.width * self.blanksym) + "] "
        else:
            output += (
                    "["
                    + ((num - 1) * self.barsym)
                    + (self.endcap if not end else self.stopsym)
                    + ((self.width - num) * (self.blanksym if not end else self.incompsym))
                    + "] "
            )

        output += extra
        if self.inplace or num > self.lastnum:
            self.save_stdout.write(output)
            self.save_stdout.flush()
        self.lastnum = num

    def end(self, extra=""):
        self.update(self.val, extra, True)
        self.save_stdout.write("\n")
        self.count = 0


def get_function_spec(func, skip=[]):
    args = inspect.getfullargspec(func).args
    defaults = inspect.getfullargspec(func).defaults
    defaults = [] if defaults is None else list(defaults)
    defaults = (len(args) - len(defaults)) * [None] + defaults

    spec = []
    for arg, default in zip(args, defaults):
        if (not (arg in skip)) and (not (arg in ["self", "cls"])):
            if default is None:
                spec.append((arg, dict()))
            else:
                spec.append(
                    (
                        arg,
                        dict(
                            type=type(default),
                            default=default,
                            help="",
                        ),
                    )
                )
    return dict(spec)


def print_options(opts):
    print("Available options (defaults shown as ...[default]):")
    ostr = lambda s: s + ":" + " ".join(max(20 - len(s), 1) * [""])
    hstr = lambda v: (v["help"] if "help" in v else "") + (
        (" [%s]" % str(v["default"])) if "default" in v else ""
    )
    fromo = lambda d: "\n\t" + "\n\t".join([ostr(k) + " " + hstr(v) for k, v in d])
    print(fromo(opts.items()))
    print('\n')


def butter_lowpass(cutOff=0.5, fs=50, order=3):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = False)
    return b, a

def butter_lowpass_filter(data, cutOff=0.5, fs=50, order=3):
    b, a = butter_lowpass(cutOff, fs, order=order)
    zi = lfilter_zi(b, a)
    y = filtfilt(b, a, data)
    return y

def remove_gravity_data(data):
    return ([xi - butter_lowpass_filter(xi.T).T for xi in data[0]],) + data[1:]
