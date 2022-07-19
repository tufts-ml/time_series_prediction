from urllib.request import Request, urlopen
import pickle, gzip
from .base import dataset, input_dataset, quantized_dataset, classification_dataset, binary_classification_dataset, \
    binary_dataset, \
    real_dataset, remote_dataset, tensorflow_dataset, uniform_dataset
import tensorflow_datasets
from .birds import CaltechBirds
from .celeb import CelebAFixed
import urllib

class kingma_mnist(dataset, binary_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'KINGMA_MNIST'
        self._noutputs = 10
        self._labels = [str(i) for i in range(10)]

    def fetch_data(self, download_dir=None):
        req = Request('https://github.com/dpkingma/nips14-ssl/raw/master/data/mnist/mnist_28.pkl.gz')
        req.add_header('Accept-Encoding', 'gzip')
        response = urlopen(req)
        content = gzip.decompress(response.read())
        train, valid, test = pickle.loads(content, encoding='latin1')
        self.data = dict(train=(train[0].reshape((-1, 28, 28, 1)), train[1]),
                         valid=(valid[0].reshape((-1, 28, 28, 1)), valid[1]), test=(
                test[0].reshape((-1, 28, 28, 1)), test[1]))

class kingma_mnist_real(dataset, uniform_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'KINGMA_MNIST'
        self._noutputs = 10
        self._labels = [str(i) for i in range(10)]
        self.use_rescale = True

    def fetch_data(self, download_dir=None):
        req = Request('https://github.com/dpkingma/nips14-ssl/raw/master/data/mnist/mnist_28.pkl.gz')
        req.add_header('Accept-Encoding', 'gzip')
        response = urlopen(req)
        content = gzip.decompress(response.read())
        train, valid, test = pickle.loads(content, encoding='latin1')
        self.data = dict(train=(train[0].reshape((-1, 28, 28, 1)), train[1]),
                         valid=(valid[0].reshape((-1, 28, 28, 1)), valid[1]), test=(
                test[0].reshape((-1, 28, 28, 1)), test[1]))

class mnist(dataset, tensorflow_dataset, binary_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'MNIST'
        self._noutputs = 10
        self.tf_dataset = tensorflow_datasets.image.mnist.MNIST


class mnist_real(dataset, tensorflow_dataset, uniform_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'MNIST'
        self._noutputs = 10
        self.tf_dataset = tensorflow_datasets.image.mnist.MNIST
        self.use_rescale = True


class fashion_mnist(dataset, tensorflow_dataset, uniform_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'Fashion_MNIST'
        self._noutputs = 10
        self._labels = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandle', 'shirt', 'sneaker', 'bag',
                        'ankle boot']
        self.tf_dataset = tensorflow_datasets.image.mnist.FashionMNIST
        self.use_rescale = True

class fashion_mnist_grey(dataset, tensorflow_dataset, uniform_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'Fashion_MNIST'
        self._noutputs = 10
        self._labels = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandle', 'shirt', 'sneaker', 'bag',
                        'ankle boot']
        self.tf_dataset = tensorflow_datasets.image.mnist.FashionMNIST
        self.greyscale = True

class cifar(dataset, tensorflow_dataset, real_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'CIFAR10'
        self._noutputs = 10
        self._labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.tf_dataset = tensorflow_datasets.image.cifar.Cifar10
        self.use_standardize = False
        self.use_zca = False
        self.rescale_images = True
        self.use_rescale = True


class cifar_zca(dataset, tensorflow_dataset, real_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'CIFAR10'
        self._noutputs = 10
        self._labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.tf_dataset = tensorflow_datasets.image.cifar.Cifar10
        self.use_standardize = False
        self.use_zca = False
        self.rescale_images = False
        self.use_zca = True


class caltech_birds(dataset, tensorflow_dataset, real_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'BIRDS'
        self._noutputs = 312
        self.tf_dataset = CaltechBirds


class celeba(dataset, tensorflow_dataset, real_dataset, binary_classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'CELEB'
        self._noutputs = 40
        self.tf_dataset = tensorflow_datasets.image.CelebA
        self.trim = (21, 1)
        self._shape = (176, 176, 3)
        self.use_rescale = True
        self.label_group = 'attributes'


class celeba_gender(dataset, tensorflow_dataset, real_dataset, binary_classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'CELEB'
        self._noutputs = 1
        self.tf_dataset = tensorflow_datasets.image.CelebA
        self.xkey = 'image'
        self.ykey = 'Male'
        self.trim = (21, 1)
        self._shape = (176, 176, 3)
        self.register_checksums = True
        self.use_rescale = True
        self.label_group = 'attributes'

class celeba_gender_smiling(dataset, tensorflow_dataset, real_dataset, binary_classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'CELEB'
        self._noutputs = 2
        self.tf_dataset = tensorflow_datasets.image.CelebA
        self.xkey = 'image'
        self.ykey = ['Male', 'Smiling']
        self.trim = (21, 1)
        self._shape = (176, 176, 3)
        self.register_checksums = True
        self.use_rescale = True
        self.label_group = 'attributes'

class norb(dataset, tensorflow_dataset, real_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'NORB'
        self._noutputs = 5
        self.tf_dataset = tensorflow_datasets.image.smallnorb.Smallnorb
        self.half_size = True
        self.use_rescale = False


class SvhnNoExtra(tensorflow_datasets.image.svhn.SvhnCropped):
    # SVHN without the "extra" split that just takes forever to download and preprocess
    def _split_generators(self, dl_manager):
        URL = "http://ufldl.stanford.edu/housenumbers/"
        output_files = dl_manager.download({
            "train": urllib.parse.urljoin(URL, "train_32x32.mat"),
            "test": urllib.parse.urljoin(URL, "test_32x32.mat"),
        })

        return [
            tensorflow_datasets.core.SplitGenerator(
                name=tensorflow_datasets.Split.TRAIN,
                num_shards=10,
                gen_kwargs=dict(
                    filepath=output_files["train"],
                )),
            tensorflow_datasets.core.SplitGenerator(
                name=tensorflow_datasets.Split.TEST,
                num_shards=1,
                gen_kwargs=dict(
                    filepath=output_files["test"],
                )),
        ]


class svhn(dataset, tensorflow_dataset, real_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.use_rescale = True


class svhn_grey(dataset, tensorflow_dataset, uniform_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.greyscale = True


class svhn_binary(dataset, tensorflow_dataset, binary_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.greyscale = True
        self._shape = (32, 32, 1)


class svhn_edge(dataset, tensorflow_dataset, uniform_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.use_rescale = False
        self.edges = True
        self._shape = (32, 32, 1)


class svhn_istd(dataset, tensorflow_dataset, real_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.instance_standardize = True
        self.use_zca = False
        self.rescale_images = False


class svhn_std(dataset, tensorflow_dataset, real_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.use_standardize = True
        self.use_zca = False
        self.rescale_images = False


class svhn_uniform(dataset, tensorflow_dataset, uniform_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.use_rescale = False


class svhn_zca(dataset, tensorflow_dataset, uniform_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.use_rescale = False
        self.use_standardize = False
        self.use_zca = False
        self.rescale_images = False
        self.use_zca = True


class svhn_all(dataset, tensorflow_dataset, real_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.use_rescale = True
        self.tf_dataset = tensorflow_datasets.image.svhn.SvhnCropped


class svhn_q16(dataset, tensorflow_dataset, quantized_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.quantize = 16
        self.use_rescale = False


class svhn_q32(dataset, tensorflow_dataset, quantized_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.quantize = 32
        self.use_rescale = False


class svhn_q64(dataset, tensorflow_dataset, quantized_dataset, classification_dataset):
    def __init__(self, *args, **kwargs):
        dataset.__init__(self, *args, **kwargs)
        self._name = 'SVHN'
        self._noutputs = 10
        self.tf_dataset = SvhnNoExtra
        self.quantize = 64
        self.use_rescale = False



