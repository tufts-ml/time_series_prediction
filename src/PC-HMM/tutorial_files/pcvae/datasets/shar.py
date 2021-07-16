import os
import numpy as np
from scipy.io import loadmat
from .base import dataset, real_dataset, classification_dataset
from ..util.util import remove_gravity_data

def load_shar(basedir="UniMiB-SHAR/", version="adl", folds=10, random_split=False, **kwargs):
    full_data = loadmat(os.path.join(basedir, "data", "%s_data.mat" % version))[
        "%s_data" % version
        ]
    labids = loadmat(os.path.join(basedir, "data", "%s_labels.mat" % version))[
        "%s_labels" % version
        ]
    labs, ids = labids[:, 0], labids[:, 1]
    names = loadmat(os.path.join(basedir, "data", "%s_names.mat" % version))[
        "%s_names" % version
        ]

    data, labels, meta = [], [], []
    names = [str(n[0]) for n in names[:, 0]]
    for di, li, si in zip(full_data, labs, ids):
        labels.append(li - 1)
        data.append(np.stack([di[:151], di[151:302], di[302:]]).T)
        si = int(np.random.random() * folds) if random_split else si
        meta.append({"subject": si - 1, "cv": (si - 1) % folds, "labels": names})
    return data, labels, meta


def get_cv_split(data, split=0, gensplits=0, seed=543, cv_semisup=0, key="cv", cv_valid=0, **kwargs):
    xtrain, ytrain, ztrain = [], [], []
    xvalid, yvalid, zvalid = [], [], []
    xtest, ytest, ztest = [], [], []
    x, y, z = data
    nsplits = max([int(zi[key]) for zi in z if key in zi]) + 1

    if gensplits:
        order = np.array([i % gensplits for i in range(len(x))], dtype=int)
        semisup_prng = np.random.RandomState(seed)
        semisup_prng.shuffle(order)
        z = [dict(cv=i) for i in order]

    for xi, yi, zi in zip(x, y, z):
        try:
            in_split = key in zi and (zi[key] == split or zi[key] in split)
        except:
            in_split = False

        if in_split:
            xtest.append(xi)
            ytest.append(yi)
            ztest.append(zi)
        elif key in zi and cv_valid > 0 and zi[key] in [((split + cvi) % nsplits) for cvi in
                                                        range(abs(cv_semisup) + 1, abs(cv_semisup) + 1 + cv_valid)]:
            xvalid.append(xi)
            yvalid.append(yi)
            zvalid.append(zi)
        elif key in zi:
            yi = yi * 0 - 1 if "mask" in zi and zi["mask"] else yi
            if cv_semisup > 0:
                yi = (
                    yi
                    if key in zi and zi[key] in [((split + cvi) % nsplits) for cvi in range(1, cv_semisup + 1)]
                    else yi * 0 - 1
                )
            if cv_semisup < 0:
                zi['remove'] = not (key in zi and zi[key] in [((split + cvi) % nsplits) for cvi in
                                                              range(1, abs(cv_semisup) + 1)])
            if "remove" not in zi or not zi["remove"]:
                xtrain.append(xi)
                ytrain.append(yi)
                ztrain.append(zi)
    if cv_valid > 0:
        return (xtrain, ytrain, ztrain), (xvalid, yvalid, zvalid), (xtest, ytest, ztest)
    return (xtrain, ytrain, ztrain), (xtest, ytest, ztest), (xtest, ytest, ztest)


class shar(dataset, real_dataset, classification_dataset):
    def __init__(self, version='adl', cv_semisup=0, split=0, folds=10, basedir='UniMiB-SHAR/', xyz_channels=True, oned_stacks=0, remove_grav=True,
                 **kwargs):
        dataset.__init__(self, **kwargs)
        self._name = 'SHAR_ALL'
        self._noutputs = 9 if version == 'adl' else 17
        self._labels = ['standing', 'getting up', 'walking', 'running', 'up stairs', 'jumping', 'down stairs', 'lying',
                        'sitting']
        self._args = kwargs
        self._cv_semisup = cv_semisup
        self._split = split
        self._folds = folds
        self._version = version
        self._basedir = basedir
        self.semisupervised = cv_semisup > 0
        self.rescale_images = False
        self.imagedata = False
        self.xyz_channels = xyz_channels
        self.remove_grav = remove_grav
        self.shar_standardize = False
        self.shar_instance_standardize = True
        self.oned_stacks = oned_stacks

    def fetch_data(self, download_dir=None):
        train, valid, testd = get_cv_split(
            load_shar(basedir=self._basedir, version=self._version, folds=self._folds, **self._args),
            cv_semisup=self._cv_semisup, split=self._split, **self._args)
        if self.remove_grav:
            train, valid, testd = remove_gravity_data(train), remove_gravity_data(valid), remove_gravity_data(testd)
        if self.shar_standardize:
            x = np.stack(train[0])
            mean = np.mean(x, axis=(0, 1), keepdims=True)[0]
            std = np.std(x, axis=(0, 1), keepdims=True)[0]
            train = ([np.tanh((t - mean) / (2 * std)) for t in train[0]],) + train[1:]
            valid = ([np.tanh((t - mean) / (2 * std)) for t in valid[0]],) + valid[1:]
            testd = ([np.tanh((t - mean) / (2 * std)) for t in testd[0]],) + testd[1:]
        if self.shar_instance_standardize:
            train = ([np.tanh((t - np.mean(t, axis=0, keepdims=True)) / (3 * np.std(t, axis=0, keepdims=True))) for t in train[0]],) + train[1:]
            valid = ([np.tanh((t - np.mean(t, axis=0, keepdims=True)) / (3 * np.std(t, axis=0, keepdims=True))) for t in valid[0]],) + valid[1:]
            testd = ([np.tanh((t - np.mean(t, axis=0, keepdims=True)) / (3 * np.std(t, axis=0, keepdims=True))) for t in testd[0]],) + testd[1:]
        if self.xyz_channels:
            shape = (-1, 1, 151, 3)
            train = np.stack(train[0]).reshape(shape)[:, :, 3:147], np.array(train[1])
            valid = np.stack(valid[0]).reshape(shape)[:, :, 3:147], np.array(valid[1])
            test = np.stack(testd[0]).reshape(shape)[:, :, 3:147], np.array(testd[1])

            if self.oned_stacks:
                train = (np.repeat(train[0], self.oned_stacks, axis=1), train[1])
                valid = (np.repeat(valid[0], self.oned_stacks, axis=1), valid[1])
                test = (np.repeat(test[0], self.oned_stacks, axis=1), test[1])
        else:
            shape = (-1, 151, 3, 1)
            train = np.stack(train[0]).reshape(shape)[:, 3:147], np.array(train[1])
            valid = np.stack(valid[0]).reshape(shape)[:, 3:147], np.array(valid[1])
            test = np.stack(testd[0]).reshape(shape)[:, 3:147], np.array(testd[1])

        self.data = dict(train=(train[0], train[1]),
                         valid=(valid[0], valid[1]), test=(test[0], test[1]))

class shar_std(shar):
    def __init__(self, **kwargs):
        shar.__init__(self, **kwargs)
        self.shar_standardize = True

class shar_istd(shar):
    def __init__(self, **kwargs):
        shar.__init__(self, **kwargs)
        self.shar_instance_standardize = True
