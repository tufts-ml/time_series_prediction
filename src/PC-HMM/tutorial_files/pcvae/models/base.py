import pickle
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.metrics import confusion_matrix as sk_confusion
import tensorflow as tf
import os
from pcvae.util import get_function_spec, print_options, scipy_min
from ..util.callbacks import VizCallback, AllVizCallback, ConsisCallback
from tqdm.keras import TqdmCallback
from ..util.util import joinpath

class BaseVAE(object):
    def __init__(self, **kwargs):
        self.built = False  # Model has been built
        self.labeled_only = False  # Model doesn't support unlabeled data
        self.metrics = {}
        self.standardize = None
        self.callbacks = []

    @classmethod
    def name(cls):
        return 'base'

    # Get a list of arguments that can be passed to __init__ (hyperparameters)
    # See hmm_model.py for example of format.
    @classmethod
    def get_init_args(cls):
        return get_function_spec(cls.__init__)

    @classmethod
    def get_fit_args(cls):
        return get_function_spec(cls.fit)

    @classmethod
    def get_all_args(cls):
        init_args = {}
        for c in reversed([cls] + list(cls.__mro__)):
            if issubclass(c, BaseVAE):
                init_args.update(dict(c.get_init_args()))
        init_args.update(cls.get_fit_args())
        return init_args

    @classmethod
    def help(cls):
        print('Model: %s' % cls.name())
        init_args = {}
        for c in reversed([cls] + list(cls.__mro__)):
            if issubclass(c, BaseVAE):
                init_args.update(c.get_init_args())
        print_options(init_args)

        print('Optimization with SGD')
        print_options(cls.get_fit_args())

    @classmethod
    def init_from_args(cls, **kwargs):
        return cls(**kwargs)

    def save(self, filepath):
        self.model.save_weights(joinpath(filepath, file='weights.h5'), save_format='h5')

    def load(self, data, load_from=None, load_checkpoint=None,):
        # A little bit of bookkeeping
        self.classes = data.classes()

        # Build the actual keras models
        if not self.built:
            self.build(data)

        if load_from:
            self.model.load_weights(os.path.join(load_from, 'weights.h5'), by_name=True, skip_mismatch=True)
        elif load_checkpoint:
            self.model.load_weights(os.path.join(load_from, 'checkpoints', 'checkpoint'), by_name=True,
                                    skip_mismatch=True)

    def build(self, data=None):
        # Build keras models for specific technique
        pass

    def callback(self, *args, **kwargs):
        pass

    def parameter(self, value, scale=1.):
        # Setup a parameter in a way that allows a schedule to be passed (e.g. or alpha, beta or lambda)
        try:
            value = value.setup(self, scale)
        except:
            pass
        return value

    def fit(self, data, lr=0.001,  epochs=10, lr_decay=0.001, decay_steps=400, decay_warmup=0,
            reduce_lr=True,  expon_decay=False, cosine_decay=False, profile=False, profile_dir='logs/profile/',
            use_multiprocessing=False, checkpoint_file=None, steps_per_epoch=None, save=None, load_from=None, load_checkpoint=None,
            labeled_only=False, use_fit_generator=False, show_recon=False, show_consistency=False, show_viz=False,
            verbose=0, show_model=False, initial_weights=None, **kwargs):

        # A little bit of bookkeeping
        self.classes = data.classes()

        # Build the actual keras models
        if not self.built:
            self.build(data)

        if load_from:
            self.model.load_weights(os.path.join(load_from, 'weights.h5'), by_name=True, skip_mismatch=True)
        elif load_checkpoint:
            self.model.load_weights(os.path.join(load_from, 'checkpoints', 'checkpoint'), by_name=True, skip_mismatch=True)

        # Handle different learning rate schedules
        cblr = self.callbacks
        cblr.append(TqdmCallback(verbose=verbose))
        if reduce_lr:
            cblr.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1,
                                          patience=10, min_lr=0.00001))
            cblr.append(EarlyStopping(patience=50))
        elif expon_decay:
            def dfunc(epoch):
                if epoch <= decay_warmup:
                    return lr
                else:
                    return lr * lr_decay ** ((epoch - decay_warmup) / decay_steps)
            cblr.append(LearningRateScheduler(dfunc))
        elif cosine_decay:
            def dfunc(epoch):
                return 0.25 * lr * (1 + np.cos(np.pi * epoch / float(epochs))) ** 2
            cblr.append(LearningRateScheduler(dfunc))
        cblr.append(TerminateOnNaN())

        # Handle checkpoints and visualizations at epoch end
        if not (checkpoint_file is None):
            cblr.append(ModelCheckpoint(checkpoint_file, save_best_only=True, save_weights_only=True))
        #if save:
        #    cblr.append(ModelCheckpoint(joinpath(save, 'checkpoints', file='checkpoint'), save_best_only=True, save_weights_only=True))
        if show_recon:
            cblr.append(VizCallback(self.constraint_autoencoder, data))
        if show_viz:
            cblr.append(AllVizCallback(m=self, data=data, save=save))
        if show_consistency:
            cblr.append(ConsisCallback(self.consistancy_loss, data))
        if profile:
            cblr.append(tf.keras.callbacks.TensorBoard(log_dir=profile_dir, histogram_freq=1))

        if show_model:
            self.model.summary()
        if initial_weights is not None:
            self.model.load_weights(initial_weights)

        # Details of history tracking
        steps_per_epoch = data.steps_per_epoch if steps_per_epoch is None else steps_per_epoch

        # Finally actually train the damn model
        vd = data.valid().numpy()
        vd = tf.data.Dataset.from_tensor_slices((vd, vd)).batch(np.sum(data.batch_size).astype(int))
        training_engine = self.model.fit if use_fit_generator else self.model.fit
        training_generator = data.optimize(labeled_only)
        self.history = training_engine(training_generator, validation_data=vd,
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs, use_multiprocessing=use_multiprocessing,
                                       callbacks=cblr, verbose=0)

        if save:
            self.save(save)

    def predict_data(self, data):
        try:
            return data.predict()
        except:
            return data

    def predict_proba(self, data, split=None):
        if not (split is None):
            data = data.get(split)
        return self.predictor.predict(self.predict_data(data))

    def predict(self, data, split=None):
        if not (split is None):
            data = data.get(split)
        # Override for other types of labels
        return self.predict_proba(data).argmax(axis=1)

    def accuracy(self, data, split=None):
        if not (split is None):
            data = data.get(split)
        Y = data.numpy_labels().argmax(axis=1)
        return np.mean(self.predict(data) == Y)

    def confusion_matrix(self, data, split=None):
        if not (split is None):
            data = data.get(split)
        Y = data.numpy_labels().argmax(axis=1)
        return sk_confusion(Y, self.predict(data))

    def reconstruct(self, data, split=None):
        if not (split is None):
            data = data.get(split)
        return self.autoencoder.predict(self.predict_data(data))

    def mse(self, data, split=None):
        if not (split is None):
            data = data.get(split)
        try:
            X = data.numpy()[0]
        except:
            X = data
        return np.mean((X - self.reconstruct(data)) ** 2)

    def llik(self, data, split=None):
        raise NotImplementedError()

    def encode(self, data, split=None, return_var=False, return_sample=False, return_mixture=False, secondary=False):
        if not (split is None):
            data = data.get(split)
        if return_mixture:
            return self.mixture_encoder.predict(self.predict_data(data))
        mean, var, sample = self.encoder.predict(self.predict_data(data))
        if secondary:
            mean, var, sample = self.secondary_encoder.predict(self.predict_data(data))
        if return_sample:
            return sample
        if return_var:
            return mean, var
        return mean

    def save_encoded_dataset(self, data, filename):
        output = dict()
        for split in ['train', 'test', 'valid', 'labeled', 'unlabeled']:
            d = data.get(split).predict()
            mean, var, sample = self.encoder.predict(d)
            labels = data.get(split).numpy_labels().argmax(axis=1)
            output[split] = dict(mean=mean, var=var, labels=labels)
        pickle.dump(output, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


    def decode(self, Z, data=None, split=None, sample=False):
        if not (split is None):
            data = data.get(split)
        if sample:
            return self.sample_decoder.predict(Z)
        return self.decoder.predict(Z)

    def inpaint(self, image, iterations=10):
        mask = np.isnan(image)
        paint = np.random.random(image.shape) * 2. - 1.
        for i in range(iterations):
            paint = np.where(mask, paint, image)
            paint = self.autoencoder.predict(paint)
        return np.where(mask, paint, image)

    def conditional_sample(self, Y, threshold=0.9):
        Y = np.squeeze(Y)
        Y = Y if Y.ndim == 1 else Y.argmax(axis=1)
        Yneeded = {c: count for (c, count) in zip(*np.unique(Y, return_counts=True))}
        Yfound = {c: [] for c in np.unique(Y)}
        Ytotal, Yfoundc = Y.shape[0], 0
        for i in range(1000):
            samples = self.sample_prior(100)
            preds = self.latent_predictor.predict(samples)
            samples, preds = samples[preds.max(axis=1) > threshold], preds[preds.max(axis=1) > threshold]
            for s, p in zip(samples, preds):
                c = np.argmax(p, axis=-1)
                if Yneeded[c] > 0:
                    Yneeded[c] -= 1
                    Yfound[c].append(s)
                    Yfoundc += 1
                    if Yfoundc == Ytotal:
                        break

            print('Collected: %d / %d samples' % (Yfoundc, Ytotal))
            if Yfoundc == Ytotal:
                break
        return self.decoder.predict(np.stack([Yfound[c].pop() for c in Y]))

    def sample(self, nsamples=1, Y=None, data=None, split=None, index=0, scale_dev=1., sample=False):
        if not (data is None):
            mean, var = self.encode(data, split, True)
            std = np.exp(0.5 * var) * scale_dev
            smpls = self.sample_prior(nsamples) * std[index].reshape((1, -1)) + mean[index].reshape(
                (1, -1))
            return self.decoder.predict(smpls) if not sample else self.sample_decoder.predict(smpls)
        return self.decoder.predict(
            self.sample_prior(nsamples)) if not sample else self.sample_decoder(
            self.sample_prior(nsamples))

    def losses(self, data, split=None):
        if not (split is None):
            data = data.get(split)
        return dict(zip(self.model.metrics_names, self.model.evaluate(data.evaluate())))
