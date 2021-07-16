import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from ..visualizations import plot_cst_confusion, plot_latent_mixture_stats, plot_prior_distribution, plot_confusion, plot_compare, plot_latent_stats, plot_samples, plot_reconstructions, plot_encodings, plot_samples_by_class
from ..util.util import joinpath

class ramp(tf.keras.callbacks.Callback):
    def __init__(self, final_value, ramp_iters=0, warmup_iters=0, init_value=0., by_epoch=False):
        self.final_value = final_value
        self.ramp_iters = ramp_iters
        self.warmup_iters = warmup_iters
        self.init = init_value if ramp_iters > 0 else final_value
        self.by_epoch = by_epoch
        self.current_batch = 0

    def setup(self, model, scale=1.):
        self.init = scale * self.init
        self.final_value = scale * self.final_value

        model.callbacks.append(self)
        self.var = tf.Variable(self.init, trainable=False)
        return self.var

    def on_batch_end(self, epoch, logs=None, override=False):
        self.current_batch += 1
        epoch = self.current_batch
        if self.ramp_iters > 0 and epoch > self.warmup_iters and (not self.by_epoch or override):
            epoch = max(epoch - self.warmup_iters, 0)
            val = min(epoch *(self.final_value / self.ramp_iters) + self.init, self.final_value)
            tf.keras.backend.set_value(self.var, val)

    def on_epoch_end(self, epoch, logs=None):
        if self.by_epoch:
            self.on_batch_end(epoch, logs, True)


class VizCallback(tf.keras.callbacks.Callback):
    # Keras callback for annealing Beta at each epoch end

    def __init__(self, ca, data):
        self.ca = ca
        self.data = data

    def on_batch_end(self, epoch, logs=None):
        if np.random.random() < 0.005:
            try:
                x, y = self.data.valid().numpy()
                x, y, = x[:1000], y[:1000]
                recon = self.ca.predict([x, y])[0]
                x, recon = self.data.norm_inv(x), self.data.norm_inv(recon)
                plot_compare(x, recon, rescale=False)
            except:
                pass

class AllVizCallback(tf.keras.callbacks.Callback):
    # Keras callback for annealing Beta at each epoch end

    def __init__(self, m=None, data=None, save=None):
        self.m = m
        self.data = data
        self.save_path = save

    def save(self, name, epoch):
        if self.save_path is None:
            return None
        return joinpath(self.save_path,  'visualizations', 'epoch_%d' % epoch, file='%s.pdf' % name)

    def on_epoch_end(self, epoch, logs=None):
        results = dict(model=self.m, dataset=self.data, labels=[str(i) for i in range(10)])
        plot_samples(results, save=self.save('samples', epoch))
        try:
            #plot_samples_by_class(results)
            plot_latent_stats(results, save=self.save('stats', epoch))
        except:
            pass
        try:
            #plot_samples_by_class(results)
            plot_latent_mixture_stats(results, save=self.save('mix_stats', epoch))
        except:
            pass
        try:
            plot_confusion(results, save=self.save('confusion', epoch))
            plot_cst_confusion(results, save=self.save('consistency_confusion', epoch))
        except:
            pass
        plot_encodings(results, save=self.save('encodings', epoch))
        plot_prior_distribution(results, save=self.save('prior', epoch))
        plot_reconstructions(results, save=self.save('reconstructions', epoch))
        plot_reconstructions(results, consistency=True, save=self.save('consistency_reconstructions', epoch))
        plot_reconstructions(results, consistency=True, split='train', save=self.save('training_reconstructions', epoch))
        try:
            pass
            #plot_samples_by_class(results)
        except:
            pass


class ConsisCallback(tf.keras.callbacks.Callback):
    # Keras callback for annealing Beta at each epoch end

    def __init__(self, ca, data):
        self.ca = ca
        self.ld, self.ldy = data.labeled().numpy()
        self.uld, self.uldy = data.unlabeled().numpy()
        self.val, self.valy = data.valid().numpy()

        self.ldy = self.ldy.argmax(axis=-1)
        self.uldy = self.uldy.argmax(axis=-1)
        self.valy = self.valy.argmax(axis=-1)

        print(self.ld.shape)
        print(self.uld.shape)
        print(self.val.shape)

    def on_epoch_end(self, epoch, logs=None):
        try:

            # TODO: Consistency vs. Accuracy

            # LD Split
            loss, pred = self.ca.predict([self.ld])
            lossc = loss[pred.argmax(axis=-1) == self.ldy]
            lossic = loss[pred.argmax(axis=-1) != self.ldy]

            plt.figure()
            plt.hist(lossc, density=True, bins='fd')
            plt.title(f'Epoch: {epoch+1} Labeled C Mean:{round(float(lossc.mean()),3)}')
            plt.show()

            plt.figure()
            plt.hist(lossic, density=True, bins='fd')
            plt.title(f'Epoch: {epoch+1} Labeled IC Mean:{round(float(lossic.mean()),3)}')
            plt.show()

            # ULD Split
            loss, pred = self.ca.predict([self.uld])
            lossc = loss[pred.argmax(axis=-1) == self.uldy]
            lossic = loss[pred.argmax(axis=-1) != self.uldy]

            plt.figure()
            plt.hist(lossc, density=True, bins='fd')
            plt.title(f'Epoch: {epoch+1} Unlabeled C Mean:{round(float(lossc.mean()),3)}')
            plt.show()

            plt.figure()
            plt.hist(lossic, density=True, bins='fd')
            plt.title(f'Epoch: {epoch+1} Unlabeled IC Mean:{round(float(lossic.mean()),3)}')
            plt.show()

            # VAL Split
            loss, pred = self.ca.predict([self.val])
            lossc = loss[pred.argmax(axis=-1) == self.valy]
            lossic = loss[pred.argmax(axis=-1) != self.valy]

            plt.figure()
            plt.hist(lossc, density=True, bins='fd')
            plt.title(f'Epoch: {epoch+1} Valid C Mean:{round(float(lossc.mean()),3)}')
            plt.show()

            plt.figure()
            plt.hist(lossic, density=True, bins='fd')
            plt.title(f'Epoch: {epoch+1} Valid IC Mean:{round(float(lossic.mean()),3)}')
            plt.show()



        except:
            raise

