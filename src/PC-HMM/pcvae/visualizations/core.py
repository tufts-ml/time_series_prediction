import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import colors
from matplotlib import cm
import pandas as pd
import math
from ..datasets import base as base_datasets
from ..util.util import isnotebook

def plot_confusion(results, split='test', save=None, show=True):
    labels = results['labels']
    model = results['model']
    with sns.axes_style("ticks"):
        C = model.confusion_matrix(results['dataset'], split)
        Y = results['dataset'].get(split).numpy_labels().argmax(axis=1)
        if C.shape[0] > 50:
            return
        labels = [str(d) for d in range(C.shape[0])] if labels is None else labels
        acc = np.diag(C).sum() / C.sum()
        f, (ax0, ax2) = plt.subplots(1, 2, figsize=(9, 8), gridspec_kw={'width_ratios': [8, 1]})
        sns.heatmap(C / C.sum(axis=1, keepdims=True), vmin=0, ax=ax0, xticklabels=labels,
                    yticklabels=labels, linewidths=2, vmax=1, cmap='Reds', cbar=False, annot=True,
                    square=True)
        name = model.name()
        name = name.upper() if len(name) < 5 else name.capitalize()
        ax0.set_title(
            '%s %s accuracy: %.2f%%' % (name, split, (100. * acc)),
            fontsize=24)
        ax0.tick_params(axis='both', which='major', labelsize=18, rotation=30)

        if labels:
            ax0.set_yticklabels(
                labels, rotation=30, ha='right', rotation_mode='anchor')
            ax0.set_xticklabels(
                labels, rotation=30, ha='right', rotation_mode='anchor')
        f.subplots_adjust(wspace=0.1, hspace=0)
        sns.heatmap(
            np.histogram(Y, bins=len(labels))[0].reshape((-1, 1)) / np.histogram(Y, bins=len(labels))[0].reshape(
                (-1, 1)).sum() * 100, ax=ax2, linewidths=2, cmap='Blues', cbar=False, annot=True)
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax2.set_title('% of labels', fontsize=24)
        ax2.margins(10, 10)
        if save:
            plt.savefig('/Users/personal/Desktop/loss_comp.pdf')
        if show and isnotebook():
            plt.show()

def plot_cst_confusion(results, split='test', save=None, show=True):
    labels = results['labels']
    model = results['model']
    with sns.axes_style("ticks"):
        C = model.consistency_confusion_matrix(results['dataset'], split)
        Y = results['dataset'].get(split).numpy_labels().argmax(axis=1)
        if C.shape[0] > 50:
            return
        labels = [str(d) for d in range(C.shape[0])] if labels is None else labels
        acc = np.diag(C).sum() / C.sum()
        f, (ax0, ax2) = plt.subplots(1, 2, figsize=(9, 8), gridspec_kw={'width_ratios': [8, 1]})
        sns.heatmap(C / C.sum(axis=1, keepdims=True), vmin=0, ax=ax0, xticklabels=labels,
                    yticklabels=labels, linewidths=2, vmax=1, cmap='Reds', cbar=False, annot=True,
                    square=True)
        name = model.name()
        name = name.upper() if len(name) < 5 else name.capitalize()
        ax0.set_title(
            '%s %s consistency accuracy: %.2f%%' % (name, split, (100. * acc)),
            fontsize=24)
        ax0.tick_params(axis='both', which='major', labelsize=18, rotation=30)

        if labels:
            ax0.set_yticklabels(
                labels, rotation=30, ha='right', rotation_mode='anchor')
            ax0.set_xticklabels(
                labels, rotation=30, ha='right', rotation_mode='anchor')
        f.subplots_adjust(wspace=0.1, hspace=0)
        sns.heatmap(
            np.histogram(Y, bins=len(labels))[0].reshape((-1, 1)) / np.histogram(Y, bins=len(labels))[0].reshape(
                (-1, 1)).sum() * 100, ax=ax2, linewidths=2, cmap='Blues', cbar=False, annot=True)
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax2.set_title('% of labels', fontsize=24)
        ax2.margins(10, 10)
        if save:
            plt.savefig('/Users/personal/Desktop/loss_comp.pdf')
        if show and isnotebook():
            plt.show()


def plot_latent_stats(results, split='test', show_contours=False, save=None, show=True):
    model = results['model']
    with sns.axes_style("ticks"):
        labels = results['dataset'].numpy_labels()
        nclasses = labels.shape[-1]
        test_labels = np.argmax(results['dataset'].test().numpy_labels(), axis=-1)
        l_labels = np.argmax(results['dataset'].labeled().numpy_labels(), axis=-1)
        ul_labels = np.argmax(results['dataset'].unlabeled().numpy_labels(), axis=-1)

        z_log_var, z_mean = model.encoder.predict(results['dataset'].test().numpy()[0])[1:]
        z_sample_l = model.encoder.predict(results['dataset'].labeled().numpy()[0])[2]
        z_sample_u = model.encoder.predict(results['dataset'].unlabeled().numpy()[0])[2]
        z_var = np.exp(z_log_var)
        ndims = min(z_mean.shape[-1], 20)
        f, ax = plt.subplots(ndims, 4, figsize=(12, 4 * ndims), sharex=False)
        for dim in range(ndims):
            mean, var, sample_l, sample_u = z_mean[:, dim], z_var[:, dim], z_sample_l[:, dim], z_sample_u[:, dim]
            ax[dim, 0].hist([mean[test_labels == label] for label in range(nclasses)], 25, stacked=True)
            ax[dim, 0].set_title('Mean (test), dim.: %d' % dim)

            ax[dim, 1].hist([var[test_labels == label] for label in range(nclasses)], 25, stacked=True)
            ax[dim, 1].set_title('Var. (test), dim.: %d' % dim)

            ax[dim, 2].hist([sample_l[l_labels == label] for label in range(nclasses)], 25, stacked=True)
            ax[dim, 2].set_title('Sample (labeled), dim.: %d' % dim)

            ax[dim, 3].hist([sample_u[ul_labels == label] for label in range(nclasses)], 25, stacked=True)
            ax[dim, 3].set_title('Sample (unlabeled), dim.: %d' % dim)

        if save:
            plt.savefig('/Users/personal/Desktop/loss_latent_stats.pdf')
        if show and isnotebook():
            plt.show()

def plot_latent_mixture_stats(results, split='test', show_contours=False, save=None, show=True):
    model = results['model']
    with sns.axes_style("ticks"):
        labels = results['dataset'].numpy_labels()
        nclasses = labels.shape[-1]
        labels = np.argmax(labels, axis=-1)
        z_mean = model.encode(results['dataset'], split=split, return_mixture=True)
        ndims = z_mean.shape[-1]
        rows = math.ceil(ndims / 3.)
        f, ax = plt.subplots(rows, 3, figsize=(9, 4 * (rows)), sharex=False)
        print('z_mean_range', z_mean.min(), z_mean.max())
        for dim in range(ndims):
            r, c = dim // 3, dim % 3
            mean = z_mean[:, dim]
            ax[r, c].hist([mean[labels == label] for label in range(nclasses)], 25,  range=(0., 1.), stacked=True)
            ax[r, c].set_title('Mix. Probability, dim.: %d' % dim)

        if save:
            plt.savefig('/Users/personal/Desktop/loss_latent_mixture_stats.pdf')
        if show and isnotebook():
            plt.show()

def plot_prior_distribution(results, save=None, show=True):
    model = results['model']
    with sns.axes_style("ticks"):
        z_mean = model.sample_prior(2000)
        latent_dim = z_mean.shape[-1]
        if latent_dim > 2:
            z_mean = PCA(n_components=2).fit_transform(z_mean)
        g = sns.jointplot(x=z_mean[:, 0], y=z_mean[:, 1], kind='kde')
        if save:
            plt.savefig(save)
        if show and isnotebook():
            plt.show()


def plot_encodings(results, split='test', show_contours=False, save=None, show=True, secondary=False):
    labels = results['labels']
    model = results['model']
    with sns.axes_style("ticks"):
        if split == 'prior':
            acc = 0.
            z_mean = model.sample_prior(2000)
            Y = np.zeros(2000)
        else:
            acc = model.accuracy(results['dataset'], split=split)
            z_mean = model.encode(results['dataset'], split=split, return_sample=True, secondary=secondary)

            X, Y = results['dataset'].get(split).numpy()
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            X, Y, z_mean = X[inds][:10000], Y[inds][:10000].argmax(axis=1), z_mean[inds][:10000]

        latent_dim = z_mean.shape[-1]
        if latent_dim > 2:
            z_mean = PCA(n_components=2).fit_transform(z_mean)

        try:
            xx, yy = np.meshgrid(np.linspace(z_mean[:, 0].min(), z_mean[:, 0].max(), 50),
                                 np.linspace(z_mean[:, 1].min(), z_mean[:, 1].max(), 50))
            dec_input = np.pad(np.stack([xx.reshape((-1,)), yy.reshape((-1,))]).T, [(0, 0), (0, latent_dim - 2)],
                               'constant')

            pb = model.latent_predictor.predict(dec_input).argmax(axis=1)
            pb = pb.reshape(xx.shape)
        except:
            print("Could not show contours")

        _, ax = plt.subplots(figsize=(12, 10))
        labels = [str(d) for d in range(model.label_shape)] if labels is None else labels
        for i, l in enumerate(labels):
            inds = Y.flatten() == i
            ax.scatter(z_mean[:, 0][inds], z_mean[:, 1][inds],
                       c=np.array(plt.cm.get_cmap('tab10', 10)(i)).reshape((1, -1)), label=str(l))
        if show_contours:
            ax.contourf(xx, yy, pb * 1.05, alpha=0.4, levels=np.arange(model.classes + 1),
                        colors=[plt.get_cmap('tab10')(ii) for ii in range(10)])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(axis='both', which='major', width=2, length=10, labelsize=18)
        sns.despine(offset=1, trim=True, ax=ax)
        leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
        leg.get_frame().set_linewidth(2)
        ax.set_xlabel("Latent dim. 1/2", fontsize=20)
        ax.set_ylabel("Latent dim. 2/2", fontsize=20)

        name = model.name()
        name = name.upper() if len(name) < 5 else name.capitalize()
        nlabels = str(results['dataset'].nlabels) if results['dataset'].nlabels >= 0 else 'all'
        ax.set_title("%s, %s labels, %s accuracy: %.2f%%" % (name, nlabels, split, (100. * acc)),
                     fontsize=24)
        if save:
            plt.savefig(save)
        if show and isnotebook():
            plt.show()



def plot_1D_reconstructions(results, split='test', rows=30, misclassified=False, show=True, save=None, **kwargs):
    model = results['model']
    classes = results['dataset'].classes()
    input_shape = model.input_shape
    recon = model.reconstruct(results['dataset'], split=split, **kwargs)
    X, Y = results['dataset'].get(split).numpy()
    Y = Y.argmax(axis=1)
    if misclassified:
        Ypred = model.predict(results['dataset'], split=split)
        X, Y, recon = X[Y.astype(int) != Ypred.astype(int)], Y[Y.astype(int) != Ypred.astype(int)], recon[Y.astype(int) != Ypred.astype(int)]
    outputs = []
    org = []
    for c in range(classes):
        Xc, Yc, reconc = X[Y.astype(int) == c][:rows], Y[Y.astype(int) == c][:rows], recon[Y.astype(int) == c][:rows]
        Xc = results['dataset'].norm_inv(Xc)
        outputs.append(results['dataset'].norm_inv(reconc)[0])
        org.append(Xc[0])

    _, all_ax = plt.subplots(rows, 1, figsize=(18, 6 * classes))
    for c, (r, x, ax) in enumerate(zip(outputs, org, all_ax)):
        r = np.squeeze(r)
        for dim in range(r.shape[-1]):
            ax.plot(np.squeeze(r)[:, dim].flatten(), c=cm.tab10(dim))
            ax.plot(np.squeeze(x)[:, dim].flatten(), ls=':', c=cm.tab10(dim))
        ax.axis('off')
        ax.set_title('Reconstruction for class %d' % c, fontsize=24)

    ax.set_title('Reconstructions by class', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()

def plot_reconstructions(results, split='test', rows=30, misclassified=False, show=True, consistency=False, save=None, **kwargs):
    model = results['model']
    classes = results['dataset'].classes()
    input_shape = model.input_shape
    if not results['dataset'].imagedata:
        return plot_1D_reconstructions(results, split, rows // 3, misclassified, show)
    cols = rows // 2
    img = np.zeros((rows * input_shape[0], cols * 3 * input_shape[1], input_shape[2]))
    #recon = model.reconstruct(results['dataset'], split=split, **kwargs)
    X, Y = results['dataset'].get(split).numpy()
    Y = Y.argmax(axis=1)
    if misclassified:
        Ypred = model.predict(results['dataset'], split=split)
        X, Y = X[Y.astype(int) != Ypred.astype(int)], Y[Y.astype(int) != Ypred.astype(int)]
    for c in range(min(classes, cols)):
        if isinstance(results['dataset'], base_datasets.classification_dataset):
            if misclassified:
                Xc, Yc = X[:rows], Y[:rows]
                X, Y = X[rows:], Y[rows:]
            else:
                Xc, Yc = X[Y.astype(int) == c][:rows], Y[Y.astype(int) == c][:rows]
        else:
            Xc, Yc = X[(c*rows):((c+1)*rows)], Y[(c*rows):((c+1)*rows)]
        YcOneHot = np.zeros([Yc.size, classes])
        YcOneHot[np.arange(Yc.size), Yc] = 1.
        reconc = model.sample_autoencoder.predict(Xc) if not consistency else model.const_autoencoder.predict([Xc, YcOneHot])
        Xc = np.concatenate(results['dataset'].norm_inv(Xc), axis=0)
        reconc = np.concatenate(results['dataset'].norm_inv(reconc), axis=0)
        column = np.concatenate([Xc, reconc, np.zeros_like(Xc)], axis=1)
        img[:, 3 * c * input_shape[1]:3 * (c + 1) * input_shape[1], :] = column

    _, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(np.squeeze(img), cmap='Greys', interpolation='nearest')
    ax.axis('off')
    ax.set_title('Reconstructions', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()
        
def plot_aligned_reconstructions(results, split='test', rows=30, misclassified=False, show=True, consistency=False, save=None, **kwargs):
    model = results['model']
    classes = results['dataset'].classes()
    input_shape = model.input_shape
    if not results['dataset'].imagedata:
        return plot_1D_reconstructions(results, split, rows // 3, misclassified, show)
    img = np.zeros((rows * input_shape[0], classes * 4 * input_shape[1], input_shape[2]))
    #recon = model.reconstruct(results['dataset'], split=split, **kwargs)
    X, Y = results['dataset'].get(split).numpy()
    Y = Y.argmax(axis=1)
    if misclassified:
        Ypred = model.predict(results['dataset'], split=split)
        X, Y = X[Y.astype(int) != Ypred.astype(int)], Y[Y.astype(int) != Ypred.astype(int)]
    for c in range(classes):
        Xc, Yc = X[Y.astype(int) == c][:rows], Y[Y.astype(int) == c][:rows]
        YcOneHot = np.zeros([Yc.size, classes])
        YcOneHot[np.arange(Yc.size), Yc] = 1.
        reconcog = model.autoencoder.predict(Xc)
        reconc = model.encoder.predict(Xc)[0]
        reconc[:, :6] = 0
        reconc = model.decoder.predict(reconc)
        Xc = np.concatenate(results['dataset'].norm_inv(Xc), axis=0)
        reconc = np.concatenate(results['dataset'].norm_inv(reconc), axis=0)
        reconcog = np.concatenate(results['dataset'].norm_inv(reconcog), axis=0)
        column = np.concatenate([Xc, reconcog, reconc, np.zeros_like(Xc)], axis=1)
        img[:, 4 * c * input_shape[1]:4 * (c + 1) * input_shape[1], :] = column

    _, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(np.squeeze(img), cmap='Greys', interpolation='nearest')
    ax.axis('off')
    ax.set_title('Reconstructions', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()

def plot_reconstructions_comparison(results, split='test', rows=30, misclassified=False, show=True, consistency=False, save=None, **kwargs):
    all_results = results
    results = results[0]
    model = results['model']
    classes = results['dataset'].classes()
    input_shape = model.input_shape
    if not results['dataset'].imagedata:
        raise NotImplementedError()
    nmodels = len(all_results)
    img = np.zeros((rows * input_shape[0], classes * (2 + nmodels) * input_shape[1], input_shape[2]))
    #recon = model.reconstruct(results['dataset'], split=split, **kwargs)
    X, Y = results['dataset'].get(split).numpy()
    Y = Y.argmax(axis=1)
    if misclassified:
        Ypred = model.predict(results['dataset'], split=split)
        X, Y = X[Y.astype(int) != Ypred.astype(int)], Y[Y.astype(int) != Ypred.astype(int)]

    for c in range(classes):
        Xc, Yc = X[Y.astype(int) == c][:rows], Y[Y.astype(int) == c][:rows]
        YcOneHot = np.zeros([Yc.size, classes])
        YcOneHot[np.arange(Yc.size), Yc] = 1.
        reconc = []
        for r in all_results:
            m = r['model']
            rc = m.autoencoder.predict(Xc) if not consistency else m.const_autoencoder.predict([Xc, YcOneHot])
            rc = np.concatenate(results['dataset'].norm_inv(rc), axis=0)
            reconc.append(rc)

        Xc = np.concatenate(results['dataset'].norm_inv(Xc), axis=0)
        column = np.concatenate([Xc] + reconc + [np.zeros_like(Xc)], axis=1)
        img[:, (2 + nmodels) * c * input_shape[1]:(2 + nmodels) * (c + 1) * input_shape[1], :] = column

    _, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(np.squeeze(img), cmap='Greys', interpolation='nearest')
    ax.axis('off')
    ax.set_title('Reconstruction comparison', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()

def plot_images(images, rows=10, misclassified=False, show=True, save=None, rescale=True, **kwargs):
    input_shape = images.shape[1:]
    cols = rows // 2
    img = np.zeros((rows * input_shape[0], cols * 2 * input_shape[1], input_shape[2]))

    for c in range(cols):
        imgs = np.concatenate(images[c*rows:(c+1)*rows], axis=0)
        column = np.concatenate([imgs, np.zeros_like(imgs)], axis=1)
        img[:, 2 * c * input_shape[1]:2 * (c + 1) * input_shape[1], :] = (column + 1) / 2 if rescale else column

    _, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(np.squeeze(img))
    ax.axis('off')
    ax.set_title('Reconstructions by class', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()


def plot_compare(images, images2, rows=10, misclassified=False, show=True, rescale=True, save=None, **kwargs):
    input_shape = images.shape[1:]
    cols = rows // 3
    img = np.zeros((rows * input_shape[0], cols * 3 * input_shape[1], input_shape[2]))

    for c in range(cols):
        imgs = np.concatenate(images[c*rows:(c+1)*rows], axis=0)
        imgs2 = np.concatenate(images2[c * rows:(c + 1) * rows], axis=0)
        column = np.concatenate([imgs, imgs2, np.zeros_like(imgs)], axis=1)
        img[:, 3 * c * input_shape[1]:3 * (c + 1) * input_shape[1], :] = (column + 1) / 2 if rescale else column

    _, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(np.squeeze(img))
    ax.axis('off')
    ax.set_title('Reconstructions by class', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()

def plot_latent_map(results, posterior_index=-1, split='test', rows=30, show=True, scale_dev=1, save=None, sample=False):
    model = results['model']
    classes = results['dataset'].classes()
    input_shape = model.input_shape
    img = np.zeros((rows * input_shape[0], classes * 3 * input_shape[1], input_shape[2]))
    cols = rows // 2
    for c in range(cols):
        x, y = np.meshgrid(np.linspace(-3, 3, rows), np.linspace(-3, 3, rows))
        coords = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=-1)
        recon = model.decode(coords)
        reconc = results['dataset'].norm_inv(np.concatenate(recon, axis=0))

        column = np.concatenate([reconc, np.zeros_like(reconc)], axis=1)
        img[:, 2 * c * input_shape[1]:2 * (c + 1) * input_shape[1], :] = column

    _, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(np.squeeze(img))
    ax.axis('off')
    ax.set_title('Reconstructions by class', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()

def plot_1D_samples(results, posterior_index=-1, split='test', rows=10, show=True, scale_dev=1, sample=False, save=None):
    model = results['model']
    input_shape = model.input_shape
    print(input_shape)
    X, Y = results['dataset'].get(split).numpy()
    if posterior_index >= 0:
        zmean, zlvar = model.encode(results['dataset'], split=split, return_var=True)
        zdev = np.exp(0.5 * zlvar)
    if posterior_index < 0:
        recon = model.sample(nsamples=rows)
        reconc = results['dataset'].norm_inv(recon)
    else:
        zmeanc, zdevc = zmean[posterior_index], zdev[posterior_index]
        Z = np.random.randn(rows - 2, zmeanc.shape[-1]) * np.expand_dims(zdevc, 0) * scale_dev + np.expand_dims(
            zmeanc, 0)
        Z = np.concatenate([np.expand_dims(zmeanc, 0), Z], axis=0)
        recon = model.decode(Z)
        reconc = results['dataset'].norm_inv(
            np.concatenate([X[posterior_index]] + [ri for ri in recon], axis=0))

    _, all_ax = plt.subplots(rows, 1, figsize=(18, 6 * rows))
    print(reconc.shape)
    for r, ax in zip(reconc, all_ax):
        r = np.squeeze(r)
        for dim in range(r.shape[-1]):
            ax.plot(r[:, dim].flatten(), c=cm.tab10(dim))
        ax.axis('off')
        ax.set_title('Samples', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()

def plot_samples(results, posterior_index=-1, split='test', rows=30, show=True, scale_dev=1, save=None, sample=False):
    model = results['model']
    classes = results['dataset'].classes()
    input_shape = model.input_shape
    if not results['dataset'].imagedata:
        return plot_1D_samples(results, posterior_index, split, rows // 3, show, scale_dev)
    cols = rows // 2
    img = np.zeros((rows * input_shape[0], cols * 3 * input_shape[1], input_shape[2]))
    X, Y = results['dataset'].get(split).numpy()
    if posterior_index >= 0:
        zmean, zlvar = model.encode(results['dataset'], split=split, return_var=True)
        zdev = np.exp(0.5 * zlvar)
    for c in range(cols):
        if posterior_index < 0:
            recon = model.sample(nsamples=rows)
            reconc = np.concatenate(results['dataset'].norm_inv(recon), axis=0)
        else:
            zmeanc, zdevc = zmean[posterior_index + c], zdev[posterior_index + c]
            Z = np.random.randn(rows - 2, zmeanc.shape[-1]) * np.expand_dims(zdevc, 0) * scale_dev + np.expand_dims(
                zmeanc, 0)
            Z = np.concatenate([np.expand_dims(zmeanc, 0), Z], axis=0)
            recon = model.decode(Z)
            reconc = results['dataset'].norm_inv(np.concatenate([X[posterior_index + c]] + [ri for ri in recon], axis=0))

        column = np.concatenate([reconc, np.zeros_like(reconc)], axis=1)
        img[:, 2 * c * input_shape[1]:2 * (c + 1) * input_shape[1], :] = column

    _, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(np.squeeze(img), 'Greys', interpolation='nearest')
    ax.axis('off')
    ax.set_title('Samples', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()

def plot_1D_samples_by_class(results,  rows=30, show=True, threshold=0.9, save=None):
    model = results['model']
    classes = results['dataset'].classes()
    input_shape = model.input_shape
    Y = np.repeat(np.arange(classes), rows)
    recon = model.conditional_sample(Y, threshold=threshold)
    Y = Y.argmax(axis=1)
    outputs = []
    org = []
    for c in range(classes):
        Yc, reconc = Y[Y.astype(int) == c][:rows], recon[Y.astype(int) == c][:rows]
        Xc = results['dataset'].norm_inv(reconc)[1]
        outputs.append(results['dataset'].norm_inv(reconc)[0])
        org.append(Xc[0])

    _, all_ax = plt.subplots(rows, 1, figsize=(18, 6 * classes))
    for c, (r, x, ax) in enumerate(zip(outputs, org, all_ax)):
        for dim in range(max(input_shape[-2], input_shape[-1])):
            ax.plot(r[:, dim].flatten(), c=cm.tab10(dim))
            ax.plot(np.squeeze(x)[:, dim].flatten(), ls=':', c=cm.tab10(dim))
        ax.axis('off')
        ax.set_title('Samples for class %d' % c, fontsize=24)

    ax.set_title('Samples by class', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()


def plot_samples_by_class(results,  rows=30, show=True, threshold=0.9, save=None):
    if not results['dataset'].imagedata:
        return plot_1D_samples_by_class(results, rows // 3, show, threshold)
    model = results['model']
    classes = results['dataset'].classes()
    input_shape = model.input_shape
    img = np.zeros((rows * input_shape[0], classes * 2 * input_shape[1], input_shape[2]))
    cols = classes
    Y = np.repeat(np.arange(classes), rows)
    recon = model.conditional_sample(Y, threshold=threshold)
    acc = np.mean(model.predictor.predict(recon).argmax(axis=-1) == Y)
    for c in range(cols):
        reconc = recon[Y == c]
        reconc = np.concatenate(reconc, axis=0)
        column = results['dataset'].norm_inv(np.concatenate([reconc, np.zeros_like(reconc)], axis=1))
        img[:, 2 * c * input_shape[1]:2 * (c + 1) * input_shape[1], :] = column

    _, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(np.squeeze(img))
    ax.axis('off')
    ax.set_title('Samples by class, reclassification accuracy: %f' % acc, fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()

def mask(img, masktype):
    img = img.copy()
    imshape = img[0].shape
    if masktype == 'bottom':
        img[:, (imshape[0]//2):] = np.nan
    elif masktype == 'top':
        img[:, :(imshape[0] // 2)] = np.nan
    elif masktype == 'right':
        img[:, :, (imshape[0]//2):] = np.nan
    elif masktype == 'left':
        img[:, :, :(imshape[0] // 2)] = np.nan
    return img

def plot_inpaintings_by_class(results, split='test', masktype='bottom', method='optimize', rows=30, extra_reps=0, show=True, save=None):
    model = results['model']
    classes = results['dataset'].classes()
    input_shape = model.input_shape
    img = np.zeros((rows * input_shape[0], classes * (4 + extra_reps) * input_shape[1], input_shape[2]))
    X, Y = results['dataset'].get(split).numpy()
    Y = Y.argmax(axis=1)
    for c in range(classes):
        Xc, Yc = X[Y.astype(int) == c][:rows], Y[Y.astype(int) == c][:rows]
        masked = mask(Xc, masktype)
        reconc = np.concatenate([model.inpaint(m.reshape((1,) + m.shape)) for m in masked], axis=0)
        column = np.concatenate([results['dataset'].norm_inv(Xc), results['dataset'].norm_inv(masked), results['dataset'].norm_inv(reconc)], axis=2)
        for i in range(extra_reps):
            reconc = np.concatenate([model.inpaint(m.reshape((1,) + m.shape)) for m in masked], axis=0)
            column = np.concatenate([column, results['dataset'].norm_inv(reconc)], axis=2)
        column = np.concatenate([column, np.zeros_like(Xc)], axis=2)
        column = np.concatenate(column, axis=0)
        column[np.isnan(column)] = 0
        img[:, (4 + extra_reps) * c * input_shape[1]:(4 + extra_reps) * (c + 1) * input_shape[1], :] = column

    _, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(np.squeeze(img), cmap='Greys', interpolation='nearest')
    ax.axis('off')
    ax.set_title('Reconstructions by class', fontsize=24)
    if save:
        plt.savefig(save)
    if show and isnotebook():
        plt.show()


def plot_history(model, plot_loss=True, plot_val=True, ylim=None, save=None, show=True, metric='loss'):
    model = model['model'] if isinstance(model, dict) else model
    if metric in ['loss', 'acc', 'mse']:
        mname = dict(loss='loss', acc='accuracy', mse='mse')[metric]
        metric = dict(loss='loss', acc='prediction_out_accuracy', mse='reconstruction_out_mse')[metric]
    else:
        mname = metric
    with sns.axes_style("ticks"):
        _, ax = plt.subplots(figsize=(10, 8))
        # Plot the loss as a function of epochs/steps
        if plot_loss:
            ax.plot(model.history.history[metric], label='Model ' + mname, lw=3)
        if plot_val:
            ax.plot(model.history.history['val_' + metric], label='Validation ' + mname, lw=3, ls='--')
        if ylim:
            ax.set_ylim([0, 210])
        ax.set_title('Training ' + mname, fontsize=24)
        ax.set_ylabel(mname.capitalize(), fontsize=20)
        ax.set_xlabel('Epoch', fontsize=20)
        leg = ax.legend(prop={'size': 16})
        leg.get_frame().set_linewidth(1.5)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(axis='both', which='major', width=2, length=10, labelsize=18)
        sns.despine(trim=True, ax=ax)
        if save:
            plt.savefig(save)
        if show and isnotebook():
            plt.show()

def plot_topics(result, words=None):
    import math
    if words is None:
        words = result['dataset'].words
    model = result['model']
    a = result['model']._predictor.weights[0].numpy()
    weights = a[:, 0] - a[:, 1]

    topic = model.topicprobs(0).numpy()
    wordorder = np.argsort(np.dot(weights, topic))

    ntopics = model.topics
    cols, rows = 5, math.ceil(ntopics / 5)
    f, ax = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    plotorder = np.argsort(np.abs(weights))[::-1]
    for row in range(rows):
        for col in range(cols):
            axi = ax[row, col]
            axi.axis('off')
            try:
                i = plotorder[5 * row + col]
                topic = model.topicprobs(0).numpy()[i].flatten()
                topic = topic / topic.sum()
                order = np.array(words)[np.argsort(topic)][::-1][:5]
                shape = math.ceil(np.sqrt(topic.shape[0]))
                topic = np.pad(topic[wordorder], (0, shape * shape - topic.shape[0])).reshape((shape, shape))
                axi.imshow(topic, cmap='cividis', interpolation='none')
                axi.set_title("Weight: " + str(weights[i]) + ', top words:\n' + str(order))
            except:
                pass

    f.suptitle('Topics by regression weight')
    plt.show()

def plot_binary_roc(lda, split='test'):
    from sklearn import metrics
    fpr, tpr, _ = metrics.roc_curve(lda['dataset'].get(split).numpy_labels()[:, 1],
                                    lda['model'].predict_proba(lda['dataset'].get(split))[:, 1])
    roc_auc = metrics.roc_auc_score(lda['dataset'].get(split).numpy_labels()[:, 1],
                                    lda['model'].predict_proba(lda['dataset'].get(split))[:, 1])

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def fix_name(name, renamer=None):
    if renamer is None:
        renamer = dict(LAMBDA='$\\lambda$', lr='Learning rate', num='Number', acc='accuracy', mse='MSE',
                       xi='$\\xi$', epsilon='$\\epsilon$', BETA='$\\beta$', recon='Reconstruction',
                       reg='Regularization',
                       pc='PC', Pc='PC', m2='M2', adgm='ADGM', sdgm='SDGM', vat='VAT', dnn='Deep-NN')
    for k, v in renamer.items():
        name = name.replace(k, v)
    name = name.replace('_', ' ').replace(' = ', ': ')
    if name and not name[0].isupper():
        name = name.capitalize()
    return name


def map_2_fg(
        fg, map_fn, x, y, shade, plot_args, logshade=False, darkshade=False, vmin=None, vmax=None,
):
    if shade is None:
        fg = fg.map(map_fn, x, y, **plot_args).add_legend(prop={'size': 14})
        for txt, lh in zip(fg._legend.texts, fg._legend.legendHandles):
            txt.set_text(fix_name(txt.get_text()))
    else:
        label_2_color = {}

        def plot_fn(xarr, yarr, shadearr, color, label, **kwargs):
            cmap = (
                sns.dark_palette(color, as_cmap=True)
                if darkshade
                else sns.light_palette(color, as_cmap=True)
            )
            shadearr = np.log(np.abs(shadearr)) if logshade else shadearr
            label_2_color[label] = color
            map_fn(xarr, yarr, c=shadearr, cmap=cmap, vmin=vmin, vmax=vmax, label=label, **kwargs)

        fg = fg.map(plot_fn, x, y, shade, **plot_args).add_legend(prop={'size': 14})
        for txt, lh in zip(fg._legend.texts, fg._legend.legendHandles):
            lh.set_color(label_2_color[txt.get_text()])
            txt.set_text(fix_name(txt.get_text()))
    fg._legend.set_title(fix_name(fg._legend.get_title().get_text()), prop={'size': 14})
    return fg


def plot_results_grid(
        results,
        x=None,
        y=None,
        row=None,
        col=None,
        hue=None,
        shade=None,
        logshade=False,
        darkshade=False,
        logx=False,
        logy=False,
        selectby='valid_acc',
        group=None,
        height=7,
        fg_args={},
        map_fn=plt.scatter,
        plot_args={},
        save=False,
        show=True,
        renamer=None,
        title='Summary of Results'
):
    if not (isinstance(results, list) or isinstance(results, tuple)):
        results = [results]
    tables = []
    for r in results:
        try:
            tables.append(r['results']['table'])
        except:
            tables.append(r)
    results = pd.concat(tables, sort=True)

    spec = [s for s in [x, row, col, hue, shade] if not (s is None)]
    if group == 'max' or group == 'min':
        results = results.sort_values(by=selectby, ascending=group == 'min')
        results = results.drop_duplicates(spec)
    elif group == 'mean':
        results = results.groupby(spec).mean()
    results = results.sort_values(by=x)

    row = row if row in results or type(row) is list else None
    col = col if col in results or type(col) is list else None
    hue = hue if hue in results else None

    fg = sns.FacetGrid(
        results, row=row, col=col, hue=hue, height=height, legend_out=True, gridspec_kws=dict(hspace=0.25), **fg_args
    )
    vmin, vmax = None, None
    if not (shade is None):
        if logshade:
            vmin, vmax = min(np.log(np.abs(results[shade]))), max(np.log(np.abs(results[shade])))
        else:
            vmin, vmax = min(results[shade]), max(results[shade])

    fg = map_2_fg(fg, map_fn, x, y, shade, plot_args, logshade, darkshade, vmin, vmax)
    for ax in np.reshape(fg.axes, (-1,)):
        ax.tick_params(axis='both', which='major', width=2, length=10, labelsize=14)

        ax.xaxis.label.set_text(fix_name(ax.xaxis.label.get_text(), renamer))
        ax.yaxis.label.set_text(fix_name(ax.yaxis.label.get_text(), renamer))
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

        ax.set_title(fix_name(ax.get_title(), renamer))
        ax.title.set_size(18)

    fg.fig.subplots_adjust(top=0.85)
    fg.fig.suptitle(title, fontsize=24)

    if not (shade is None):
        cmap = (
            sns.dark_palette('darkgrey', as_cmap=True)
            if darkshade
            else sns.light_palette('darkgrey', as_cmap=True)
        )
        sm = cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        cbar = fg.fig.colorbar(sm, ax=fg.fig.axes, aspect=40, orientation="horizontal", shrink=0.5)
        cbar.ax.xaxis.label.set_text(fix_name(shade, renamer))
        cbar.ax.xaxis.label.set_size(12)
        cbar.ax.xaxis.set_label_position("bottom")
        cbar.ax.xaxis.label.set_horizontalalignment('center')
        cbar.ax.tick_params(axis='both', which='major', width=2, length=6, labelsize=12)
        sns.despine(ax=cbar.ax, trim=True)
    sns.despine(fg.fig, trim=True)
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    return fg
