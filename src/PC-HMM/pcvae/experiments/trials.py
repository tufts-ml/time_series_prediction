import pandas as pd
import os, uuid
from pcvae.visualizations import plot_reconstructions, plot_encodings, plot_confusion, plot_history
from ..util.util import joinpath
import os, pickle

def summary_dict(kwargs):
    summary = {}
    for entry, value in kwargs.items():
        value = str(value)
        try:
            summary[entry] = [int(value)]
            continue
        except:
            pass
        try:
            summary[entry] = [float(value)]
            continue
        except:
            pass
        try:
            summary[entry] = [str(value)]
            continue
        except:
            pass
    return summary


def make_visualizations(output, split='test'):
    try:
        plot_confusion(output, split=split)
    except:
        pass

    try:
        plot_encodings(output, split=split)
    except:
        pass

    try:
        plot_reconstructions(output, split=split)
    except:
        pass

    try:
        plot_history(output)
    except:
        pass

    try:
        plot_history(output, metric='acc')
    except:
        pass

    try:
        plot_history(output, metric='mse')
    except:
        pass


def get_metrics(PCmodel, data, split, summary):
    e = PCmodel.model.evaluate(data.get(split).evaluate(), verbose=0)
    e = {out: e[i] for i, out in enumerate(PCmodel.model.metrics_names)}
    try:
        summary[(split + '_acc')] = e['accuracy']
        acc = e['accuracy']
    except:
        if 'predictor_accuracy' in e:
            summary[(split + '_acc')] = e['predictor_accuracy']
            acc = e['predictor_accuracy']
    else:
        summary[(split + '_acc')] = 0.
        acc = 0.
    if 'mean_squared_error' in e:
        summary[(split + '_mse')] = e['mean_squared_error']
    if 'loss' in e:
        summary[(split + '_loss')] = e['loss']
    return acc


def check_args(dataset, model, kwargs):
    unused = {k: v for k, v in kwargs.items() if
              k not in
              list(dataset.get_all_args().keys()) + list(
                  model.get_all_args().keys())}
    if len(unused) > 0:
        raise TypeError("Arguments: " + str(unused) + ' do not match any known options.')


def run_trial(dataset, model, save=None, results=None, constructor=None, download_dir=None,
              return_data=True, validate_args=False, replications=1, load_from=None, **kwargs):
    dataset = dataset(download_dir=download_dir, **kwargs).load_data(download_dir=download_dir)
    summary = summary_dict(kwargs)
    summary.update(dict(model=[model.name()]))
    if validate_args:
        check_args(dataset, model, kwargs)

    results = {} if results is None else results
    modelargs = {}
    if load_from:
        with open(os.path.join(load_from, 'args.pkl'), 'rb') as areader:
            modelargs = pickle.load(areader)

    modelargs.update(kwargs)
    if save:
        with open(joinpath(save, file='args.pkl'), 'wb') as awriter:
            pickle.dump(modelargs, awriter)

    trial_id = str(uuid.uuid4())[:6]
    for r in range(replications):
        PCmodel = model(**modelargs)
        PCmodel.fit(dataset, save=save, load_from=load_from, **kwargs)

        get_metrics(PCmodel, dataset, 'train', summary)
        get_metrics(PCmodel, dataset, 'test', summary)
        valid_acc = get_metrics(PCmodel, dataset, 'valid', summary)
        summary.update(dict(trial_id=trial_id, replication=r))

        if 'table' not in results:
            results['table'] = pd.DataFrame(summary)
        else:
            results['table'] = pd.concat([results['table'], pd.DataFrame(summary)])

    if save:
        results['table'].to_csv(joinpath(save, file='results.csv'))

    dataset = dataset if return_data else dataset.clean()
    return dict(model=PCmodel, dataset=dataset, results=results,
                labels=dataset.labels(), args=kwargs, objective=valid_acc)


def load_model(dataset, model, download_dir=None, return_data=True, validate_args=False, load_from=None, **kwargs):
    dataset = dataset(download_dir=download_dir, **kwargs).load_data(download_dir=download_dir)
    summary = summary_dict(kwargs)
    summary.update(dict(model=[model.name()]))
    if validate_args:
        check_args(dataset, model, kwargs)

    results = {} if results is None else results
    modelargs = {}
    if load_from:
        with open(os.path.join(load_from, 'args.pkl'), 'rb') as areader:
            modelargs = pickle.load(areader)

    modelargs.update(kwargs)
    PCmodel = model(**modelargs)
    PCmodel.load(dataset, load_from=load_from)
    dataset = dataset if return_data else dataset.clean()
    return dict(model=PCmodel, dataset=dataset, results=results,
                labels=dataset.labels(), args=kwargs, objective=valid_acc)
