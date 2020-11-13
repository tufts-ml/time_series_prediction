import optuna
from copy import deepcopy
from .trials import run_trial
import numpy as np
import os
from ..util.util import joinpath, current_time


class hp_param(object):
    def __init__(self, min=0, max=1, step=1):
        self.min = min
        self.max = max
        self.step = step

    def get(self, name, trial):
        raise NotImplementedError()


class integer(hp_param):
    def get(self, name, trial):
        if self.step > 1:
            return int(trial.suggest_discrete_uniform(name, self.min, self.max, self.step))
        return trial.suggest_int(name, self.min, self.max)


class categorical(hp_param):
    def get(self, name, trial):
        return trial.suggest_categorical(name, self.min)


class uniform(hp_param):
    def get(self, name, trial):
        return trial.suggest_uniform(name, self.min, self.max)


class loguniform(hp_param):
    def get(self, name, trial):
        return trial.suggest_loguniform(name, self.min, self.max)


class discrete(hp_param):
    def get(self, name, trial):
        return trial.suggest_discrete_uniform(name, self.min, self.max, self.step)

class grid(object):
    def __init__(self, values):
        self.values = values

class conditional(object):
    def __init__(self, values):
        self.values = values

    def get(self, name, trial):
        return trial.suggest_categorical(name, [(k, v) for k, v in self.values.items()])

def get_args(args, trial):
    output = {}
    for k, v in args.items():
        if isinstance(v, hp_param):
            output[k] = v.get(k, trial)
        elif isinstance(v, conditional):
            v, d = tuple(v.get(k, trial))
            output[k] = v
            output.update(get_args(d, trial))
        else:
            output[k] = v
    return output

def get_grid(arg_dict=dict(), **kwargs):
    all_args = list(kwargs.items())
    d = dict(**arg_dict)
    if len(all_args) == 0:
        yield d
    else:
        name, arg = all_args[0]
        if not isinstance(arg, grid):
            arg = grid([arg])
        for a in arg.values:
            d[name] = a
            yield from get_grid(arg_dict=d, **dict(all_args[1:]))


def opt_hyperparameters(dataset, model, save=None, n_trials=50, download_dir=None, return_data=True, verbose=2, results=None,
                        grid=0,
                        **kwargs):

    # Just use run_trial if no hp arguments
    if not any([isinstance(a, hp_param) for k, a in kwargs.items()]):
        return run_trial(dataset, model, save=save, verbose=1, download_dir=download_dir, return_data=return_data,
                         results=results, **kwargs)

    results = {} if results is None else results
    best_obj, best_output = [-np.inf], [None]
    cur_time = current_time()

    def objective(trial=None):
        if save:
            trial_save = os.path.join(save, cur_time, str(grid), str(trial))

        args = get_args(kwargs, trial)
        output = run_trial(dataset, model, trial_save, results, return_data=return_data, download_dir=download_dir, verbose=verbose, **args)
        best = output['objective'] > best_obj[-1]
        best_obj.append(output['objective'] if best else best_obj[-1])
        best_output[0] = output if best else best_output[0]
        return output['objective']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    finalargs = deepcopy(kwargs)
    finalargs.update(study.best_params)
    best_output[0]['args'] = finalargs
    best_output[0]['hp_trace'] = best_obj[1:]
    return best_output[0]


def run_trials(dataset, model, save=None, n_trials=1, download_dir=None, return_data=True, results=None, **kwargs):
    results = {} if results is None else results
    kwargs.update(dict(model=model))
    outs = [
        opt_hyperparameters(dataset=dataset, save=save, n_trials=n_trials, download_dir=download_dir, return_data=return_data,
                            results=results, grid=i, **args)
        for i, args in enumerate(get_grid(**kwargs))]

    best_output = outs[0]
    for out in outs[1:]:
        if out['objective'] > best_output['objective']:
            best_output = out

    if save:
        best_output['model'].save(joinpath(save, file='weights'))
    return best_output
