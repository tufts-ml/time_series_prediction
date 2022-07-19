from __future__ import absolute_import, division, print_function

from tensorflow.keras import activations
from tensorflow.keras.backend import exp, relu
from tensorflow.keras.utils import custom_object_scope


__all__ = ['exp_relu', 'get_activation']


def exp_relu(x):
    return exp(-relu(x))

def get_activation(identifier):
    if identifier is None:
        return None
    with custom_object_scope(_get_activations_scope_dict()):
        return activations.get(identifier)

def _get_activations_scope_dict():
    return {
        'exp_relu': exp_relu,
    }
