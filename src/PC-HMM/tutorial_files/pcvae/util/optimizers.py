from tensorflow.keras import optimizers

def get_optimizer(optimizer, lr=0.01, **kwargs):
    if optimizer is None:
        optimizer = 'adam'
    if type(optimizer) is str:
        optimizer = dict(adam=optimizers.Adam, default=optimizers.Adam,
                         sgd=optimizers.SGD, rmsprop=optimizers.RMSprop)[optimizer]
        optimizer = optimizer(lr=lr)
    return optimizer