import skorch
import numpy as np
import torch

from RNNBinaryClassifierModule import RNNBinaryClassifierModule

class RNNBinaryClassifier(skorch.NeuralNet):

    def __init__(
            self,
            criterion=torch.nn.CrossEntropyLoss,
            clip=0.25,
            lr=1.00,
            *args,
            **kwargs,
            ):
        self.clip = clip
        kwargs.update(module=RNNBinaryClassifierModule)
        super(RNNBinaryClassifier, self).__init__(
            criterion=criterion, lr=lr, *args, **kwargs)

    def score(self, X, y, sample_weight=None):
        return self.module_.score(X, y, sample_weight)

    '''
    def on_epoch_begin(self, *args, **kwargs):
        super().on_epoch_begin(*args, **kwargs)

    def train_step(self, X, y):
        self.module_.train()
        self.module_.zero_grad()
        yproba_N2, _  = self.module_.forward(X)
        loss = self.get_loss(yproba_N2, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.clip)
        for p in self.module_.parameters():
            p.data.add_(-self.lr, p.grad.data)
        return {'loss': loss, 'yproba': yproba_N2}

    def validation_step(self, X, y):
        self.module_.eval()
        yproba_N2, _ = self.module_.forward(X)
        loss = self.get_loss(yproba_N2, y)
        return {'loss': loss, 'yproba':yproba_N2}

    def predict_proba(self, X):
        return super().predict_proba(X)

    def predict(self, X):
        return np.argmax(super().predict_proba(X), -1)
    '''
