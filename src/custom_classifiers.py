import numpy as np
import copy
from sklearn.base import BaseEstimator, ClassifierMixin

# Extends an existing sklearn classifier with a custom probability threshold
# for prediction.
# See https://github.com/dtak/prediction-constrained-topic-models/blob/master/
#     pc_toolbox/binary_classifiers/train_and_eval_sklearn_binary_classifier.py
class ThresholdClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clf=None, threshold=0.5, classes=None, cache=None, **kwargs):
        if clf is None:
            raise ValueError("Bad classifier")
        self.clf = clf
        self.threshold = threshold
        try:
            self.classes_ = clf.classes_
        except AttributeError:
            if classes is not None:
                self.classes_ = classes
            else:
                self.classes_ = [0, 1]
        #assert len(self.classes_) == 2

        if cache is not None:
            self._cached_clf = cache
        else:
            self._cached_clf = dict()

    def get_params(self, deep=False):
        if deep:
            return {'threshold': self.threshold, 'classes':copy.deepcopy(self.classes_), 'clf':copy.deepcopy(self.clf)}
        else:
            return {'threshold': self.threshold, 'classes':self.classes_, 'clf':self.clf, 'cache':self._cached_clf}

    def set_params(self, clf=None, threshold=None, classes=None, cache=None, **kwargs):
        if clf is not None:
            self.clf = clf
        if threshold is not None:
            self.threshold = threshold
        if classes is not None:
            self.classes_ = classes
        if cache is not None:
            self._cached_clf = cache
        return self

    def fit(self, X, y):
        ## Quick hash of X to check uniqueness
        U = np.random.RandomState(42).randn(X.shape[1])
        key = int(100000 * np.mean(np.dot(X, U))) + 1000 * X.shape[0] + X.shape[1]
        if key not in self._cached_clf:
            ans = self.clf.fit(X, y)
            self._cached_clf[key] = ans
        self.clf = self._cached_clf[key]
        return self

    def decision_function(self, X):
        return self.clf.decision_function(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        probs = self.predict_proba(X)[:,1]
        return np.where(probs <= self.threshold, *self.classes_)


