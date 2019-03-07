import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Extends an existing sklearn classifier with a custom probability threshold
# for prediction.
# See https://github.com/dtak/prediction-constrained-topic-models/blob/master/
#     pc_toolbox/binary_classifiers/train_and_eval_sklearn_binary_classifier.py
class ThresholdClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clf, threshold=0.5, classes=None):
        self.clf = clf
        self.threshold = threshold
        try:
            self.classes_ = clf.classes_
        except AttributeError:
            if classes is not None:
                self.classes_ = classes
            else:
                self.classes_ = [0,1]
        assert len(self.classes_) == 2

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def decision_function(self, X):
        return self.clf.decision_function(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        probs = self.predict_proba(X)[:,1]
        return np.where(probs <= self.threshold, *self.classes_)


