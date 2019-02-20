import numpy as np
from sklearn.linear_model import LogisticRegression

# Modifies sklearn's LogisticRegression to allow custom thresholds

class LogisticRegressionWithThreshold(LogisticRegression):

    # See LogisticRegression.__init__() in sklearn source code
    # linear_model/logistic.py. This is not robust to changes in sklearn,
    # but it won't let me pass through **kwargs to a subclass constructor.
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None, threshold=0):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.threshold = threshold

    # Same as LinearClassifierMixin.predict() in sklearn source code
    # linear_model/base.py, but with a custom threshold rather than 0.
    def predict(self, X):
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > self.threshold).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]
