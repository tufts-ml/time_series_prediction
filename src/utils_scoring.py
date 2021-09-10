import numpy as np
from sklearn.metrics import (
     accuracy_score,
     f1_score,
     average_precision_score,
     precision_score, recall_score,
     balanced_accuracy_score)

from sklearn.metrics import (make_scorer, accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve)

def calc_cross_entropy_base2_score(y, y_pred_proba):
    return log_loss(y, y_pred_proba, normalize=True) / np.log(2)

HYPERSEARCH_SCORING_OPTIONS = {
    'roc_auc_score':make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True),
    'average_precision_score':make_scorer(average_precision_score, greater_is_better=True, needs_threshold=True),
    'cross_entropy_base2_score':make_scorer(calc_cross_entropy_base2_score, greater_is_better=False, needs_proba=True),
}

average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)

HARD_DECISION_SCORERS = dict(
     accuracy_score=make_scorer(accuracy_score),
     balanced_accuracy_score=make_scorer(balanced_accuracy_score),
     f1_score=make_scorer(f1_score, pos_label=1, average='binary'),
     precision_score=make_scorer(precision_score, pos_label=1, average='binary'),
     recall_score=make_scorer(recall_score, pos_label=1, average='binary'),
     )

HARD_DECISION_PARAMS = dict(
     accuracy_score=(accuracy_score, {}),
     balanced_accuracy_score=(balanced_accuracy_score, {}),
     f1_score=(f1_score, dict(pos_label=1, average='binary')),
     precision_score=(precision_score, dict(pos_label=1, average='binary')),
     recall_score=(recall_score, dict(pos_label=1, average='binary')),
     )

THRESHOLD_SCORING_OPTIONS = list(HARD_DECISION_SCORERS.keys())

def calc_score_for_binary_predictions(
          y_true=None,
          y_pred=None,
          scoring='balanced_accuracy'):

    calc_score_func, kwargs = HARD_DECISION_PARAMS[scoring]
    score = calc_score_func(y_true, y_pred, **kwargs)
    return score           
