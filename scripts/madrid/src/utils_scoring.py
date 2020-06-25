from sklearn.metrics import (
     accuracy_score,
     f1_score,
     average_precision_score,
     precision_score, recall_score,
     balanced_accuracy_score)

HARD_DECISION_SCORERS = dict(
     accuracy=(accuracy_score, {}),
     balanced_accuracy=(balanced_accuracy_score, {}),
     f1_score=(f1_score, dict(pos_label=1, average='binary')),
     precision_score=(precision_score, dict(pos_label=1, average='binary')),
     recall_score=(recall_score, dict(pos_label=1, average='binary')),
     )
THRESHOLD_SCORING_OPTIONS = list(HARD_DECISION_SCORERS.keys())

def calc_score_for_binary_predictions(
          y_true=None,
          y_pred=None,
          scoring='balanced_accuracy'):

     calc_score_func, kwargs = HARD_DECISION_SCORERS[scoring]
     score = calc_score_func(y_true, y_pred, **kwargs)
     return score           