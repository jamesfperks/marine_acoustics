"""
Calculate model performance metrics for binary classification.

"""


import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score)                     
from marine_acoustics.configuration import settings as s


def get_metrics(y_test, y_proba, y_pred):
    """Return a dictionary containing a selection of evaluation metrics."""
    
    metrics_dict = {}
    
    # Accuracy
    metrics_dict['accuracy'] = get_accuracy(y_test, y_pred)
    
    # Confusion matrix
    metrics_dict['c_matrix'] = calculate_confusion_matrix(y_test, y_pred)
    
    # F1
    metrics_dict['f1'] = calculate_f1(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = compute_medfilt_roc(y_test, y_proba)
    metrics_dict.update({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
    
    # ROC AUC
    metrics_dict['roc_auc'] = calculate_roc_auc(fpr, tpr)
    
    return metrics_dict


def get_accuracy(y_test, y_pred):
    """Return accuracy classification score for the test set."""
    
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy


def calculate_confusion_matrix(y_true, y_pred):
    """Caclulate the confusion matrix."""
    
    c_matrix = confusion_matrix(y_true, y_pred)

    return c_matrix


def calculate_f1(y_true, y_pred):
    """Compute the F1 score."""
    
    f1 = f1_score(y_true, y_pred)
    
    return f1


def calculate_roc_auc(fpr, tpr):
    """Calculate ROC AUC given fpr, tpr."""
    
    roc_auc = auc(fpr, tpr)
    
    return roc_auc


def compute_medfilt_roc(y_true, y_proba, drop_intermediate=True):
    """Compute Receiver operating characteristic (ROC)"""
    
    # ------------------- code from sklearn
    
    fps, tps, thresholds = positives_per_threshold(y_true, y_proba)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    # get dtype of `y_score` even if it is an array-like
    thresholds = np.r_[np.inf, thresholds]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    return fpr, tpr, thresholds

    
def positives_per_threshold(y_true, y_score, pos_label=1):
    """Calculate true and false positives per threshold."""

    # sort scores in descending order
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_desc_score = y_score[desc_score_indices]
 
    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_desc_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    thresholds = y_desc_score[threshold_idxs]

    tps = np.zeros(thresholds.size)
    fps = np.zeros(thresholds.size)
    
    for i in range(thresholds.size):
      y_pred = np.where(y_score >= thresholds[i], 1, 0)
      y_medfilt_pred = medfilt(y_pred, kernel_size=s.MEDIAN_FILTER_SIZE)

      tps[i] = np.sum((y_medfilt_pred == 1) & (y_true == 1))
      fps[i] = np.sum((y_medfilt_pred == 1) & (y_true == 0))
    
    return fps, tps, thresholds

