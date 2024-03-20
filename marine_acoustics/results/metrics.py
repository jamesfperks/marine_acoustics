"""
Calculate model performance metrics.

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             RocCurveDisplay)
                            
from marine_acoustics.configuration import settings as s


def get_accuracy(y_train, y_train_pred, y_test, y_test_pred):
    """Return accuracy classification score for the train and test sets."""
    
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score
    

def median_filter_accuracy(y_train, y_train_pred, y_test, y_test_pred):
    """Apply a median filter to model predictions."""
    
    # Apply median filter
    y_train_med_pred = medfilt(y_train_pred, kernel_size=s.MEDIAN_FILTER_SIZE)
    y_test_med_pred = medfilt(y_test_pred, kernel_size=s.MEDIAN_FILTER_SIZE)
    
    # Recalculate accuracy with filtered predicitons
    train_med_score = accuracy_score(y_train, y_train_med_pred)
    test_med_score = accuracy_score(y_test, y_test_med_pred)
    
    return train_med_score, test_med_score


def calculate_confusion_matrix(y_true, y_pred):
    """Caclulate the confusion matrix."""
    
    y_med_pred = medfilt(y_pred, kernel_size=s.MEDIAN_FILTER_SIZE)
    
    c_matrix = confusion_matrix(y_true, y_med_pred)

    return c_matrix


def calculate_f1(y_true, y_pred):
    """Compute the F1 score."""
    
    f1 = f1_score(y_true, y_pred)
    
    y_med_pred = medfilt(y_pred, kernel_size=s.MEDIAN_FILTER_SIZE)
    
    f1_med = f1_score(y_true, y_med_pred)
    
    return f1, f1_med


def plot_roc(y_test, y_test_pred_proba):
    """Plot the ROC curve and return AUC."""

    # Plot ROC
    roc_display = RocCurveDisplay.from_predictions(y_test, y_test_pred_proba,
                        name='HistGradientBoost')   
    ax = roc_display.ax_
    roc_auc = roc_display.roc_auc
    
    # Plot median filtered ROC
    fpr, tpr, thresholds = compute_medfilt_roc(y_test, y_test_pred_proba)
    medfilt_roc_auc = auc(fpr, tpr)
    medfilt_display = RocCurveDisplay(fpr=fpr, tpr=tpr,
                                      roc_auc=medfilt_roc_auc,
                                      estimator_name='MedianFilter')
    medfilt_display.plot(ax=ax, plot_chance_level=True)
    
    # Plot formatting
    plt.title('Receiver Operating Characteristic (ROC)')
    
    # Reorder legend
    h, l = ax.get_legend_handles_labels()
    leg_order = [1,0,2]
    plt.legend([h[idx] for idx in leg_order],[l[idx] for idx in leg_order])
    
    return roc_auc, medfilt_roc_auc


def compute_medfilt_roc(y_true, y_score, drop_intermediate=True):
    """Compute Receiver operating characteristic (ROC)"""
    
    fps, tps, thresholds = positives_per_threshold(y_true, y_score)

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
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    return fpr, tpr, thresholds

    
def positives_per_threshold(y_true, y_score):
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

