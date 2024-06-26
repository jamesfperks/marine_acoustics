"""
Calculate model performance metrics for multiclass classification.

"""


import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             f1_score, ConfusionMatrixDisplay, RocCurveDisplay)


def get_metrics(y_test, y_proba, y_pred):
    """Return a dictionary containing a selection of evaluation metrics."""
    
    metrics_dict = {}
    
    # Accuracy
    metrics_dict['accuracy'] = get_accuracy(y_test, y_pred)
    
    # Confusion matrix
    metrics_dict['c_matrix'] = calculate_confusion_matrix(y_test, y_pred)
    
    # F1
    metrics_dict['f1'] = calculate_f1(y_test, y_pred)
        
    # ROC AUC
    metrics_dict['roc_auc'] = calculate_roc_auc(y_test, y_proba)
    
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
    
    f1 = f1_score(y_true, y_pred, average='micro')
    
    return f1


def calculate_roc_auc(y_true, y_proba):
    """Calculate ROC AUC given fpr, tpr."""
    
    roc_auc = roc_auc_score(y_true, y_proba,
                            average='micro', multi_class='ovr')
    
    return roc_auc    
    
    