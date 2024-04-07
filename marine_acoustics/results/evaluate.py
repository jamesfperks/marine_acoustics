"""
Evaluate the model performance.

"""


import numpy as np
from scipy.signal import medfilt
from marine_acoustics.configuration import settings as s
from marine_acoustics.results import metrics
from marine_acoustics.visualisation import roc


def get_results():
    """Calculate and print classification scoring metrics."""
        
    # Load probabilities and ground truth labels
    y_test, y_proba = load_predictions()
    
    # Convert to binary predictions
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Apply median filter
    y_pred = medfilt(y_pred, kernel_size=s.MEDIAN_FILTER_SIZE)
    
    # Calculate metrics
    metrics_dict = get_metrics(y_test, y_proba, y_pred)
    
    # Print results
    print_results(metrics_dict)


def load_predictions():
    """Load predicted probabilities and the associated ground truth labels."""
    
    # Load ground truth labels
    y_test = np.load(s.SAVE_DATA_FILEPATH + 'y_test.npy') 
    
    # Load predicted probabilities for positive class
    y_proba = np.load(s.SAVE_PREDICTIONS_FILEPATH + s.MODEL + '-' +
                           s.FEATURES + '-y-proba.npy')
    
    return y_test, y_proba


def get_metrics(y_test, y_proba, y_pred):
    """Return a dictionary containing a selection of evaluation metrics."""
    
    metrics_dict = {}
    
    # Accuracy
    metrics_dict['accuracy'] = metrics.get_accuracy(y_test, y_pred)
    
    # Confusion matrix
    metrics_dict['c_matrix'] = metrics.calculate_confusion_matrix(y_test, y_pred)
    
    # F1
    metrics_dict['f1'] = metrics.calculate_f1(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = metrics.compute_medfilt_roc(y_test, y_proba)
    metrics_dict.update({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
    
    # ROC AUC
    metrics_dict['roc_auc'] = metrics.calculate_roc_auc(fpr, tpr)
    
    return metrics_dict


def print_results(metrics_dict):
    """Print classification scoring metrics."""
    
    accuracy = metrics_dict['accuracy']
    c_matrix = metrics_dict['c_matrix']
    f1 = metrics_dict['f1']
    fpr = metrics_dict['fpr']
    tpr = metrics_dict['tpr']
    roc_auc = metrics_dict['roc_auc']

    
    # Results Header
    print('\n' + '-'*s.HEADER_LEN + '\nRESULTS\n' + '-'*s.HEADER_LEN)
    
    # Accuracy
    print('\nAccuracy:\n' + '-'*s.SUBHEADER_LEN +
          f'\n  - Testing: {accuracy:.2f}')
    
    # Confusion Matrix                
    print('\n' + '\nConfusion Matrix:\n' + '-'*s.SUBHEADER_LEN + 
          '\n   [TN FP]' + '-'*3 + f'{c_matrix[0]}' + 
          '\n   [FN TP]' + '-'*3 + f'{c_matrix[1]}')
    
    # F1               
    print('\n'*2 + 'F1:\n' + '-'*s.SUBHEADER_LEN +
          f'\n  - F1: {f1:.2f}')
    
    # ROC AUC
    print('\n'*2 + 'ROC Curve:\n' + '-'*s.SUBHEADER_LEN +
          f'\n  - ROC AUC: {roc_auc:.2f}')
    
    # Plot ROC curve
    roc.plot_roc(fpr, tpr, roc_auc)
    
    