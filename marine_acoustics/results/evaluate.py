"""
Evaluate the model performance.

"""


import numpy as np
from scipy.signal import medfilt
from marine_acoustics.configuration import settings as s
from marine_acoustics.results import binary_metrics, multiclass_metrics
from marine_acoustics.visualisation import roc


def get_results():
    """Calculate and print classification scoring metrics."""
     
    # Load probabilities and ground truth labels
    y_test, y_proba = load_predictions()

    if s.BINARY == True:
        get_binary_class_results(y_test, y_proba)
    else:
        get_multi_class_results(y_test, y_proba)


def load_predictions():
    """Load predicted probabilities and the associated ground truth labels."""
    
    # Load ground truth labels
    y_test = np.load(s.SAVE_DATA_FILEPATH + 'y_test.npy') 
    
    # Load predicted probabilities for positive class
    y_proba = np.load(s.SAVE_PREDICTIONS_FILEPATH + s.MODEL + '-' +
                           s.FEATURES + '-y-proba.npy')
    
    return y_test, y_proba    


def get_binary_class_results(y_test, y_proba):
    """Get metrics for binary classification."""
    
    # Convert to binary predictions
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Apply median filter
    y_pred = medfilt(y_pred, kernel_size=s.MEDIAN_FILTER_SIZE)
    
    # Calculate metrics
    metrics_dict = binary_metrics.get_metrics(y_test, y_proba, y_pred)
    
    # Print results
    print_binary_class_results(metrics_dict)
    

def get_multi_class_results(y_test, y_proba):
    """Get metrics for multiclass classification."""
    
    # Convert probabilities to a class label
    y_pred = np.argmax(y_proba, axis=1)
    
    # Calculate metrics
    metrics_dict = multiclass_metrics.get_metrics(y_test, y_proba, y_pred)
    
    # Print results
    print_multi_class_results(metrics_dict)
    

def print_binary_class_results(metrics_dict):
    """Print classification scoring metrics."""
    
    accuracy = metrics_dict['accuracy']
    c_matrix = metrics_dict['c_matrix']
    f1 = metrics_dict['f1']
    #fpr = metrics_dict['fpr']
    #tpr = metrics_dict['tpr']
    #roc_auc = metrics_dict['roc_auc']

    
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
    

def print_multi_class_results(metrics_dict):
    """Print classification scoring metrics."""
    
    accuracy = metrics_dict['accuracy']
    c_matrix = metrics_dict['c_matrix']
    f1 = metrics_dict['f1']
    roc_auc = metrics_dict['roc_auc']

    
    # Results Header
    print('\n' + '-'*s.HEADER_LEN + '\nRESULTS\n' + '-'*s.HEADER_LEN)
    
    # Accuracy
    print(f'\n  - Accuracy: {accuracy:.2f}')
    
    # F1               
    print(f'\n  - F1: {f1:.2f}')
    
    # ROC AUC
    print(f'\n  - ROC AUC: {roc_auc:.2f}')
    
    # Confusion Matrix                
    print('\n' + '\nConfusion Matrix:\n' + '-'*s.SUBHEADER_LEN + 
          f'\n{c_matrix}')
    
    
    
