"""
Evaluate the model performance.

"""


import numpy as np
from marine_acoustics.configuration import settings as s
from marine_acoustics.results import metrics


def get_results():
    """Calculate and print classification scoring metrics."""
        
    # Load ground truth labels
    y_train = np.load(s.SAVE_DATA_FILEPATH + '/y_train.npy')
    y_test = np.load(s.SAVE_DATA_FILEPATH + '/y_test.npy') 
    
    # Load predicted probabilities for positive class
    y_train_proba = np.load(s.SAVE_PREDICTIONS_FILEPATH + '/y_train_proba.npy')
    y_test_proba = np.load(s.SAVE_PREDICTIONS_FILEPATH + '/y_test_proba.npy')
    
    # Class binary predictions
    y_train_pred = (y_train_proba >= 0.5).astype(int)
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    
    # Accuracy
    train_score, test_score = metrics.get_accuracy(y_train, y_train_pred,
                                                   y_test, y_test_pred)
    
    # Median Filter Accuracy
    train_med_score, test_med_score = metrics.median_filter_accuracy(y_train,
                                            y_train_pred, y_test, y_test_pred)
    
    # Confusion matrix
    c_matrix = metrics.calculate_confusion_matrix(y_test, y_test_pred)
    
    # F1
    f1, f1_med = metrics.calculate_f1(y_test, y_test_pred)
    
    # Calculate AUC and plot ROC curve
    roc_auc, medfilt_roc_auc = metrics.plot_roc(y_test, y_test_proba)
    
    # Print results
    print_results(train_score, test_score, train_med_score, test_med_score,
                  c_matrix, f1, f1_med, roc_auc, medfilt_roc_auc)


def print_results(train_score, test_score, train_med_score, test_med_score,
                  c_matrix, f1, f1_med, roc_auc, medfilt_roc_auc):
    """Print classification scoring metrics."""
    
    # Results Header
    print('\n' + '-'*s.HEADER_LEN + '\nRESULTS\n' + '-'*s.HEADER_LEN)
    
    # Accuracy
    print('\nAccuracy:\n' + '-'*s.SUBHEADER_LEN +
          f'\n  - Training: {train_score:.3f}\n  - Testing: {test_score:.3f}')
    
    # Median Filter Accuracy
    print('\n' + '\nMedian Filtered Accuracy:\n' + '-'*s.SUBHEADER_LEN +
          f'\n  - Training: {train_med_score:.3f}' +
          f'\n  - Testing: {test_med_score:.3f}')
    
    # Confusion Matrix                
    print('\n' + '\nConfusion Matrix:\n' + '-'*s.SUBHEADER_LEN + 
          '\n   [TN FP]' + '-'*3 + f'{c_matrix[0]}' + 
          '\n   [FN TP]' + '-'*3 + f'{c_matrix[1]}')
    
    # F1               
    print('\n'*2 + 'F1:\n' + '-'*s.SUBHEADER_LEN +
          f'\n  - F1: {f1:.2f}' +
          f'\n  - F1 (median filtered): {f1_med:.2f}')
    
    # ROC AUC
    print('\n'*2 + 'ROC Curve:\n' + '-'*s.SUBHEADER_LEN +
          f'\n  - ROC AUC: {roc_auc:.2f}' +
          f'\n  - ROC AUC (median filtered): {medfilt_roc_auc:.2f}')
    
    