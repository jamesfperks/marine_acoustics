"""
Evaluate the model performance.

"""


from marine_acoustics.configuration import settings as s
from marine_acoustics.model import metrics


def get_results(clf, X_train, y_train, X_test, y_test):
    """Calculate and print classification scoring metrics."""
        
    # Class predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    # CLass probabilities (for positive class "whale")
    y_train_pred_proba = clf.predict_proba(X_train)[:,1] 
    y_test_pred_proba = clf.predict_proba(X_test)[:,1]
    
    
    # Accuracy
    train_score, test_score = metrics.get_accuracy(y_train, y_train_pred,
                                                   y_test, y_test_pred)
    
    # Median Filter Accuracy
    train_med_score, test_med_score = metrics.median_filter_accuracy(y_train,
                                            y_train_pred, y_test, y_test_pred)
    
    # Confusion matrix
    c_matrix = metrics.calculate_confusion_matrix(y_test, y_test_pred)
    
    # Calculate AUC and plot ROC curve
    roc_auc = metrics.plot_roc(y_test, y_test_pred_proba)
    
    # Print results
    print_results(train_score, test_score,
                  train_med_score, test_med_score, c_matrix, roc_auc)


def print_results(train_score, test_score,
                  train_med_score, test_med_score, c_matrix, roc_auc):
    """Print classification scoring metrics."""
    
    # Results Header
    print('\n' + '-'*s.HEADER_LEN + '\nRESULTS\n' + '-'*s.HEADER_LEN)
    
    # Accuracy
    print('\n' + '\nAccuracy:\n' + '-'*s.SUBHEADER_LEN +
          f'\n  - Training: {train_score:.3f}\n  - Testing: {test_score:.3f}')
    
    # Median Filter Accuracy
    print('\n' + '\nMedian Filtered Accuracy:\n' + '-'*s.SUBHEADER_LEN +
          f'\n  - Training: {train_med_score:.3f}' +
          f'\n  - Testing: {test_med_score:.3f}')
    
    # Confusion Matrix                
    print('\n' + '\nConfusion Matrix:\n' + '-'*s.SUBHEADER_LEN + 
          '\n   [TN FP]' + '-'*3 + f'{c_matrix[0]}' + 
          '\n   [FN TP]' + '-'*3 + f'{c_matrix[1]}')
    
    # ROC AUC
    print('\n'*2 + 'ROC Curve:\n' + '-'*s.SUBHEADER_LEN +
          f'\n  - ROC AUC: {roc_auc:.2f}')
    
    