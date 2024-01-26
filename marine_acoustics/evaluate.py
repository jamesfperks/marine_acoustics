"""
Evaluate the model performance.

"""


import numpy as np
from sklearn.metrics import confusion_matrix


def get_results(clf, X_train, y_train, X_test, y_test):
    """Calculate and print classification results."""
    
    # Accuracy
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print('\n'*2 + '-'*50 + '\nRESULTS\n' + '-'*50)
    print(f'\n  - Training: {train_score:.3f}\n  - Testing: {test_score:.3f}')
    
    # Confusion matrix
    tn, fp, fn, tp = calculate_confusion_matrix(X_test, y_test, clf)
    
    
def calculate_confusion_matrix(X_test, y_test, clf):
    """Caclulate and print confusion matrix."""
    
    y_pred = clf.predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred)
    
    # Printout
    ref = np.array([['TN', 'FP'], ['FN', 'TP']])
                  
    print('\n' + '-'*0 + '\nConfusion Matrix:\n' + '-'*30 + 
          f'\n   {ref[0]}' + '-'*3 + f'{c_matrix[0]}' + 
          f'\n   {ref[1]}' + '-'*3 + f'{c_matrix[1]}')
        
    tn, fp, fn, tp = c_matrix.ravel()
    
    return tn, fp, fn, tp

