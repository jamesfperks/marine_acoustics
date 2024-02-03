"""
Evaluate the model performance.

"""


from sklearn.metrics import confusion_matrix


def get_results(clf, X_train, y_train, X_test, y_test):
    """Calculate classification scoring metrics."""
    
    # Accuracy
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    # Confusion matrix
    c_matrix = calculate_confusion_matrix(X_test, y_test, clf)
    
    # Print results
    print_results(train_score, test_score, c_matrix)
    
    
def calculate_confusion_matrix(X_test, y_test, clf):
    """Caclulate and print confusion matrix."""
    
    y_pred = clf.predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred)

    return c_matrix


def print_results(train_score, test_score, c_matrix):
    """Print classification scoring metrics."""
    
    # Results Header
    print('\n'*2 + '-'*50 + '\nRESULTS\n' + '-'*50)
    
    # Accuracy
    print(f'\n  - Training: {train_score:.3f}\n  - Testing: {test_score:.3f}')
    
    # Confusion Matrix                
    print('\n' + '-'*0 + '\nConfusion Matrix:\n' + '-'*30 + 
          '\n   [TN FP]' + '-'*3 + f'{c_matrix[0]}' + 
          '\n   [FN TP]' + '-'*3 + f'{c_matrix[1]}')
    
    