"""
Calculate model performance metrics.

"""


from scipy.signal import medfilt
from sklearn.metrics import accuracy_score, confusion_matrix
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
    
    c_matrix = confusion_matrix(y_true, y_pred)

    return c_matrix

