"""
Calculate model performance metrics.

"""


from sklearn.metrics import confusion_matrix


def calculate_confusion_matrix(y_true, y_pred):
    """Caclulate the confusion matrix."""
    
    c_matrix = confusion_matrix(y_true, y_pred)

    return c_matrix

