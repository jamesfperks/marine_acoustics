"""
Plot the confusion matrix.
"""


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, normalize=None,
                          title='Confusion matrix: ABZ call classification'):
    """sklearn confusion matrix display object."""
    
    display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                            display_labels=['A', 'B', 'Z'],
                            cmap='Blues',
                            normalize=None,
                            colorbar=False) # 'true', 'pred', 'all', None
    
    plt.title(title)
    
    return display.ax_
    
    