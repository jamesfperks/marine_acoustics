"""
Plot ROC curve.
"""


import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from marine_acoustics.configuration import settings as s


def plot_roc(fpr, tpr, roc_auc, ax=None,
             name=f'{s.MODEL}-{s.FEATURES}',
             plot_chance_level=False,
             title='Receiver Operating Characteristic (ROC)'):
    
    """Plot the ROC curve."""

    # Plot median filtered ROC
    display = RocCurveDisplay(fpr=fpr, tpr=tpr,
                              roc_auc=roc_auc,
                              estimator_name=name)
    display.plot(ax=ax, plot_chance_level=plot_chance_level)
    
    # Plot formatting
    plt.title(title)
    
    return display.ax_
    
    