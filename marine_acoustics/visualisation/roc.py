"""
Plot ROC curve.
"""


import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize
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
    
 
def plot_micro_ovr_roc(y_test, y_proba,
                       ax=None,
                       classes=[0,1,2],
                       name='micro-average OvR',
                       plot_chance_level=False,
                       title='Micro-averaged One-vs-Rest ROC'):
    
    """Plot micro-averaged OvR ROC for all models at one site."""
    
    y_onehot_test = label_binarize(y_test, classes=classes)

    display = RocCurveDisplay.from_predictions(
                            y_onehot_test.ravel(),
                            y_proba.ravel(),
                            ax=ax,
                            name=name,
                            plot_chance_level=plot_chance_level
                            )
    
    display.ax_.set(xlabel="False Positive Rate",
                    ylabel="True Positive Rate",
                    title=title)
    
    return display.ax_
    

def plot_multiclass_roc(y_test, y_proba,
                        title='Receiver Operating Characteristic\n'
                        'One-vs-Rest (OvR) Multiclass'):
    
    """Plot the one vs rest ROC curve for each class, for a given model."""
    
    classes = [0,1,2]
    labels = ['A','B','Z']
    
    y_onehot_test = label_binarize(y_test, classes=classes)
    
    fig, ax = plt.subplots()
    for class_id in range(len(labels)):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_proba[:, class_id],
            name=f" {labels[class_id]}",
            ax=ax,
            plot_chance_level=(class_id == 2))
        
    ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=title)
    
    return ax
    
    