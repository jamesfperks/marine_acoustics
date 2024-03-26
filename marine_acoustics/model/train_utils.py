"""
Utility functions used during model training.
"""


import matplotlib.pyplot as plt
from marine_acoustics.configuration import settings as s


def plot_training_loss(epoch_loss):
    """Plot the training loss per epoch"""
    
    plt.figure()
    plt.plot(range(1, s.N_EPOCHS+1), epoch_loss, color='blue')
    plt.title('Training loss per epoch')
    plt.legend(['Training Loss'], loc='upper right')
    plt.xlabel('Epoch Number')
    plt.ylabel('Training loss (Binary Cross Entropy loss)')
    
    