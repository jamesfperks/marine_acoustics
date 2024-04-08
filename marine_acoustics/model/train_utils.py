"""
Utility functions used during model training.
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from marine_acoustics.configuration import settings as s


def numpy_to_tensor(X, y):
    """
    Convert X, y numpy arrays into tensors compatable with pytorch.
    
    Input:
      X: (n_samples, feature_width, feature_height)
      y: (n_samples,)
      
    Return torch tensors
      X: (n_samples, n_channels, feature_width, feature_height) 
      y: Binary: (n_samples, 1) list of labels
         Multiclass: (n_samples,)
      
    """
    
    # Torch convolution expects n_samples x n_channels x w x h
    # Add dimension of 1 to represent n_channels = 1
    X = np.expand_dims(X, axis=1)
    
    # Create torch tensor
    X = torch.from_numpy(X)
    
    if s.BINARY == True:
        y = torch.from_numpy(y).float().reshape(-1, 1)
        
    else:
        y = torch.from_numpy(y).long()
    
    return X, y


def plot_training_loss(epoch_loss):
    """Plot the training loss per epoch"""
    
    plt.figure()
    plt.plot(range(1, s.N_EPOCHS+1), epoch_loss, color='blue')
    plt.title('Training loss per epoch')
    plt.legend(['Training Loss'], loc='upper right')
    plt.xlabel('Epoch Number')
    plt.ylabel('Training loss (Binary Cross Entropy loss)')
    
    