"""
Model prediction probabiliites for the positve class.
Get predictions for both train and test sets.

Return predictions = (y_train_pred_proba, y_test_pred_proba)

"""


import torch
import numpy as np
from marine_acoustics.configuration import settings as s


def get_predictions(model):
    """
    Get prediction probabiliites for the positve class
    for the chosen classifier.
    
    Return predictions = (y_train_pred_proba, y_test_pred_proba)
    
    """
    
    # Load samples
    X_train = np.load(s.SAVE_DATA_FILEPATH + '/X_train.npy')
    X_test = np.load(s.SAVE_DATA_FILEPATH + '/X_test.npy')
    
    # Select model for predictions
    if s.MODEL == 'HGB':
        predictions = pred_grad_boost(X_train, X_test, model)
        
    elif s.MODEL == 'CNN':
        predictions = pred_cnn(X_train, X_test, model)
    
    else:
        raise NotImplementedError('Model chosen not implemented: ', s.MODEL)
 
    return predictions


def pred_grad_boost(X_train, X_test, model):
    """Positive class predicitons for HistGradientBoostingClassifier."""
      
    # Class probabilities (for positive class "whale")
    y_train_pred_proba = model.predict_proba(X_train)[:,1] 
    y_test_pred_proba = model.predict_proba(X_test)[:,1]
    
    predictions = (y_train_pred_proba, y_test_pred_proba)
    
    return predictions


def pred_cnn(X_train, X_test, model):
    """Positive class predicitons for CNN."""
    
    # Torch expects n_samples x n_channels x w x h
    # Add dimension of 1 to represent n_channels = 1
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    # Create torch tensor
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    
    with torch.no_grad():
        model.eval()

        # Class probabilities (for positive class "whale")
        y_train_pred_proba = np.squeeze(model(X_train).detach().numpy())
        y_test_pred_proba = np.squeeze(model(X_test).detach().numpy())
    
    predictions = (y_train_pred_proba, y_test_pred_proba)
    
    return predictions




