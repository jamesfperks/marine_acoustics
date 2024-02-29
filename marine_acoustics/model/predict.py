"""
Model prediction probabiliites for the positve class.
Get predictions for both train and test sets.

Return predictions = (y_train_pred_proba, y_test_pred_proba)

"""


import torch
import numpy as np
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import sample


def get_predictions(train_samples, test_samples, model):
    """
    Get prediction probabiliites for the positve class
    for the chosen classifier.
    
    Return predictions = (y_train_pred_proba, y_test_pred_proba)
    
    """
    
    if s.MODEL == 'HGBC':
        predictions = pred_grad_boost(train_samples, test_samples, model)
        
    elif s.MODEL == 'CNN':
        predictions = pred_cnn(train_samples, test_samples, model)
    
    else:
        raise NotImplementedError('Model chosen not implemented: ', s.MODEL)
 
    return predictions


def pred_grad_boost(train_samples, test_samples, model):
    """Positive class predicitons for HistGradientBoostingClassifier."""
    
    X_train, y_train = sample.split_samples(train_samples)
    X_test, y_test = sample.split_samples(test_samples)
    
    # CLass probabilities (for positive class "whale")
    y_train_pred_proba = model.predict_proba(X_train)[:,1] 
    y_test_pred_proba = model.predict_proba(X_test)[:,1]
    
    predictions = (y_train_pred_proba, y_test_pred_proba)
    
    return predictions


def pred_cnn(train_samples, test_samples, model):
    """Positive class predicitons for CNN."""
    
    X_train, y_train = sample.samples_to_tensors(train_samples)
    X_test, y_test = sample.samples_to_tensors(test_samples)
    
    with torch.no_grad():
        model.eval()

        # Class probabilities (for positive class "whale")
        y_train_pred_proba = np.squeeze(model(X_train).detach().numpy())
        y_test_pred_proba = np.squeeze(model(X_test).detach().numpy())
    
    predictions = (y_train_pred_proba, y_test_pred_proba)
    
    return predictions




