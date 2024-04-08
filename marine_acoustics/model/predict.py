"""
Model prediction probabiliites for the positve class.
Get predictions for both train and test sets.

Return predictions = (y_train_pred_proba, y_test_pred_proba)

"""


import torch
import numpy as np
import torch.nn.functional as F
from joblib import load
from marine_acoustics.model import cnn
from marine_acoustics.configuration import settings as s


def get_predictions():
    """
    Get prediction probabiliites for the positve class
    for the chosen classifier.
    
    Return predictions = (y_train_pred_proba, y_test_pred_proba)
    
    """
    
    # Load samples
    X_test = np.load(s.SAVE_DATA_FILEPATH + 'X_test.npy')
    
    # Get model predictions
    if s.MODEL == 'HGB':
        predictions = pred_grad_boost(X_test)
        
    elif s.MODEL == 'CNN':
        predictions = pred_cnn(X_test)
    
    else:
        raise NotImplementedError('Model chosen not implemented: ', s.MODEL)
        
    # Save predictions
    save_predictions(predictions)
 
    return predictions


def pred_grad_boost(X_test):
    """Positive class predicitons for HistGradientBoostingClassifier."""
      
    # Load model
    model = load(s.SAVE_MODEL_FILEPATH + s.MODEL + '-' + s.FEATURES)
    
    if s.BINARY == True:
        
        # Binary: get probability for +ve class "whale" only (n_samples,)
        y_test_pred_proba = model.predict_proba(X_test)[:,1]
    
    else:
        
        # Multiclass: probabilities for each class (n_samples x n_classes)
        y_test_pred_proba = model.predict_proba(X_test)
    
    return y_test_pred_proba


def pred_cnn(X_test):
    """Positive class predicitons for CNN."""
    
    # Torch expects n_samples x n_channels x w x h
    # Add dimension of 1 to represent n_channels = 1
    X_test = np.expand_dims(X_test, axis=1)
    
    # Create torch tensor
    X_test = torch.from_numpy(X_test)
    
    # Load model
    if s.BINARY == True:
        model = cnn.BinaryNet()
        
    else:
        model = cnn.MultiNet()   
    
    model.load_state_dict(torch.load(s.SAVE_MODEL_FILEPATH +
                                     s.MODEL + '-' + s.FEATURES))
    

    batch_size = s.PRED_BATCH_SIZE
    predictions = []
    with torch.no_grad():
        model.eval()

        for i in range(0, len(X_test), batch_size):
            X_test_batch = X_test[i:i+batch_size]
            batch_pred = np.squeeze(model(X_test_batch).detach().numpy())
            predictions.append(batch_pred)

    predictions = np.concatenate(predictions)
    
    if s.BINARY == False:
        predictions = F.softmax(torch.from_numpy(predictions), dim=1)
    
    return predictions


def save_predictions(predictions):
    """Write predictions to .npy files."""

    y_test_proba = predictions
    
    # Save as binary file in .npy format
    np.save(s.SAVE_PREDICTIONS_FILEPATH + s.MODEL + '-' + s.FEATURES +
            '-y-proba.npy', y_test_proba)

    