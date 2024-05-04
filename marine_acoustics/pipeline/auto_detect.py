# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:33:56 2024

@author: james
"""


import torch
import numpy as np
import itertools
from scipy.signal import medfilt
from marine_acoustics.configuration import settings as s


def get_probabilities(model, X_test):
    """get model predictions given X_test."""

    if s.MODEL == 'HGB':
        y_proba = model.predict_proba(X_test)[:,1]
        
    elif s.MODEL == 'CNN':
         # Torch expects n_samples x n_channels x w x h
        # Add dimension of 1 to represent n_channels = 1
        X_test = np.expand_dims(X_test, axis=1)
        X_test = torch.from_numpy(X_test)

        batch_size = s.PRED_BATCH_SIZE
        predictions = []
        with torch.no_grad():
            model.eval()
    
            for i in range(0, len(X_test), batch_size):
                X_test_batch = X_test[i:i+batch_size]
                batch_pred = np.squeeze(model(X_test_batch).detach().numpy())
                predictions.append(batch_pred)
    
        y_proba = np.concatenate(predictions)

    else: raise ValueError()

    return y_proba


def get_detection_times(y_proba, threshold):
    
    y_pred = get_auto_predictions(y_proba, threshold)

    # Get times of all model call detections
    auto_call_times = get_auto_call_times(y_pred)
    
    # if no calls detected return []
    if auto_call_times.size == 0:
        detection_times = []
        
    else:
        # find midpoints of all automated detections
        all_detection_midpoints = get_midpoints(auto_call_times)
    
        # remove overlapping automated detection midpoints
        detection_midpoints = remove_overlaps(all_detection_midpoints)
    
        # automated detection start and end times
        detection_times = get_detection_bounds(detection_midpoints)

    return detection_times


def get_auto_predictions(y_proba, threshold):
    
    # Convert to binary predictions and apply median filter
    y_pred = (y_proba >= threshold).astype(int)
    y_pred = medfilt(y_pred, kernel_size=s.MEDIAN_FILTER_SIZE)
    
    return y_pred


def get_auto_call_times(y_pred):
    """Start and end times of all s.MIN_CONSEC or more conecutive detections."""
    
    i_frame_start = 0
    call_frame_idxs = []

    # Group all consec. frames
    for k, g in itertools.groupby(y_pred):
    
        # Length of current group of consecutive frames 
        g_len = len(list(g))
    
        # End frame index of current group (end idx inclusive)
        i_frame_end = i_frame_start + g_len - 1
        
        # call detected if there are more than min_consec +ve frames 
        if (k == 1) and (g_len >= s.MIN_CONSEC):
            call_frame_idxs.append((i_frame_start, i_frame_end))
    
        # update to index of the first frame in the next group
        i_frame_start += g_len
    
    # convert to numpy array
    call_frame_idxs = np.asarray(call_frame_idxs)
    
    # Convert frame indexes to sample indexes for the midpoint of each frame
    # convert sample index to time
    auto_call_idxs = call_frame_idxs * s.HOP_LENGTH + (s.FRAME_LENGTH//2)
    auto_call_times = auto_call_idxs/s.SR

    return auto_call_times


def get_midpoints(auto_call_times):
    
    # Calculate detectin midpoints
    diff = np.squeeze(np.diff(auto_call_times, axis=1), axis=1)
    midpoints = []

    for i in range(diff.shape[0]):
        if diff[i] < 2*s.D:
            midpoint = np.mean(auto_call_times[i,:])
            midpoints.append(midpoint)
    
        else:
            n_calls = int(diff[i]//s.D)
            leftover = diff[i] - (n_calls*s.D)
            start = auto_call_times[i,0]
            midpoint = start + (leftover/2) + s.D/2
            for k in range(n_calls):
                midpoints.append(midpoint)
                midpoint+=s.D

    return midpoints

def remove_overlaps(midpoints):

    filtered_midpoints = [midpoints[0]]

    for i in range(1, len(midpoints)):
        diff = midpoints[i] - filtered_midpoints[-1]

        if diff >= s.D:
            filtered_midpoints.append(midpoints[i])

    return filtered_midpoints


def get_detection_bounds(midpoints):

    detection_bounds = np.zeros((len(midpoints),2))

    for i in range(len(midpoints)):
        mid = midpoints[i]
        detection_bounds[i,:] = [mid-s.D/2, mid+s.D/2]

    return detection_bounds

    


