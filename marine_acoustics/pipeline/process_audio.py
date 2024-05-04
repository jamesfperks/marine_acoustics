# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:16:16 2024

@author: james
"""


from marine_acoustics.pipeline import manual_detect, auto_detect, match
from marine_acoustics.data_processing import read
from marine_acoustics.data_processing.features import binary_features


def read_wav(site, wavfile, logs, df_folder_structure, model):
    
    # Read in audio
    y, sr_default = read.read_audio(site, wavfile, df_folder_structure)
    
    # Frame and extract features
    y_features = binary_features.extract_features(y)

    # Get model predicted probabilities
    y_proba = auto_detect.get_probabilities(model, y_features)
    
    # manual detection start and end times (s)
    true_call_times = manual_detect.get_manual_detections(logs, sr_default)
    
    return y, y_proba, true_call_times


def get_cmatrix(y, y_proba, threshold, true_call_times):
    
    # automated detection start and end times
    detection_times = auto_detect.get_detection_times(y_proba, threshold)
    
    # count detection matches
    TP, FP, FN = match.count_matches(detection_times, true_call_times)
    TN = match.calculate_tn(y, true_call_times, FP)
    
    return TP, FP, FN, TN

