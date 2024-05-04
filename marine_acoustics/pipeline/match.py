# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:53:33 2024

@author: james
"""


import numpy as np
from marine_acoustics.configuration import settings as s


def count_matches(detection_times, true_call_times):

    TP, FP, FN = 0,0,0
    
    if detection_times == []:
        FN = true_call_times.shape[0]
        return TP, FP, FN
        
    unmatched_true_call_times = true_call_times
    
    for auto_start, auto_end in detection_times:
        
        overlap_found, match_idx = check_overlap(auto_start, auto_end,
                                                 unmatched_true_call_times)
    
        if overlap_found == True:
            # true positive if overlap found
            # delete corresonding mannual annotation to prevent duplicates
            TP += 1
            unmatched_true_call_times = np.delete(unmatched_true_call_times,
                                                 match_idx, axis=0)
        
        else:
            # false positive detection if no overlap found
            FP += 1
            
    # False negatives = no. of unmatched mannual annotations
    FN = unmatched_true_call_times.shape[0]

    return TP, FP, FN


def check_overlap(auto_start, auto_end, unmatched_true_call_times):

    overlap_found = False
    match_idx = None

    # check all manual annotations
    for k in range(unmatched_true_call_times.shape[0]):
        true_start, true_end = unmatched_true_call_times[k,:]

        # if overlap then there is true detection
        # break from mannual annotation loop to stop duplicate matches
        if max(auto_start, true_start) <= min(auto_end, true_end):
            overlap_found = True
            match_idx = k
            break

    return overlap_found, match_idx





def calculate_tn(y, true_call_times, FP):
    
    # total audio duration (s)
    total_dur = len(y)/s.SR
    
    # total positive duration (s)
    P_dur = true_call_times.shape[0] * s.D
    
    # total negative duration (s)
    N_dur = total_dur - P_dur
    
    # total number of possible negative detections
    N = N_dur//s.D
    
    # true negatives = all negatives - false positives
    TN = N - FP

    return int(TN)

