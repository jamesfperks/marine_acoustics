"""
Pipeline
"""

import numpy as np
from marine_acoustics.data_processing import label
from marine_acoustics.configuration import settings as s

    
def get_manual_detections(logs, sr_default):
    
    # manual detection start and end times (s)
    true_call_times_unsorted = label.get_call_indexes(logs, sr_default)
    true_call_times = np.sort(true_call_times_unsorted, axis=0)/s.SR

    return true_call_times

