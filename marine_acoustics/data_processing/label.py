"""
Label audio frames as "background" or "whale".

"""


import numpy as np
from marine_acoustics.configuration import settings as s


def label_features(y_features, logs, sr_default):
    """Label feature vectors as 1 for 'whale' or 0 for 'background'."""
    
    # Get the sample indexes of start/end of whale calls
    call_indexes = get_call_indexes(logs, sr_default)
    
    # Convert call indexes to frame indexes
    frame_indexes = index2frame(call_indexes)
    
    # For each annotation, label corresponding frames 1 if "whale", else 0 
    feature_labels = np.zeros(y_features.shape[0])  
    for start, end in frame_indexes:  
        whale_indexes = np.arange(start, end+1) # Non-inclusive end frame idx
        feature_labels[whale_indexes] = np.ones(len(whale_indexes))
    
    # Create list of sample tuples [(X1, y1), ...]
    y_labelled_features = list(zip(y_features, feature_labels))
    
    return y_labelled_features


def get_call_indexes(logs, sr_default):
    """Read the call log time stamps and calculate the index of the start
    and end of each call for the resampled audio."""
    
    original_audio_indexes = logs[['Beg File Samp (samples)',
                                   'End File Samp (samples)']].to_numpy()
    
    call_indexes = np.rint(s.SR*original_audio_indexes/sr_default) - 1
    
    return call_indexes


def index2frame(call_indexes):
    """Convert call start and end indexes to frame indexes.
    
    Overlapping frames cause an index to appear in multiple frames.
    The start frame index is the last frame in which the start index appears.
    The end frame index is the first frame in which the end index appears.
    
    """
    
    # Start frame idx given by the last frame a sample idx is in
    call_start_idxs = call_indexes[:,0]
    offset = int(s.FRAME_LENGTH // 2)
    start_frame_idxs = np.asarray((call_start_idxs+offset)//s.HOP_LENGTH,
                                 dtype=int)

    # End frame idx given by the first frame a sample idx is in
    call_end_idxs = call_indexes[:,1]
    end_frame_idxs = np.asarray(call_end_idxs//s.HOP_LENGTH, dtype=int)
    
    frame_indexes = np.column_stack((start_frame_idxs, end_frame_idxs))

    return frame_indexes

