"""
Label audio frames as "background" or "whale".

"""


import numpy as np
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import info


def label_features(y_features, logs, sr_default):
    """Label feature vectors as 1 for 'whale' or 0 for 'background'."""
    
    # Get the sample indexes of start/end of whale calls
    call_indexes = get_call_indexes(logs, sr_default)
    
    # Convert call indexes to frame indexes
    frame_indexes = index2frame(call_indexes)
    
    # Apply whale call labels to features
    feature_labels = apply_labels(y_features, frame_indexes)

    # Create list of sample tuples [(X1, y1), ...]
    y_labelled_features = list(zip(y_features, feature_labels))
    
    return y_labelled_features


def get_call_indexes(logs, sr_default):
    """Read the call log time stamps and calculate the index of the start
    and end of each call for the resampled audio."""
    
    original_audio_indexes = logs[['Beg File Samp (samples)',
                                   'End File Samp (samples)']].to_numpy() - 1
    
    call_indexes = np.rint(s.SR*original_audio_indexes/sr_default)
    
    return call_indexes


def index2frame(call_indexes):
    """Convert call start and end indexes to frame indexes.
    
    Overlapping frames cause an index to appear in multiple frames.
    The start frame index is the last frame in which the start index appears.
    The end frame index is the first frame in which the end index appears.
    
    """
    
    #Label for when frame duration is less than call duration
    # Tested to work well for 3 second frame duration on A and Z calls
    # Start frame idx given by the last frame a sample idx is in
    call_start_idxs = call_indexes[:,0]
    start_frame_idxs = np.asarray(call_start_idxs//s.HOP_LENGTH, dtype=int)

    # End frame idx given by the first frame a sample idx is in
    call_end_idxs = call_indexes[:,1]
    end_frame_idxs = np.asarray((call_end_idxs - s.FRAME_LENGTH + 
                                 s.HOP_LENGTH)//s.HOP_LENGTH, dtype=int)
    
    frame_indexes = np.column_stack((start_frame_idxs, end_frame_idxs))
    
    return frame_indexes


def apply_labels(y_features, frame_indexes):
    """
    Any feature not contained within an annotated frame index is assumed to
    be background, 0. All frames within annotated call indexes are labelled 1.
    
    """

    # Initially assume all features are background "0"
    feature_labels = np.zeros(y_features.shape[0])
    
    # Apply labels to section of annotated frames
    for i in range(frame_indexes.shape[0]):
        start, end = frame_indexes[i]
        
        if end >= len(feature_labels):
            end = len(feature_labels)-1
            
        annotated_indexes = np.arange(start, end+1)
        section_labels = np.ones(len(annotated_indexes),dtype=int)
        feature_labels[annotated_indexes] = section_labels

    return feature_labels


def get_multi_class_labels(logs):
    """
    Return numberic labels for each call type annotaions in the given logs.
    
    0  Bm-A
    1  Bm-B
    2  Bm-Z
    3  Bm-D
    4  Bp-20
    5  Bp-20+
    6  Bp-Downsweep
    7  Unidentified
    
    """
    
    # Dict to convert call string to numeric call label
    call_type_strings = info.get_call_types()
    call_type_nums = list(range(0,len(call_type_strings)))
    class_label_dict = dict(zip(call_type_strings, call_type_nums))
    
    # Convert labels in logs to numeric labels
    call_label_strings = logs['Call Label'].to_list()
    multi_class_labels = [class_label_dict[c] for c in call_label_strings]
    
    return multi_class_labels

