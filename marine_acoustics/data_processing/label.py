"""
Label audio frames as "background" or "whale".

"""

import librosa
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
        whale_indexes = np.arange(start, end) # Non-inclusive end frame idx
        feature_labels[whale_indexes] = np.ones(len(whale_indexes))
    
    # Append label to the end of the feature vector for each sample
    y_labelled_features = np.column_stack((y_features, feature_labels))
    
    return y_labelled_features


def get_call_indexes(logs, sr_default):
    """Read the call log time stamps and calculate the index of the start
    and end of each call for the resampled audio."""
    
    original_audio_indexes = logs[['Beg File Samp (samples)',
                                   'End File Samp (samples)']].to_numpy() - 1
    
    call_indexes = np.rint(s.SR*original_audio_indexes/sr_default)
    
    return call_indexes


def index2frame(call_indexes):
    """Convert time log indexes to frame indexes."""
    
    # # Frame index of the last frame that the sample is in
    # frame_indexes = np.apply_along_axis(librosa.samples_to_frames,
    #                                     axis=0,
    #                                     arr=call_indexes,
    #                                     hop_length=s.HOP_LENGTH,
    #                                     n_fft=s.FRAME_LENGTH)
    
  
    offset = int(s.FRAME_LENGTH // 2)

    frame_indexes = np.asarray(np.floor((call_indexes + offset)//s.HOP_LENGTH),
                               dtype=int)

    # # Deal with negative indexes caused by librosa n_fft offset
    # frame_indexes[frame_indexes<0] = 0
    
    # # Check
    # for idx in frame_indexes.flatten():
    #     if idx < 0:
    #         raise ValueError('Negative frame index calculated during sample '
    #                          'index to frame index conversion.')
    
    return frame_indexes

