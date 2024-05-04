"""
Extract samples for binary classification from .wav files.
"""


import matplotlib.pyplot as plt
import random
import numpy as np
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import read, label
from marine_acoustics.data_processing.features import binary_features


def extract_samples(site, gb_wavfile, df_folder_structure, is_train):
    """Generate labelled samples for a site given all call logs."""
    
    # For .wav in groupby object
    for wavfile, logs in gb_wavfile:
        
        # Read in audio
        y, sr_default = read.read_audio(site, wavfile, df_folder_structure)
        
        # Frame and extract features
        y_features = binary_features.extract_features(y)
        

        # Label features
        y_labelled_features = label.label_features(y_features,
                                                   logs,
                                                   sr_default)
        
        # Balance training samples and test samples (if selected)
        if (is_train == True) or (s.IS_TEST_BALANCED == True):
            y_labelled_features = balance_dataset(y_labelled_features)
        
        # Write the features and labels from the .wav file to the temp folder
        write_samples_to_temp_folder(y_labelled_features, site,
                                     wavfile, is_train)


def write_samples_to_temp_folder(y_labelled_features, site, wavfile, is_train):
    """Write the features and labels from the .wav file to the
    temp data folder. This reduces memory requirements for the script."""
    
    # Require that the sample list is not empty
    # very few samples begin and end on different .wav files, so the
    # start index is greater than the end index, causing an empty slice to be 
    # selected.
    
    if len(y_labelled_features) > 0:
        X_wav, y_wav = split_samples(y_labelled_features)
        if is_train:
            temp_data_fp = s.SAVE_DATA_FILEPATH + 'temp/train-data/'
        else:
            temp_data_fp = s.SAVE_DATA_FILEPATH + 'temp/test-data/'
            
        X_data_fp = temp_data_fp + 'X/' + site + '-' + wavfile + '-X.npy'
        y_data_fp = temp_data_fp + 'y/' + site + '-' + wavfile + '-y.npy' 
        np.save(X_data_fp, X_wav)
        np.save(y_data_fp, y_wav)
        

def balance_dataset(samples):
    """Sub-sample the majority class to balance the dataset."""
    
    one_indexes = []
    zero_indexes = []
    
    # Find sample indexes for positive and negative class
    for i in range(len(samples)):
        if samples[i][1] == 1:
            one_indexes.append(i)
        else:
            zero_indexes.append(i)
    
    if len(zero_indexes) > len(one_indexes):
        major_indexes = zero_indexes
        min_indexes = one_indexes
    else:
        print('Warning: undersampling majority class whale calls')
        major_indexes = one_indexes
        min_indexes = zero_indexes
        
    # Randomly sub-sample indexes from majority to match minority
    random.seed(s.SEED)
    sampled_major_indexes = random.sample(major_indexes, len(min_indexes))
    
    # Recombine major and min indexes preserving sample order
    balanced_indexes = min_indexes + sampled_major_indexes
    balanced_indexes.sort()
    
    # Index samples using balanced indexes
    balanced_samples = [samples[i] for i in balanced_indexes]

    return balanced_samples


def split_samples(samples):
    """Split a list of sample tuples [(X1, y1), (X2, y2), ...] into X, y.
    
    Return numppy arrays
      X: (n_samples, features) list of feature vectors/matrix
      y: (n_samples,) list of labels
      
    """
    
    y = np.asarray([y for X, y in samples])
    
    # Attempt to save memory by reassigning "samples" instead of creating X?
    samples = np.asarray([X for X, y in samples])
    
    return samples, y

