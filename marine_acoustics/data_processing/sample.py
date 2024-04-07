"""
Extract samples from the raw .wav files

"""


import os
import random
import torch
import numpy as np
import pandas as pd
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import info, read, features, label


def get_samples(df_selected_dataset, is_train):
    """
    Get sample set from selected sites and call types.
    Returns samples as a list of tuples [(X1, y1), (X2, y2), ...]
    """
        
    # Sample from selected sites and call types
    sites = df_selected_dataset.index
    call_types = df_selected_dataset.columns
    df_folder_structure = info.get_folder_structure()
    X, y = create_sample_set(sites, call_types,
                                   df_folder_structure, is_train)
    
    return X, y


def create_sample_set(sites, call_types, df_folder_structure, is_train):
    """Return a list of samples from given sites and call-types"""
    
    # Empty temp data folder
    empty_temp_data_dir(is_train)
    
    for site in sites:
        
        # Combine all call-type logs
        df_logs = concat_call_logs(site, call_types, df_folder_structure)

        # Groupby .wav filename
        gb_wavfile = df_logs.groupby('Begin File')
        
        # Generate labelled samples from site and write to temp data folder
        extract_samples(site, gb_wavfile, df_folder_structure, is_train)
    
    X, y = combine_site_samples(is_train)
    
    return X, y


def empty_temp_data_dir(is_train):
    """Empty temporary data folder."""
    
    # Empty temp data folder
    if is_train:
        temp_data_dir = 'temp/train-data/'
    else:
        temp_data_dir = 'temp/test-data/'
        
    X_data_dir = s.SAVE_DATA_FILEPATH + temp_data_dir + 'X/'
    y_data_dir = s.SAVE_DATA_FILEPATH + temp_data_dir + 'y/'
    
    X_filepaths = [X_data_dir + fname for fname in os.listdir(X_data_dir)]
    y_filepaths = [y_data_dir + fname for fname in os.listdir(y_data_dir)]
    
    for file in X_filepaths:
        os.remove(file)
        
    for file in y_filepaths:
        os.remove(file)
        

def combine_site_samples(is_train):
    """Read in X, y for each site and combine. Remove file in temp folder
    after concatenation."""
    
    if is_train:
        temp_data_dir = 'temp/train-data/'
    else:
        temp_data_dir = 'temp/test-data/'
        
    X_data_dir = s.SAVE_DATA_FILEPATH + temp_data_dir + 'X/'
    y_data_dir = s.SAVE_DATA_FILEPATH + temp_data_dir + 'y/'
        
    X_filepaths = [X_data_dir + fname for fname in os.listdir(X_data_dir)]
    y_filepaths = [y_data_dir + fname for fname in os.listdir(y_data_dir)]
    
    X = np.concatenate([np.load(path) for path in X_filepaths])
    y = np.concatenate([np.load(path) for path in y_filepaths])
    
    return X, y
    

def concat_call_logs(site, call_types, df_folder_structure):
    """Return a df of all call logs for a given site and list of call types."""
    
    logs = []
    
    for call_type in call_types:
        df_log = read.read_log(site, call_type, df_folder_structure)
        
        if not df_log.empty:
            logs.append(df_log)
    
    df_logs = pd.concat(logs)

    return df_logs


def extract_samples(site, gb_wavfile, df_folder_structure, is_train):
    """Generate labelled samples for a site given all call logs."""
    
    # For .wav in groupby object
    for wavfile, logs in gb_wavfile:
        
        # Read in audio
        y, sr_default = read.read_audio(site, wavfile, df_folder_structure)
        
        # Frame and extract features
        y_features = features.extract_features(y)
        
        # Label features
        y_labelled_features = label.label_features(y_features,
                                                   logs,
                                                   sr_default)
        
        # Balance training samples and test samples (if selected)
        if (is_train == True) or (s.IS_TEST_BALANCED == True):
            y_labelled_features = balance_dataset(y_labelled_features)
        
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


def numpy_to_tensor(X, y):
    """
    Convert X, y numpy arrays into tensors compatable with pytorch.
    
    Input:
      X: (n_samples, feature_width, feature_height)
      y: (n_samples,)
      
    Return torch tensors
      X: (n_samples, n_channels, feature_width, feature_height) 
      y: (n_samples, 1) list of labels
      
    """
    
    # Torch convolution expects n_samples x n_channels x w x h
    # Add dimension of 1 to represent n_channels = 1
    X = np.expand_dims(X, axis=1)
    
    # Create torch tensor
    X = torch.from_numpy(X)
    y = torch.from_numpy(y).float().reshape(-1, 1)
    
    return X, y


