"""
Extract samples from the raw .wav files

"""


import time
import random
import torch
import numpy as np
import pandas as pd
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import read, features, label


def get_training_samples(df_trainset, df_folder_structure):
    """Return extracted trainings samples and print time taken."""
    
    print('\n'*2 + '-'*s.HEADER_LEN + '\nTRAINING PROGRESS\n' +
          '-'*s.HEADER_LEN + '\n'*2 + '  - Extracting trainings samples...',
          end='')
    
    start = time.time()
    train_samples = get_samples(df_trainset,
                                df_folder_structure,
                                is_train=True)
    end = time.time()
    print(f'100% ({end-start:.1f} s)\n')
    
    return train_samples


def get_test_samples(df_testset, df_folder_structure):
    """Return extracted test samples and print time taken."""
    
    print('  - Extracting test samples...', end='')
    
    start = time.time()
    test_samples = get_samples(df_testset,
                                 df_folder_structure,
                                 is_train=False)
    end = time.time()
    print(f'100% ({end-start:.1f} s)\n')
    
    return test_samples


def get_samples(df_selected_dataset, df_folder_structure, is_train):
    """
    Get sample set from selected sites and call types.
    Returns samples as a list of tuples [(X1, y1), (X2, y2), ...]
    """
        
    # Sample from selected sites and call types
    sites = df_selected_dataset.index
    call_types = df_selected_dataset.columns
    sample_set = create_sample_set(sites, call_types, df_folder_structure)
    
    # Balance training samples and test samples (if selected)
    if (is_train == True) or (s.IS_TEST_BALANCED == True):
        sample_set = balance_dataset(sample_set)
    
    return sample_set


def create_sample_set(sites, call_types, df_folder_structure):
    """Return a list of samples from given sites and call-types"""
    
    sample_set = []
    
    for site in sites:
        
        # Combine all call-type logs
        df_logs = concat_call_logs(site, call_types, df_folder_structure)

        # Groupby .wav filename
        gb_wavfile = df_logs.groupby('Begin File')
        
        # Generate labelled samples from site
        site_samples = extract_samples(site, gb_wavfile, df_folder_structure)
        
        # Add to sample set
        sample_set.extend(site_samples)
    
    return sample_set


def concat_call_logs(site, call_types, df_folder_structure):
    """Return a df of all call logs for a given site and list of call types."""
    
    logs = []
    
    for call_type in call_types:
        df_log = read.read_log(site, call_type, df_folder_structure)
        
        if not df_log.empty:
            logs.append(df_log)
    
    df_logs = pd.concat(logs)
    
    return df_logs


def extract_samples(site, gb_wavfile, df_folder_structure):
    """Generate labelled samples for a site given all call logs."""
    
    site_samples = []
    
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
        
        # Add samples to site samples
        site_samples.extend(y_labelled_features)

    return site_samples 


def balance_dataset(samples):
    """Sub-sample the majority class to balance the dataset."""
    
    whale_indexes = []
    background_indexes = []
    
    # Find sample indexes for whale and background
    for i in range(len(samples)):
        if samples[i][1] == 1:
            whale_indexes.append(i)
        else:
            background_indexes.append(i)
        
    if len(background_indexes) < len(whale_indexes):
        raise ValueError('Selected call type is the majority class.')
        
    # Randomly sub-sample indexes from background to match n_whale samples
    random.seed(s.SEED)
    sampled_background_indexes = random.sample(background_indexes,
                                               len(whale_indexes))
    
    # Recombine whale and background indexes preserving sample order
    balanced_indexes = whale_indexes + sampled_background_indexes
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
    
    X, y = [], []
    for sample in samples:
        X.append(sample[0])
        y.append(sample[1])
    
    return np.asarray(X), np.asarray(y)


def samples_to_tensors(samples):
    """
    Split a list of sample tuples [(X1, y1), (X2, y2), ...] into X, y.
    
    Convert X, y into tensors compatable with pytorch.
    
    Return torch tensors
      X: (n_samples, n_channels, feature_width, feature_height) 
      y: (n_samples, 1) list of labels
      
    """
    
    X, y = split_samples(samples)
    
    # Torch convolution expects n_samples x n_channels x w x h
    # Add dimension of 1 to represent n_channels = 1
    X = np.expand_dims(X, axis=1)
    
    # Create torch tensor
    X = torch.from_numpy(X)
    y = torch.from_numpy(y).float().reshape(-1, 1)
    
    return X, y


