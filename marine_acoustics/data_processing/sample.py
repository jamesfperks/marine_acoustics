"""
Extract samples from the raw .wav files

"""


import time
import random
import numpy as np
import pandas as pd
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import read, features, label


def get_training_samples(df_trainset, df_folder_structure):
    """Return extracted trainings samples and print time taken."""
    
    print('\n'*2 + '-'*50 + '\nTRAINING PROGRESS\n' + '-'*50 + 
          '\n  - Extracting trainings samples...', end='')
    

    start = time.time()
    X_train, y_train = get_samples(df_trainset,
                                   df_folder_structure,
                                   is_train=True)
    end = time.time()
    
    print(f'100% ({end-start:.1f} s)')
    
    return X_train, y_train


def get_test_samples(df_testset, df_folder_structure):
    """Return extracted test samples and print time taken."""
    
    print('  - Extracting test samples...', end='')
    
    start = time.time()
    X_test, y_test = get_samples(df_testset,
                                 df_folder_structure,
                                 is_train=False)
    end = time.time()
    
    print(f'100% ({end-start:.1f} s)')
    
    return X_test, y_test


def get_samples(df_selected_dataset, df_folder_structure, is_train):
    """Extract labelled samples from .wav files."""
        
    # Sample from selected sites and call types
    sites = df_selected_dataset.index
    call_types = df_selected_dataset.columns
    sample_set = create_sample_set(sites, call_types, df_folder_structure)
    
    # Create sample vector (n_samples x n_features)
    samples = np.vstack(sample_set)
    
    # Balance training samples and test samples (if selected)
    if (is_train == True) or (s.IS_TEST_BALANCED == True):
        samples = balance_dataset(samples)
    
    # Split sample vector
    X, y = split_sample_vector(samples)
    
    return X, y


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
        
        # Append to sample set
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
    
    labels = samples[:,-1]
    
    whale_indexes = []
    background_indexes = []
    
    # Split into whale and background indexes
    for i in range(len(labels)):
        if labels[i] == 1:
            whale_indexes.append(i)
        else:
            background_indexes.append(i)
       
        
    # Randomly sub-sample indexes from background to match n_whale samples
    random.seed(s.SEED)
    sampled_background_indexes = random.sample(background_indexes,
                                               len(whale_indexes))
    
    # Recombine whale and background indexes preserving sample order
    balanced_indexes = whale_indexes + sampled_background_indexes
    balanced_indexes.sort()
    
    # Index samples using balanced indexes
    balanced_samples = samples[balanced_indexes, :]
  
    return balanced_samples


def split_sample_vector(samples):
    """Split samples into vectors X, y."""
    
    # Label y is the last element in the feature vector
    X = samples[:,0:-1]
    y = samples[:,-1]
    
    return X, y

