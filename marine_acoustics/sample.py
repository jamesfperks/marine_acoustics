"""
Extract samples from the raw .wav files

"""


import time
import numpy as np
import pandas as pd
from marine_acoustics import read, features, label
from marine_acoustics import settings as s


def get_training_samples(df_trainset, df_folder_structure):
    """Return extracted trainings samples and print time taken."""
    
    print('\n'*2 + '-'*50 + '\nTRAINING PROGRESS\n' + '-'*50 + 
          '\n  - Extracting trainings samples...', end='')
    

    start = time.time()
    X_train, y_train = extract_samples(df_trainset, df_folder_structure)
    end = time.time()
    
    print(f'100% ({end-start:.1f} s)')
    
    return X_train, y_train


def get_test_samples(df_testset, df_folder_structure):
    """Return extracted test samples and print time taken."""
    
    print('  - Extracting test samples...', end='')
    
    start = time.time()
    X_test, y_test = extract_samples(df_testset, df_folder_structure)
    end = time.time()
    
    print(f'100% ({end-start:.1f} s)')
    
    return X_test, y_test


def extract_samples(df_data_summary, df_folder_structure):
    """Extract labelled samples from .wav files."""
    
    sample_set = []
    
    for site in df_data_summary.index:
        
        logs = []
        
        for call_type in df_data_summary.columns:
            
            df_log = read.read_log(site, call_type, df_folder_structure)
            
            if not df_log.empty:
                logs.append(df_log)
        
        # concatenate all logs into one DF
        if len(logs) == 0:
            continue
        df_logs = pd.concat(logs)

        # Groupby .wav filename
        gb_wavfile = df_logs.groupby('Begin File')
        
        
        # For .wav in groupby object
        for wavfile, logs in gb_wavfile:
            
            # Read in audio
            y, sr_default = read.read_audio(site, wavfile, df_folder_structure)
            
            # Frame and extract features
            y_features = features.extract_features(y)
            
            # Label features
            y_labelled_features = label.label_features(y_features, logs, sr_default)
            
            # add samples to sample set
            sample_set.extend(y_labelled_features)
    
    # Create sample vector
    samples = np.vstack(sample_set)
    
    # Balance and randomise samples
    balanced_samples = balance_dataset(samples)
    print('\n',balanced_samples.shape, '\n')
    
    # Split sample vector
    X, y = split_sample_vector(balanced_samples)
    
    return X, y


def balance_dataset(samples):
    """Sub-sample the majority class to balance the dataset."""
    
    # Split into classes
    whale_samples = samples[samples[:,-1] == 1]
    background_samples = samples[samples[:,-1] == 0]
    print(f'\nNumber of whale call samples: {whale_samples.shape}\nNumber of background samples: {background_samples.shape}\n')
    
    # Randomise sample order
    np.random.seed(s.SEED)
    np.random.shuffle(whale_samples)
    np.random.shuffle(background_samples)
    
    # Subsample the majority class
    n_minority = min(len(whale_samples), len(background_samples))
    balanced_whale = whale_samples[0:n_minority, :]
    balanced_background = background_samples[0:n_minority, :]
    
    # Recombine and randomise samples from each class
    balanced_samples = np.vstack((balanced_whale, balanced_background))
    np.random.shuffle(balanced_samples)
  
    return balanced_samples


def split_sample_vector(samples):
    """Split samples into vectors X, y."""
    
    X = samples[:,0:-1]
    y = samples[:,-1]
    
    return X, y

