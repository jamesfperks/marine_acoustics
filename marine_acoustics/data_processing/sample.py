"""
Collect samples from the selected train and test sites.

"""


import os
import numpy as np
import pandas as pd
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import info, read
from marine_acoustics.data_processing.extract import (binary_extract,
                                                      multiclass_extract)


def get_samples(df_selected_dataset, is_train):
    """
    Get sample set from selected sites and call types.
    Returns sample features as X, labels as y
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
        
        # Generate samples from site and write to temp data folder
        if s.BINARY == True:
            binary_extract.extract_samples(site, gb_wavfile,
                                           df_folder_structure, is_train)
        else:
            multiclass_extract.extract_samples(site, gb_wavfile,
                                               df_folder_structure, is_train)
    
    X, y = combine_samples(is_train)
    
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
    

def combine_samples(is_train):
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

