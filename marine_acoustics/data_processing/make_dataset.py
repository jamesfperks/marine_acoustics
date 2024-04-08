"""
Read the AADC dataset.

"""


import time
import numpy as np
from marine_acoustics.configuration import selector
from marine_acoustics.data_processing import sample
from marine_acoustics.configuration import settings as s


def get_train_samples():
    """Return extracted trainings samples and print time taken."""
    
    # Select training set
    df_trainset = selector.select_training_set()
    
    # Print training set selection
    selector.print_train_selection(df_trainset)
    
    # Get training samples
    print('\n'*2 + '  - Extracting trainings samples...', end='')
    start = time.time()
    X_train, y_train = sample.get_samples(df_trainset, is_train=True)
    
    # Save as binary file in .npy format
    np.save(s.SAVE_DATA_FILEPATH + 'X_train.npy', X_train)
    np.save(s.SAVE_DATA_FILEPATH + 'y_train.npy', y_train)
    
    end = time.time()
    print(f'100% ({end-start:.1f} s)\n')


def get_test_samples():
    """Return extracted test samples and print time taken."""
    
    
    # Select test set
    df_testset = selector.select_test_set()
    
    # Print test set selection
    selector.print_test_selection(df_testset)
    
    print('\n'*2 + '  - Extracting test samples...', end='')
    
    start = time.time()
    X_test, y_test = sample.get_samples(df_testset, is_train=False)
        
    # Save as binary file in .npy format
    np.save(s.SAVE_DATA_FILEPATH + 'X_test.npy', X_test)
    np.save(s.SAVE_DATA_FILEPATH + 'y_test.npy', y_test)
    
    end = time.time()
    print(f'100% ({end-start:.1f} s)\n')
    
