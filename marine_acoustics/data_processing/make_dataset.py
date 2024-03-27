"""
Read the AADC dataset.

"""


import time
import numpy as np
from marine_acoustics.configuration import selector
from marine_acoustics.data_processing import info, sample
from marine_acoustics.configuration import settings as s


def make_dataset():
    """Separate data into train and test sets and extract samples."""
    
    # Get data folder structure
    df_folder_structure = info.get_folder_structure()
    
    # Count total annotations
    df_annotations = info.get_total_annotation_count(df_folder_structure)
    
    # Select training set
    df_trainset = selector.select_training_set(df_annotations)
    
    # Select test set
    df_testset = selector.select_test_set(df_annotations)
    
    # Print training and test set summary
    selector.print_train_test_selection(df_trainset, df_testset)
    
    # Get training samples
    train_samples = get_training_samples(df_trainset, df_folder_structure)
    
    # Get test samples
    test_samples = get_test_samples(df_testset, df_folder_structure)
    
    # Save train and test sets
    save_datasets(train_samples, test_samples)
    
    return train_samples, test_samples


def get_training_samples(df_trainset, df_folder_structure):
    """Return extracted trainings samples and print time taken."""
    
    print('\n'*2 + '-'*s.HEADER_LEN + '\nTRAINING PROGRESS\n' +
          '-'*s.HEADER_LEN + '\n'*2 + '  - Extracting trainings samples...',
          end='')
    
    start = time.time()
    train_samples = sample.get_samples(df_trainset, df_folder_structure,
                                       is_train=True)
    end = time.time()
    print(f'100% ({end-start:.1f} s)\n')
    
    return train_samples


def get_test_samples(df_testset, df_folder_structure):
    """Return extracted test samples and print time taken."""
    
    print('  - Extracting test samples...', end='')
    
    start = time.time()
    test_samples = sample.get_samples(df_testset, df_folder_structure,
                                      is_train=False)
    end = time.time()
    print(f'100% ({end-start:.1f} s)\n')
    
    return test_samples


def save_datasets(train_samples, test_samples):
    """Write train and tests sets to a json file."""
    
    # Split samples into X, y
    X_train, y_train = sample.split_samples(train_samples)
    X_test, y_test = sample.split_samples(test_samples)
    
    # Save as binary file in .npy format
    np.save(s.SAVE_DATA_FILEPATH + '/X_train.npy', X_train)
    np.save(s.SAVE_DATA_FILEPATH + '/y_train.npy', y_train)
    np.save(s.SAVE_DATA_FILEPATH + '/X_test.npy', X_test)
    np.save(s.SAVE_DATA_FILEPATH + '/y_test.npy', y_test)

    
