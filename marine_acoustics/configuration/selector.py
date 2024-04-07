"""
Select the training and test sets from the AADC dataset.

"""


from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import info


def select_training_set():
    """Select sites and call types to use for training."""
 
    train_sites = [i-1 for i in s.TRAIN_SITES]
    train_call_types = [i-1 for i in s.TRAIN_CALL_TYPES]
    
    # Get total annotation summary
    df_folder_structure = info.get_folder_structure()
    df_annotations = info.get_total_annotation_count(df_folder_structure)
    
    # Training set annotation summary
    df_trainset = df_annotations.iloc[train_sites, train_call_types]
    
    # Raise error if no annotations exist
    no_annotations_check(df_annotations, df_trainset,
                         train_sites, train_call_types)
    
    return df_trainset


def print_train_selection(df_trainset):
    """Print a summary of the number of whale call annotations for the
    training set."""
    
    # Annotation count
    train_tot = df_trainset.to_numpy().sum()
    
    # Train/test header
    print('\n'*2 +
          f'\n Training set selection ({train_tot} annotations)\n' +
          '-'*s.HEADER_LEN)
    
    # Print training set summary
    print(f'\n{df_trainset}\n\n' + '-'*s.HEADER_LEN, end='')



def select_test_set():
    """Select sites and call types to use for testing."""
    
    test_sites = [i-1 for i in s.TEST_SITES]
    train_call_types = [i-1 for i in s.TRAIN_CALL_TYPES]
    test_call_types = [i-1 for i in s.TEST_CALL_TYPES]

    # Default to using training call types if unspecified []
    if len(test_call_types) == 0:
        test_call_types = train_call_types
        
    # Get total annotation summary
    df_folder_structure = info.get_folder_structure()
    df_annotations = info.get_total_annotation_count(df_folder_structure)
                
    # Test set summary
    df_testset = df_annotations.iloc[test_sites, test_call_types]
    
    # Raise error if no annotations exist
    no_annotations_check(df_annotations, df_testset,
                         test_sites, test_call_types)
    
    return df_testset


def print_test_selection(df_testset):
    """Print a summary of the number of whale call annotations for the
    test set."""
    
    # Annotation count
    test_tot = df_testset.to_numpy().sum()
    
    # Train/test header
    print('\n'*2 +
          f'\n Test set selection ({test_tot} annotations)\n' +
          '-'*s.HEADER_LEN)
    
    # Print training set summary
    print(f'\n{df_testset}\n\n' + '-'*s.HEADER_LEN, end='')


def no_annotations_check(df_annotations, df_dataset, sites, call_types):
    """
    Raise error if given site and call type indexes contain no annotations.
    """
    
    if not df_dataset.any(axis=None):
        sites = df_annotations.index[sites].to_list()
        calls = df_annotations.columns[call_types].to_list()
        raise ValueError('Chosen sites and call-types '
                          'contain zero annotations.', sites, calls)
      



    