"""
Select the training and test sets from the AADC dataset.

"""


from marine_acoustics.configuration import settings as s


def select_training_set(df_annotations):
    """Select sites and call types to use for training."""
 
    train_sites = [i-1 for i in s.TRAIN_SITES]
    train_call_types = [i-1 for i in s.TRAIN_CALL_TYPES]
    
    # Training set annotation summary
    df_trainset = df_annotations.iloc[train_sites, train_call_types]
    
    # Raise error if no annotations exist
    no_annotations_check(df_annotations, df_trainset,
                         train_sites, train_call_types)
    
    return df_trainset


def select_test_set(df_annotations):
    """Select sites and call types to use for testing."""
    
    test_sites = [i-1 for i in s.TEST_SITES]
    train_call_types = [i-1 for i in s.TRAIN_CALL_TYPES]
    test_call_types = [i-1 for i in s.TEST_CALL_TYPES]

    # Default to using training call types if unspecified []
    if len(test_call_types) == 0:
        test_call_types = train_call_types
                
    # Test set summary
    df_testset = df_annotations.iloc[test_sites, test_call_types]
    
    # Raise error if no annotations exist
    no_annotations_check(df_annotations, df_testset,
                         test_sites, test_call_types)
    
    return df_testset


def no_annotations_check(df_annotations, df_dataset, sites, call_types):
    """
    Raise error if given site and call type indexes contain no annotations.
    """
    
    if not df_dataset.any(axis=None):
        sites = df_annotations.index[sites].to_list()
        calls = df_annotations.columns[call_types].to_list()
        raise ValueError('Chosen sites and call-types '
                          'contain zero annotations.', sites, calls)
      

def print_train_test_selection(df_trainset, df_testset):
    """Print a summary of the number of whale call annotations for the
    training set and test set."""
    
    # Train/test header
    print('\n'*2 + '-'*s.HEADER_LEN +
          '\nTRAIN AND TEST SET SELECTION\n' +
          '-'*s.HEADER_LEN)
    
    # Annotation count
    train_tot = df_trainset.to_numpy().sum()
    test_tot = df_testset.to_numpy().sum()
    #train_percent = round(100*train_tot/(train_tot + test_tot))
    #test_percent = round(100*test_tot/(train_tot + test_tot))
    
    # Print training set summary
    print(f'\nTraining set: ({train_tot})\n'
          + '-'*30 + f'\n{df_trainset}', end='')
    
    # Print testset summary
    print('\n'*2 + f'\nTest set: ({test_tot})\n'
          + '-'*30 + f'\n{df_testset}', end='')
    
    # Train/test ratio printout
    #print('\n' + '  - Percentage split train/test is ' + 
    #      f'{train_percent}/{test_percent}')


    