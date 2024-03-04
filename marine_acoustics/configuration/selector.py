"""
Select the training and test sets from the AADC dataset.

"""


from marine_acoustics.configuration import settings as s


def select_training_set(df_annotations):
    """Select sites and call types to use for training."""
 
    # Convert site labels to indexes
    train_sites = [i-1 for i in s.TRAINING_SITES]
    train_call_types = [i-1 for i in s.TRAINING_CALL_TYPES]
    
    # Training set annotation summary
    df_trainset = df_annotations.iloc[train_sites, train_call_types]
    
    # Raise error if no annotations exist
    if not df_trainset.any(axis=None):
        sites = df_annotations.index[train_sites].to_list()
        calls = df_annotations.columns[train_call_types].to_list()
        raise ValueError('Chosen sites and call-types '
                          'contain zero annotations.', sites, calls)
    
    return df_trainset


def select_test_set(df_annotations):
    """Select sites and call types to use for testing."""
    
    test_sites = [i-1 for i in s.TEST_SITES]
    test_call_types = [i-1 for i in s.TEST_CALL_TYPES]
    train_sites = [i-1 for i in s.TRAINING_SITES]
    train_call_types = [i-1 for i in s.TRAINING_CALL_TYPES]
    
    # Default to using all non-training sites if unspecified []
    if len(test_sites) == 0:
        test_sites = list(range(0, 11))  
        for site_idx in train_sites:   
            test_sites.remove(site_idx)
    
    # Default to using training call types if unspecified []
    if len(test_call_types) == 0:
        test_call_types = train_call_types
            
    # Test set summary
    df_testset = df_annotations.iloc[test_sites, test_call_types]
    
    # Raise error if no annotations exist
    if not df_testset.any(axis=None):
        sites = df_annotations.index[test_sites].to_list()
        calls = df_annotations.columns[test_call_types].to_list()
        raise ValueError('Chosen sites and call-types '
                          'contain zero annotations.', sites, calls)
    
    return df_testset


def print_selection_summary(df_trainset, df_testset):
    """Print a summary of the number of whale call annotations for the
    training set and test set."""
    
    # Train/test ratio
    train_tot = df_trainset.to_numpy().sum()
    test_tot = df_testset.to_numpy().sum()
    train_percent = round(100*train_tot/(train_tot + test_tot))
    test_percent = round(100*test_tot/(train_tot + test_tot))
    
    # Train/test header
    print('\n'*2 + '-'*50 + '\nTRAINING AND TEST SET SELECTION\n' + '-'*50)
    
    # Print training set summary
    print('\n' + f'\nTraining set: ({train_tot})\n'
          + '-'*30 + f'\n{df_trainset}')
    
    # Print testset summary
    print('\n'*2 + f'\nTest set: ({test_tot})\n'
          + '-'*30 + f'\n{df_testset}')
    
    # Dataset summary header
    print('\n'*2 + '-'*50 + '\nSELECTION SUMMARY\n' + '-'*50)
    
    # Train/test ratio printout
    print('\n'*2 + '  - Percentage split train/test is ' + 
          f'{train_percent}/{test_percent}\n')
    
    # Feature extraction method
    print(f'  - Feature extraction method: {s.FEATURES}')
    
    