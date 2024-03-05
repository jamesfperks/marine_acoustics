"""
Select the training and test sets from the AADC dataset.

"""


from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import info


def select_training_set(df_annotations):
    """Select sites and call types to use for training."""
 
    train_sites = [i-1 for i in s.TRAIN_SITES]
    
    # Add any additional negative class call types
    train_call_labels = sorted(set(s.TRAIN_CALL_TYPES +
                                   s.TRAIN_NEGATIVE_CLASS))
    train_call_types = [i-1 for i in train_call_labels]
    
    # Training set annotation summary
    df_trainset = df_annotations.iloc[train_sites, train_call_types]
    
    # Raise error if no annotations exist
    no_annotations_check(df_annotations, df_trainset,
                         train_sites, train_call_types)
    
    return df_trainset


def select_test_set(df_annotations):
    """Select sites and call types to use for testing."""
    
    train_sites = [i-1 for i in s.TRAIN_SITES]
    train_call_types = [i-1 for i in s.TRAIN_CALL_TYPES]
    test_sites = [i-1 for i in s.TEST_SITES]
    test_call_types = [i-1 for i in s.TEST_CALL_TYPES]
    test_call_types_neg_class = [i-1 for i in s.TEST_NEGATIVE_CLASS]
    
    # Default to using all non-training sites if unspecified []
    if len(test_sites) == 0:
        test_sites = list(range(0, 11))  
        for site_idx in train_sites:   
            test_sites.remove(site_idx)
    
    # Default to using training call types if unspecified []
    if len(test_call_types) == 0:
        test_call_types = train_call_types
        
    
    # Add any additional negative class call types
    test_call_types = sorted(set(test_call_types + test_call_types_neg_class))
            
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
    
    
def print_selection_summary(df_trainset, df_testset):
    """Print a summary of the training set and test set."""
    
    
    print_class_selection(df_testset)
    print_train_test_selection(df_trainset, df_testset)
    
    
def print_class_selection(df_testset):
    """Print selected binary classes (positive / negative class).""" 
    
    # Header
    print('\n'*2 + '-'*s.HEADER_LEN +
          '\nBINARY CLASSIFICATION\n' +
          '-'*s.HEADER_LEN)
    
    call_types = info.get_call_types()
    
    # Training (positive / negative classes)
    train_pos_class_str = ''
    for i in s.TRAIN_CALL_TYPES:
        train_pos_class_str += call_types[i-1] + ', '
    
    if len(s.TRAIN_NEGATIVE_CLASS) > 0:
        train_neg_class_str = call_types[s.TRAIN_NEGATIVE_CLASS[0]-1]
    else:
        train_neg_class_str = 'Background'
        
        
    # Test (positive / negative classes)
    test_pos_class = df_testset.columns.to_list()
    if len(s.TEST_NEGATIVE_CLASS) > 0:
        test_pos_class.remove(call_types[s.TEST_NEGATIVE_CLASS[0]-1])
    test_pos_class_str = ', '.join(test_pos_class)
    
    if len(s.TEST_NEGATIVE_CLASS) > 0:
        test_neg_class_str = call_types[s.TEST_NEGATIVE_CLASS[0]-1]
    else:
        test_neg_class_str = 'Background'
    

    print('\n'+f'  Train: {train_pos_class_str[:-2]} / {train_neg_class_str}')
    print('\n'+f'  Test: {test_pos_class_str} / {test_neg_class_str}',
          end='')
    

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


    