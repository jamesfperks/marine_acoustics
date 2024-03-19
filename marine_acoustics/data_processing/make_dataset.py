"""
Read the AADC dataset.

"""


from marine_acoustics.configuration import selector
from marine_acoustics.data_processing import info, sample


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
    selector.print_selection_summary(df_trainset, df_testset)
    
    # Get training samples
    train_samples = sample.get_training_samples(df_trainset,
                                                   df_folder_structure)
    
    # Get test samples
    test_samples = sample.get_test_samples(df_testset, df_folder_structure)
    
    return train_samples, test_samples

