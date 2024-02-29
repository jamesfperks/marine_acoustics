"""
Automated detection of Antarctic Blue and Fin Whale sounds using 
the Australian Antarctic Data Centre Annotated Library.

Author: James Perks
Email: jamesperks@outlook.com

"""


import time
from marine_acoustics.configuration import intro, selector
from marine_acoustics.data_processing import info, sample
from marine_acoustics.model import train, predict, evaluate


def run():
    """Executes script."""
    
    # Start script
    intro.print_introduction()
    
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
    
    # Train model
    model = train.train_classifier(train_samples)
    
    # Get predictions
    predictions = predict.get_predictions(train_samples, test_samples, model)
    
    # Evaluate model
    evaluate.get_results(train_samples, test_samples, predictions)
    

def main():
    
    # Run and time script
    start = time.time()
    run()
    end = time.time()
    
    # End of script
    print('\n'*2 + f'Total runtime: {end-start:0.1f} seconds.\n' +
          '-'*47 + 'End')
          
    
if __name__ == '__main__':
    main()
    