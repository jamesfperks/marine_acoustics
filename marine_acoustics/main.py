"""
Automated detection of Antarctic Blue and Fin Whale sounds using 
the Australian Antarctic Data Centre Annotated Library.

Author: James Perks
Email: jamesperks@outlook.com

"""


import time
from marine_acoustics import intro, info, select, sample, train, evaluate


def run():
    """Executes script."""
    
    # Start script
    intro.print_introduction()
    
    # Get data folder structure
    df_folder_structure = info.get_folder_structure()
    
    # Count total annotations
    df_annotations = info.get_total_annotation_count(df_folder_structure)
    
    # Select training set
    df_trainset = select.select_training_set(df_annotations)
    
    # Select test set
    df_testset = select.select_test_set(df_annotations)
    
    # Print training and test set summary
    select.print_selection_summary(df_trainset, df_testset)
    
    # Get training samples
    X_train, y_train = sample.get_training_samples(df_trainset, df_folder_structure)
    
    # Get test samples
    X_test, y_test = sample.get_test_samples(df_testset, df_folder_structure)
    
    # Train model
    clf = train.train_classifier(X_train, y_train)
    
    # Print results
    evaluate.get_results(clf, X_train, y_train, X_test, y_test)
    

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
    