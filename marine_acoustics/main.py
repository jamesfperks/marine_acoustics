"""
Automated detection of Antarctic Blue and Fin Whale sounds using 
the Australian Antarctic Data Centre Annotated Library.

Author: James Perks
Email: jamesperks@outlook.com

"""


import time
from marine_acoustics.configuration import intro
from marine_acoustics.data_processing import make_dataset
from marine_acoustics.model import train, predict, evaluate


def run():
    """Executes script."""
    
    # Start script
    intro.print_introduction()
    
    # Extract and save train and test sets
    train_samples, test_samples = make_dataset.make_dataset()
    
    # Train model
    model = train.train_classifier()
    
    # Get predictions
    predictions = predict.get_predictions(model)
    
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
    