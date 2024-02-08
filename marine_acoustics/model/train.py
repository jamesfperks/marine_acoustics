"""
Train the model.

"""

import time
from sklearn.ensemble import HistGradientBoostingClassifier
from marine_acoustics.configuration import settings as s


def train_classifier(X_train, y_train):
    """Train classifier."""
    
    print('  - Training model...', end='')
    
    start = time.time()
    clf = HistGradientBoostingClassifier(random_state=s.SEED).fit(X_train,
                                                                  y_train)
    end = time.time()
    
    print(f'100% ({end-start:.1f} s)')
   
    return clf

