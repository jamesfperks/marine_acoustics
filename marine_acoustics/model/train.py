"""
Train the model.

"""


import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np

from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import sample
from marine_acoustics.model.cnn import LeNet


def train_classifier():
    """Train the model used for classification.."""
    
    print('  - Training model...', end='')
    start = time.time()
    
    # Load training set
    X_train = np.load(s.SAVE_DATA_FILEPATH + '/X_train.npy')
    y_train = np.load(s.SAVE_DATA_FILEPATH + '/y_train.npy')
    
    # Select model
    if s.MODEL == 'HGB':
        model = train_grad_boost(X_train, y_train)
        
    elif s.MODEL == 'CNN':
        model = train_cnn(X_train, y_train)
    
    else:
        raise NotImplementedError('Model chosen not implemented: ', s.MODEL)
    
    end = time.time()
    print(f'100% ({end-start:.1f} s)')
   
    return model


def train_grad_boost(X_train, y_train):
    """Train scikit learn HistGradientBoostingClassifier."""
    
    model = HistGradientBoostingClassifier(random_state=s.SEED).fit(X_train,
                                                                    y_train)
    
    dump(model, s.SAVE_MODEL_FILEPATH + '/HGB')
    
    return model
    

def train_cnn(X_train, y_train):
    """Train CNN."""
    
    # Convert to tensor
    X_train, y_train = sample.numpy_to_tensor(X_train, y_train)
    
    torch.manual_seed(s.SEED)
    n_epochs = 10
    batch_size_train = 16
    learning_rate = 0.01
    momentum = 0.5
    
    
    # Train loader accepts list(zip(X_train, y_train)
    # Train_samples can be indexed [i] to give (X[i], y[i])
    
    loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)),
                                        batch_size=batch_size_train,
                                        shuffle=True)
    
    model = LeNet()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)
    loss_fn = nn.BCELoss()
    
    # Train the model
    model.train()
    epoch_loss = []
    for epoch in range(n_epochs):    
        acc_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            acc_loss += loss.detach().numpy()
            loss.backward()
            optimizer.step()
    
        # Average batch loss over one epoch
        epoch_loss.append(acc_loss / len(loader))
        
    plt.figure()
    plt.plot(range(1, n_epochs+1), epoch_loss, color='blue')
    plt.title('Training loss per epoch')
    plt.legend(['Training Loss'], loc='upper right')
    plt.xlabel('Epoch Number')
    plt.ylabel('Training loss (Binary Cross Entropy loss)')

    
    # Save model
    torch.save(model.state_dict(), s.SAVE_MODEL_FILEPATH + '/CNN')

    return model
    




