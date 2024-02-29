"""
Train the model.

"""


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import time
import librosa
import matplotlib.pyplot as plt
import numpy as np


from sklearn.ensemble import HistGradientBoostingClassifier
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import sample
from marine_acoustics.model.cnn import LeNet


def train_classifier(train_samples):
    """Train the model used for classification.."""
    
    print('  - Training model...', end='')
    start = time.time()
    
    if s.MODEL == 'HGBC':
        model = train_grad_boost(train_samples)
        
    elif s.MODEL == 'CNN':
        model = train_cnn(train_samples)
    
    else:
        raise NotImplementedError('Model chosen not implemented: ', s.MODEL)
    
    end = time.time()
    print(f'100% ({end-start:.1f} s)')
   
    return model


def train_grad_boost(train_samples):
    """Train scikit learn HistGradientBoostingClassifier."""
    
    X_train, y_train = sample.split_samples(train_samples)
    
    model = HistGradientBoostingClassifier(random_state=s.SEED).fit(X_train,
                                                                    y_train)
    
    return model
    

def train_cnn(train_samples):
    """Train CNN."""
    
    X_train, y_train = sample.samples_to_tensors(train_samples)
    
    torch.manual_seed(s.SEED)
    n_epochs = 10
    batch_size_train = 16
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    
    
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
    test_loss = []
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

    
    return model
    




