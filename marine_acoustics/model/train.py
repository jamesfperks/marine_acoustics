"""
Train the model.

"""


import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import librosa
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from marine_acoustics.configuration import settings as s
from marine_acoustics.model import train_utils, cnn


def train_classifier():
    """Train the model used for classification.."""
    
    print('\n\n' + '-'*s.HEADER_LEN +
          f'\nTraining model {s.MODEL}-{s.FEATURES}...', end='')
    start = time.time()
    
    # Load training set
    X_train = np.load(s.SAVE_DATA_FILEPATH + 'X_train.npy')
    y_train = np.load(s.SAVE_DATA_FILEPATH + 'y_train.npy')
    
    # Select model
    if s.MODEL == 'HGB':
        model = train_grad_boost(X_train, y_train)
        
    elif s.MODEL == 'CNN':
        model = train_cnn(X_train, y_train)
    
    else:
        raise NotImplementedError('Model chosen not implemented: ', s.MODEL)
    
    end = time.time()
    print(f'100% ({end-start:.1f} s)\n' + '-'*s.HEADER_LEN)
   
    return model


def train_grad_boost(X_train, y_train):
    """Train scikit learn HistGradientBoostingClassifier."""
    
    model = HistGradientBoostingClassifier(random_state=s.SEED,
                                           validation_fraction=0.15,
                                           early_stopping=True).fit(X_train,
                                                                    y_train)
    
    dump(model, s.SAVE_MODEL_FILEPATH + s.MODEL + '-' + s.FEATURES)
    
    return model
    

def train_cnn(X_train, y_train):
    """Train CNN."""
    
    # Convert to tensor
    X_train, y_train = train_utils.numpy_to_tensor(X_train, y_train)
        
    torch.manual_seed(s.SEED)

    # Train loader accepts list(zip(X_train, y_train)
    # Train_samples can be indexed [i] to give (X[i], y[i])
    loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)),
                                        batch_size=s.BATCH_SIZE,
                                        shuffle=True)
    
    if s.BINARY == True:
        model = cnn.BinaryNet()
        loss_fn = nn.BCELoss()
        
    else:
        model = cnn.MultiNet()
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=s.LR,
                      momentum=s.MOMENTUM)
    
    #optimizer = optim.Adam(model.parameters(), lr=s.LR)
    
    # Train the model
    model.train()
    epoch_loss = []
    for epoch in range(s.N_EPOCHS):    
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
        
    # Plot training loss per epoch
    train_utils.plot_training_loss(epoch_loss)
    
    # Save model
    torch.save(model.state_dict(),
               s.SAVE_MODEL_FILEPATH + s.MODEL + '-' + s.FEATURES)

    return model
    




