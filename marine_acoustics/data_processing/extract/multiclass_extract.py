"""
Extract samples for multiclass classification from .wav files.
"""


import random
import matplotlib.pyplot as plt
import numpy as np
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import read, label
from marine_acoustics.data_processing.features import multiclass_features


def extract_samples(site, gb_wavfile, df_folder_structure, is_train):
    """Generate labelled samples for a site given all call logs."""
    
    X_site = []
    y_site = []
    
    # For .wav in groupby object
    for wavfile, logs in gb_wavfile:
        
        # Read in audio
        y, sr_default = read.read_audio(site, wavfile, df_folder_structure)
        
        # Call start and end indexes
        call_indexes = label.get_call_indexes(logs, sr_default)
        
        # Call labels
        labels = label.get_multi_class_labels(logs)
        
        # For call in logs:
        for i in range(len(labels)):
            
            # get call start-end index
            i_start, i_end = call_indexes[i]
            
            # calculate call centre index ic
            ic = round((i_start+i_end)/2)
 
            # calculate start and end index of the framed call
            half_frame = s.FRAME_LENGTH//2
            i_frame_start = ic - half_frame
            i_frame_end = ic + half_frame
            
            # skip if the the frame falls outside of audio range
            if (i_frame_start < 0) or (i_frame_end >= len(y)):
                continue
            
            else:
                # slice y to get a frame centred on ic
                y_frame = y[i_frame_start:i_frame_end+1]
                
                # calculate frame features and add to X
                y_frame_features = multiclass_features.extract_features(y_frame)
                X_site.append(y_frame_features)
                
                # get frame label and add to y 
                y_site.append(labels[i])
    

    #for i in range(1):
        # Plot sample features
     #   plt.figure(figsize=(5,5))
      #  plt.axis('off')
       # plt.pcolormesh(X_site[58].T, cmap='magma')
        #fig1 = plt.gcf()
        #fig1.savefig(r'C:\Users\james\OneDrive - Nexus365\Engineering\Year4\4YP\4YP Report\Figures/label-Z.png', format='png')
        
    #raise NotImplementedError()
    
    #if is_train == True:
     #   X_site, y_site = balance_AB(X_site, y_site)
    # balance X, y to get even number of A,B,Z from each site
    #if (is_train == True) or (s.IS_TEST_BALANCED == True):
    #    X_site, y_site = balance_data(X_site, y_site) 
    
    # write X_site, y_site to 'temp/train-data/' or 'temp/test-data/'
    write_samples_to_temp_folder(X_site, y_site, site, is_train)
     

def write_samples_to_temp_folder(X_site, y_site, site, is_train):
    
    if is_train:
        temp_data_fp = s.SAVE_DATA_FILEPATH + 'temp/train-data/'
    else:
        temp_data_fp = s.SAVE_DATA_FILEPATH + 'temp/test-data/'

    X_data_fp = temp_data_fp + 'X/' + site + '-X.npy'
    y_data_fp = temp_data_fp + 'y/' + site + '-y.npy' 
    np.save(X_data_fp, X_site)
    np.save(y_data_fp, y_site)



def balance_AB(X, y):
    """Balance the classes by randomly subsampling majority classes.
    Only implemented for A, B, Z calls in multiclass classification."""
    
    zero_indexes = []
    one_indexes = []
    two_indexes = []
    
    # Find sample indexes for positive and negative class
    for i in range(len(y)):
        if y[i] == 0:
            zero_indexes.append(i)
        
        elif y[i] == 1:
            one_indexes.append(i)
            
        else:
            two_indexes.append(i)
    
    print(len(zero_indexes), len(one_indexes), len(two_indexes))
    
    # Randomly sub-sample indexes from A to match B
    random.seed(s.SEED)
    sampled_zero_indexes = random.sample(zero_indexes, len(one_indexes))
    
    # Recombine sampled indexes preserving sample order
    balanced_indexes = sampled_zero_indexes + one_indexes + two_indexes
    balanced_indexes.sort()
    
    print(len(sampled_zero_indexes), len(one_indexes), len(two_indexes))
    
    # Index samples using balanced indexes
    X = [X[i] for i in balanced_indexes]
    y = [y[i] for i in balanced_indexes]
    
    return X, y



def balance_data(X, y):
    """Balance the classes by randomly subsampling majority classes.
    Only implemented for A, B, Z calls in multiclass classification."""
    
    zero_indexes = []
    one_indexes = []
    two_indexes = []
    
    # Find sample indexes for positive and negative class
    for i in range(len(y)):
        if y[i] == 0:
            zero_indexes.append(i)
        
        elif y[i] == 1:
            one_indexes.append(i)
            
        else:
            two_indexes.append(i)
    
    
    # Randomly sub-sample indexes from A and B calls to match Z
    random.seed(s.SEED)
    sampled_zero_indexes = random.sample(zero_indexes, len(two_indexes))
    sampled_one_indexes = random.sample(one_indexes, len(two_indexes))
    
    # Recombine sampled indexes preserving sample order
    balanced_indexes = sampled_zero_indexes + sampled_one_indexes + two_indexes
    balanced_indexes.sort()
    
    # Index samples using balanced indexes
    X = [X[i] for i in balanced_indexes]
    y = [y[i] for i in balanced_indexes]
    
    return X, y
