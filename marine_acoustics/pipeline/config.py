"""
Functions for the detection-classification pipeline
"""

import torch
from joblib import load
from marine_acoustics.configuration import settings as s
from marine_acoustics.configuration import selector
from marine_acoustics.data_processing import info, sample
from marine_acoustics.model import cnn


def get_groupby():
    # Select test set
    df_testset = selector.select_test_set()
    
    # Print test set selection
    selector.print_test_selection(df_testset)
    
    # Sample from selected sites and call types
    site = df_testset.index[0]
    call_types = df_testset.columns
    df_folder_structure = info.get_folder_structure()
    
    # Combine all call-type logs
    df_logs = sample.concat_call_logs(site, call_types, df_folder_structure)
    
    # Groupby .wav filename
    gb_wavfile = df_logs.groupby('Begin File')

    return gb_wavfile, site, df_folder_structure


def load_model():
    """Load model from logs/models/detection/"""

    if s.MODEL == 'HGB':
        model = load('logs/models/detection/' + s.MODEL + '-' + s.FEATURES)
        
    elif s.MODEL == 'CNN':
        model = cnn.BinaryNet()
        model.load_state_dict(torch.load('logs/models/detection/' +
                                         s.MODEL + '-' + s.FEATURES))

    else: raise ValueError()

    return model

