# Test the AADC dataset

import pytest
import pandas as pd
from marine_acoustics.data_processing import info, sample


# Invoke pytest
pytest.main()


@pytest.fixture
def df_folder_structure():
    """Return folder structure"""
    
    df_folder_structure = info.get_folder_structure()
    
    return df_folder_structure


@pytest.fixture
def df_total_annotations(df_folder_structure):
    """Return df of all annotations"""
    
    # Count total annotations
    df_annotations = info.get_total_annotation_count(df_folder_structure)
    
    return df_annotations


@pytest.fixture
def all_audio_files(df_total_annotations, df_folder_structure):
    """Return groupby with all .wav files and logs for all calls."""
    
    # Sample from selected sites and call types
    all_sites = df_total_annotations.index
    all_call_types = df_total_annotations.columns
    
    
    list_all_logs = []
    
    for site in all_sites:
        
        # Combine all call-type logs
        df_logs = sample.concat_call_logs(site,
                                          all_call_types,
                                          df_folder_structure)
        list_all_logs.append(df_logs)


    # Groupby .wav filename
    df_all_logs = pd.concat(list_all_logs)
    gb_wavfile = df_all_logs.groupby('Begin File')
    
    return gb_wavfile


def test_all_audio(all_audio_files):
    """Pass if all .wav files contain more background samples than whale."""
    
    gb_wavfile = all_audio_files
    print(len(gb_wavfile['Begin File']))
    
    assert 1==2

def test_subtraction():
    assert 5 - 3 == 2
    
    