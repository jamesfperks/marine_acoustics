"""
Read the AADC dataset.

"""


import librosa
import pandas as pd
from marine_acoustics import settings as s


def get_log_filepath(site, call_type, df_folder_structure):
    """Return filepath to a log file given a site name and call type."""
    
    log_header = call_type_2_log_header(call_type)
    rel_filepath = df_folder_structure.loc[site, ['Folder', log_header]].str.cat()
    log_filepath = s.DATA_FILEPATH + '/' + rel_filepath
    
    return log_filepath


def call_type_2_log_header(call_type):
    """Convert call-type string to the RavenFile name in the log headings."""
    
    call_2_log = {'Bm-A': 'Abw-A_RavenFile',
                  'Bm-B': 'Abw-B_RavenFile',
                  'Bm-Z': 'Abw-Z_RavenFile',
                  'Bm-D': 'BmD_RavenFile',
                  'Bp-20': 'Bp20_RavenFile',
                  'Bp-20+': 'Bp20Plus_RavenFile',
                  'Bp-Downsweep': 'BpDownsweep_RavenFile',
                  'Unidentified': 'UnidentifiedCalls_RavenFile'
                  }
    
    return call_2_log[call_type]


def read_log(site, call_type, df_folder_structure):
    """Read log .txt file into a dataframe."""
    
    log_filepath = get_log_filepath(site, call_type, df_folder_structure)
    
    
    fields = ['Begin File', 'End File','Begin Time (s)', 'End Time (s)',
              'Beg File Samp (samples)', 'End File Samp (samples)',
              'Begin Date Time', 'Delta Time (s)', 'Low Freq (Hz)',
              'High Freq (Hz)', 'Dur 90% (s)', 'Freq 5% (Hz)', 'Freq 95% (Hz)']
    
    df_log = pd.read_csv(log_filepath, sep='\t', usecols=fields)
    
    return df_log


def read_audio(site, wav_filename, df_folder_structure):
    """Read, resample and normalise the given .wav file. Return the resampled
    audio along with the default sample rate."""
    
    # Truncated redundant .wav filenames to match the G64 2015 folder
    if site == 'G64 2015':
        wav_filename = wav_filename[:-21] + '.wav'
    
    
    # File path to .wav file
    site_folder = df_folder_structure.loc[site, 'Folder'][:-1]
    wav_filepath = s.DATA_FILEPATH + '/' + site_folder + '/wav/' + wav_filename
    
    # Read entire mono .wav file and resample to preset global sample rate
    y, sr = librosa.load(wav_filepath, sr=s.SR)
    
    # Normalise to [-1, 1]
    y = librosa.util.normalize(y)
    
    # Store the default sample rate
    sr_default = librosa.get_samplerate(wav_filepath)
    
    return y, sr_default

