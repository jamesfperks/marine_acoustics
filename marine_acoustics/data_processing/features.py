"""
Frame audio and extract features.

"""


import librosa
import numpy as np

from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import feature_utils as utils


def extract_features(y):
    """Frame data and extract features for each frame.
    
    Returns:
        (n_frames X n_features...)
    """
    
    # 1D Features    
    if s.FEATURES == 'DFT':
        y_features = calculate_dft(y)    # Calculate DFT per frame
        
    elif s.FEATURES == 'MEL':
        y_features = calculate_mel_scale_dft(y)  # Calculate mel-scaled DFT
        
    elif s.FEATURES == 'MFCC':
        y_features = calculate_mfccs(y)   # Calculate MFCCs per frame
        
    elif s.FEATURES == 'CWT_AVG':
        y_features = calculate_cwt_avg(y)   # Calculate CWT average per frame
      
    # 2D Features
    elif s.FEATURES == 'STFT':
        y_features = calculate_stft_frames(y)    # Calculate STFT frames
        
    elif s.FEATURES == 'CWT':
        y_features = calculate_cwt(y)    # Calculate STFT frames
        
    else:
        raise NotImplementedError('Feature representation chosen ' 
                                        'is not implemented', s.FEATURES)
       
    return y_features


def calculate_dft(y):
    """Frame data and compute the DFT for each frame."""
    
    # Frame and compute DFT for each frame
    dft = librosa.stft(y, n_fft=s.FRAME_LENGTH, hop_length=s.HOP_LENGTH,
                       win_length=s.FRAME_LENGTH, center=False)
    

    # Cnovert to dB and transpose to match (n_frames x n_features)
    dft = librosa.amplitude_to_db(np.abs(dft), ref=np.max).T
    
    # Select frequency bin range for STFT
    fmin_idx, fmax_idx = utils.get_stft_freq_indexes(n_fft=s.FRAME_LENGTH)
    dft = dft[:, fmin_idx:fmax_idx+1]
    
    return dft


def calculate_mel_scale_dft(y):
    """Compute the mel-scaled DFT for each frame."""
    
    # mel-power spectrogram of y
    D = librosa.feature.melspectrogram(y=y,
                                       sr=s.SR,
                                       n_fft=s.FRAME_LENGTH,
                                       hop_length=s.HOP_LENGTH,
                                       n_mels=s.N_MELS,
                                       center=False,
                                       fmin=s.FMIN,
                                       fmax=s.FMAX)
    
    
    # mel-power spectrogram in dB
    S_db = librosa.power_to_db(D, ref=np.max).T
    
    return S_db


def calculate_mfccs(y):
    """Frame data and compute MFCCs for each frame."""
    
    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(y=y,
                                 sr=s.SR,
                                 n_mfcc=s.N_MFCC,
                                 n_fft=s.FRAME_LENGTH,
                                 hop_length=s.HOP_LENGTH,
                                 n_mels=s.N_MELS,
                                 center=False,
                                 fmin=s.FMIN,
                                 fmax=s.FMAX).T
    
    return mfccs
  

def calculate_cwt_avg(y):
    """Return the avg. coeffs for each cwt frame."""
    
    cwt_frames = utils.calculate_cwt_frames(y)   
    cwt_avg = np.mean(cwt_frames, axis=1)
    
    return cwt_avg


def calculate_cwt(y):
    """Return smoothed cwt coeffs to match CNN size for each frame."""
    
    # (n_frames x n_coefs_per_frame x n_freq_bins)
    cwt_frames = utils.calculate_cwt_frames(y)
    
    # Reshape to (n_frames x n_cnn_width x n_coeffs_to_avg x n_freq_bins)
    cnn_width = 30
    n_coeffs_to_avg = cwt_frames.shape[1]//cnn_width
    cwt_reshape = librosa.util.frame(cwt_frames,
                                  frame_length=n_coeffs_to_avg,
                                  hop_length=n_coeffs_to_avg, axis=1)

    # Average adjacent coeffs to get (n_frames x n_cnn_width x n_freq_bins)
    cwt = np.mean(cwt_reshape, axis=2)
    
    return cwt


def calculate_stft_frames(y):
    """Frame audio and calculate STFT for each frame."""
      
    # Frame audio (n_frames x frame_len)
    y_framed = librosa.util.frame(y,
                                  frame_length=s.FRAME_LENGTH,
                                  hop_length=s.HOP_LENGTH, axis=0)
    
    # STFT of each frame. STFT freq bins are truncated between FMIN, FMAX
    # Returns (n_frames x n_windows x n_freq_bins)
    stft_frames = np.apply_along_axis(utils.apply_stft,
                                      axis=1,
                                      arr=y_framed)
    
    return stft_frames

