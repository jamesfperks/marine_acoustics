"""
Feature respresentations used to create a feature vector from .wav files.

"""


import librosa
import pywt
import numpy as np

from marine_acoustics import settings as s


def extract_features(y):
    """Frame data and extract features for each frame. (FRAMES X s.FEATURES)"""
    

    if s.FEATURES == 'MFCC':
        y_features = calculate_mfccs(y)   # Calculate MFCCs
        
    elif s.FEATURES == 'STFT':
        y_features = calculate_stft(y)    # Calculate STFT
        
    elif s.FEATURES == 'MEL':
        y_features = calculate_melspectrogram(y)  # Calculate mel-spectrogram
        
    elif s.FEATURES == 'CWT':
        y_features = calculate_cwt(y)   # Calculate cwt
    
    else:
        raise NotImplementedError('Feature representation chosen ' 
                                        'is not implemented', s.FEATURES)
       
    return y_features


def calculate_mfccs(y):
    """Split data into DFT windows and compute MFCCs for each window."""
    
    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(y=y,
                                 sr=s.SR,
                                 n_mfcc=s.N_MFCC,
                                 n_fft=s.FRAME_LENGTH,
                                 hop_length=s.HOP_LENGTH,
                                 n_mels=s.N_MELS,
                                 fmin=s.FMIN,
                                 fmax=s.FMAX).T
    
    return mfccs
    

def calculate_stft(y):
    """Compute STFT and split data into frames."""
    
    # STFT of y
    D = librosa.stft(y, n_fft=s.FRAME_LENGTH, hop_length=s.HOP_LENGTH)
    
    # STFT in dB
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max).T
    
    # Select frequency bin range for STFT
    stft_fmin_idx, stft_fmax_idx = get_stft_freq_range()
    S_db = S_db[:,stft_fmin_idx:stft_fmax_idx+1]
           
    return S_db


def get_stft_freq_range():
    """Return indexes of STFT frequency bins which correspond to FMIN, FMAX"""
    
    # Calculate frequency bins
    freqs = librosa.fft_frequencies(sr=s.SR, n_fft=s.FRAME_LENGTH)
    
    # Find index of closest frequency bin to FMIN, FMAX
    stft_fmin_idx = np.argmin(np.abs(freqs-s.FMIN))
    stft_fmax_idx = np.argmin(np.abs(freqs-s.FMAX))
    
    return stft_fmin_idx, stft_fmax_idx


def calculate_melspectrogram(y):
    """Compute the mel-spectrogram and return a vector for each frame."""
    
    # mel-power spectrogram of y
    D = librosa.feature.melspectrogram(y=y,
                                       sr=s.SR,
                                       n_fft=s.FRAME_LENGTH,
                                       hop_length=s.HOP_LENGTH,
                                       n_mels=s.N_MELS,
                                       fmin=s.FMIN,
                                       fmax=s.FMAX)
    
    
    # mel-power spectrogram in dB
    S_db = librosa.power_to_db(D, ref=np.max).T
    
    return S_db


def calculate_cwt(y):
    """Compute the cwt and return a vector for each frame."""
    
    # Choose wavelet pseudo frequencies
    desired_freqs = np.arange(s.FMIN, s.FMAX+1, 1)
    scales = frequency2scale(desired_freqs)
    
    # Compute continuous wavelet transform
    wavelet_coeffs, wavelet_freqs = apply_cwt(y, scales)
    
    cwt = frame_data(wavelet_coeffs.T)
    
    cwt = np.mean(cwt, axis=1)
    
    return cwt


def frame_data(data):
    """
    Slice 1D array into frames with a given overlap: (n_frames x frame_length)
    """
    
    frame_view = librosa.util.frame(data,
                                    frame_length=s.FRAME_LENGTH,
                                    hop_length=s.HOP_LENGTH,
                                    axis=0)
    
    return frame_view


def apply_cwt(y, scales):
    """Apply cwt to a 1D array"""
    
    # Compute continuous wavelet transform
    wavelet_coeffs, wavelet_freqs = pywt.cwt(y,
                                             scales,
                                             s.WAVELET,
                                             sampling_period=1/s.SR)

    return wavelet_coeffs, wavelet_freqs
    

def scale2frequency(scales):
    """Convert from cwt scale to to pseudo-frequency"""

    # pywt function returns normalised frequency so need to multiply by sr
    freqs = pywt.scale2frequency(s.WAVELET, scales) * s.SR

    return freqs


def frequency2scale(desired_freqs):
    """Convert from desired frequencies to a cwt scale"""

    # pywt function input is normalised frequency so need to normalise by sr
    normalised_freqs = desired_freqs / s.SR
    
    freqs = pywt.scale2frequency(s.WAVELET, normalised_freqs)

    return freqs

