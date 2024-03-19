"""
Feature respresentations used to create a feature vector from .wav files.

"""


import librosa
import pywt
import opensmile
import numpy as np

from marine_acoustics.configuration import settings as s


def extract_features(y):
    """Frame data and extract features for each frame. (FRAMES X FEATURES)"""
    

    if s.FEATURES == 'MFCC':
        y_features = calculate_mfccs(y)   # Calculate MFCCs
        
    elif s.FEATURES == 'STFT':
        y_features = calculate_stft(y)    # Calculate DFT per frame
        
    elif s.FEATURES == 'STFT_FRAME':
        y_features = calculate_stft_frames(y)    # Calculate STFT frames
        
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
                                 center=False,
                                 fmin=s.FMIN,
                                 fmax=s.FMAX).T
    
    return mfccs
    

def calculate_stft(y, n_fft=s.FRAME_LENGTH, hop_length=s.HOP_LENGTH,
                   win_length=s.FRAME_LENGTH, center=False):
    """
    Compute STFT. Select freq max/min limits.
    Returns n_windows x n_freq_bins
    """
    
    # STFT of y
    D = librosa.stft(y,
                     n_fft=n_fft,
                     hop_length=hop_length,
                     win_length=win_length,
                     center=center)
    
    # STFT in dB
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max).T
    
    # Select frequency bin range for STFT
    stft_fmin_idx, stft_fmax_idx = get_stft_freq_range(n_fft)
    S_db = S_db[:,stft_fmin_idx:stft_fmax_idx+1]
    
    return S_db


def calculate_stft_frames(y):
    """Frame audio and calculate STFT for each frame."""
      
    # Frame audio (n_frames x frame_len)
    y_framed = librosa.util.frame(y,
                                  frame_length=s.FRAME_LENGTH,
                                  hop_length=s.HOP_LENGTH, axis=0)
    
    # STFT of each frame. STFT freq bins are truncated between FMIN, FMAX
    # Returns (n_frames x n_windows x n_freq_bins)
    stft_frames = np.apply_along_axis(calculate_stft,
                                      axis=1,
                                      arr=y_framed,
                                      n_fft=s.N_FFT,
                                      hop_length=s.STFT_HOP,
                                      win_length=s.STFT_LEN, 
                                      center=True)
    
    return stft_frames


def get_stft_freq_range(n_fft):
    """Return indexes of STFT frequency bins which correspond to FMIN, FMAX"""
    
    # Calculate frequency bins
    freqs = librosa.fft_frequencies(sr=s.SR, n_fft=n_fft)
    
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
                                       center=False,
                                       fmin=s.FMIN,
                                       fmax=s.FMAX)
    
    
    # mel-power spectrogram in dB
    S_db = librosa.power_to_db(D, ref=np.max).T
    
    return S_db


def calculate_cwt(y):
    """Compute the cwt and return a vector for each frame."""
    
    # Choose wavelet pseudo frequencies
    desired_freqs = np.arange(s.FMIN, s.FMAX+1, s.CWT_FREQ_RES)
    scales = frequency2scale(desired_freqs)
    
    # Compute continuous wavelet transform
    wavelet_coeffs, wavelet_freqs = apply_cwt(y, scales)
    cwt = librosa.amplitude_to_db(np.abs(wavelet_coeffs), ref=np.max)
      
    # Split coefficients into frames along sample axis
    cwt_framed = librosa.util.frame(cwt,
                                    frame_length=s.FRAME_LENGTH,
                                    hop_length=s.HOP_LENGTH, axis=1)
    
    # Average coeffs for each frame
    cwt_avg = np.mean(cwt_framed, axis=2)
 
    
    return cwt_avg.T     # Transpose to match (n_frames x n_features)


def frequency2scale(desired_freqs):
    """Convert from desired frequencies to a cwt scale"""

    # pywt function input is normalised frequency so need to normalise by sr
    normalised_freqs = desired_freqs / s.SR
    
    freqs = pywt.scale2frequency(s.WAVELET, normalised_freqs)

    return freqs


def apply_cwt(y, scales):
    """Apply cwt to a 1D array"""
    
    # Compute continuous wavelet transform
    wavelet_coeffs, wavelet_freqs = pywt.cwt(y,
                                             scales,
                                             s.WAVELET,
                                             sampling_period=1/s.SR)

    return wavelet_coeffs, wavelet_freqs

