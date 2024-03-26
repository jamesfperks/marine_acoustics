"""
Frame audio and extract features.

"""


import librosa
import numpy as np

from marine_acoustics.configuration import settings as s


def get_stft_freq_indexes(n_fft):
    """Return indexes of STFT frequency bins which correspond to FMIN, FMAX"""
    
    # Calculate frequency bins
    freqs = librosa.fft_frequencies(sr=s.SR, n_fft=n_fft)
    
    # Find index of closest frequency bin to FMIN, FMAX
    fmin_idx = np.argmin(np.abs(freqs-s.FMIN))
    fmax_idx = np.argmin(np.abs(freqs-s.FMAX))
    
    return fmin_idx, fmax_idx


def apply_stft(y):
    """
    Compute STFT. Select freq max/min limits.
    Returns n_windows x n_freq_bins
    """
    
    # STFT of y
    D = librosa.stft(y,
                     n_fft=s.N_FFT,
                     hop_length=s.STFT_HOP,
                     win_length=s.STFT_LEN,
                     center=True)
  
    # STFT in dB
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max).T
    
    # Select frequency bin range for STFT
    stft_fmin_idx, stft_fmax_idx = get_stft_freq_indexes(s.N_FFT)
    S_db = S_db[:,stft_fmin_idx:stft_fmax_idx+1]
    
    return S_db

