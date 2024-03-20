"""
Frame audio and extract features.

"""


import librosa
import pywt
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


def calculate_cwt_frames(y):
    """Compute the cwt and return a vector for each frame."""
    
    # Choose wavelet pseudo frequencies
    desired_freqs = np.arange(s.FMIN, s.FMAX+1, s.CWT_FREQ_RES)
    scales = frequency2scale(desired_freqs)
    
    # Compute continuous wavelet transform (n_samples x n_freq_bins)
    wavelet_coeffs, wavelet_freqs = apply_cwt(y, scales)
    cwt = librosa.amplitude_to_db(np.abs(wavelet_coeffs), ref=np.max).T
    
    # Split coefficients into frames along sample axis
    # (n_frames x n_coefs_per_frame x n_freq_bins)
    cwt_frames = librosa.util.frame(cwt,
                                    frame_length=s.FRAME_LENGTH,
                                    hop_length=s.HOP_LENGTH, axis=0)

    return cwt_frames


def frequency2scale(desired_freqs):
    """Convert from desired frequencies to a cwt scale"""

    # pywt function input is normalised frequency so need to normalise by sr
    normalised_freqs = desired_freqs / s.SR
    
    scales = pywt.frequency2scale(s.WAVELET, normalised_freqs)

    return scales


def apply_cwt(y, scales):
    """Apply cwt to a 1D array"""
    
    # Compute continuous wavelet transform
    wavelet_coeffs, wavelet_freqs = pywt.cwt(y,
                                             scales,
                                             s.WAVELET,
                                             sampling_period=1/s.SR)

    return wavelet_coeffs, wavelet_freqs


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

