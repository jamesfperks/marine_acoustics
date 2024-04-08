"""
Frame audio and extract features.

"""


import librosa
import pywt
import numpy as np
from scipy.stats import describe
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing.features import feature_utils


def extract_features(y_frame):
    """Extract features for each raw audio frame."""
      
    if s.FEATURES == 'STFT':
        y_features = calculate_stft(y_frame)
    
    elif s.FEATURES == 'STFT_STATS':
        y_features = stft_stats(y_frame)        # 1D summary stats for STFT
        
    elif s.FEATURES == 'CWT':
        y_features = calculate_cwt(y_frame)
        
    elif s.FEATURES == 'CWT_STATS':
        y_features = cwt_stats(y_frame)         # 1D summary stats for CWT
        
    elif s.FEATURES == 'MFCC':
        y_features = calculate_mfccs(y_frame)   # Average MFCCs for the frame
        
    else:
        raise NotImplementedError('Feature representation chosen ' 
                                        'is not implemented', s.FEATURES)
       
    return y_features


def calculate_stft(y_frame):
    """Calculate the STFT for the given frame.
    Returns: (n_windows x n_freq_bins)
    """
    stft = feature_utils.apply_stft(y_frame)
    
    return stft


def stft_stats(y_frame):
    """Return summary statistics along the time dimension
    for a single STFT spectrogram frame.
    
    STFT is (time x freq)
    
    Return (1 x freq X n_summary_stats_used)
    """
    
    # (n_windows x n_freq_bins)
    stft = calculate_stft(y_frame)
    
    # Compute statistics along time dimension
    stats = describe(stft, axis=0)
    mean = stats.mean
    variance = stats.variance
    skewness = stats.skewness
    kurtosis = stats.kurtosis
    
    stft_summary_stats = np.hstack((mean, variance, skewness, kurtosis))
    
    return stft_summary_stats


def calculate_cwt(y_frame):
    """Compute cwt scalogram for the given frame.
    
    Matches the number of dimensions in STFT in both time and frequency axes.
    
    Returns: (n_coefs_per_frame x n_freq_bins)
        
    """
    
    # Compute n_scales to match n_STFT_freq_bins
    fmin_idx, fmax_idx = feature_utils.get_stft_freq_indexes(s.N_FFT)
    n_scales = 1+fmax_idx-fmin_idx
    
    # Choose wavelet scales (pywt freq2scale uses normalised frequency)
    desired_freqs = np.linspace(s.FMIN, s.FMAX+1, num=n_scales)
    scales = pywt.frequency2scale(s.WAVELET, desired_freqs/s.SR)

    # Compute continuous wavelet transform (n_samples x n_freq_bins)
    wavelet_coeffs, wavelet_freqs = pywt.cwt(y_frame, scales, s.WAVELET,
                                             sampling_period=1/s.SR)
    wavelet_coeffs = librosa.amplitude_to_db(np.abs(wavelet_coeffs), ref=np.max).T

    # Subsample coefficients axis to equal hop length of STFT
    cwt = wavelet_coeffs[::s.STFT_HOP, :]  
    
    return cwt


def cwt_stats(y_frame):
    """Return summary statistics along the time dimension
    for a CWT scalogram frame.
    
    CWT is (time x freq)
    
    Return (1 x freq X n_summary_stats_used)
    """
    
    # (n_windows x n_freq_bins)
    cwt = calculate_cwt(y_frame)
    
    stats = describe(cwt, axis=0)
    mean = stats.mean
    variance = stats.variance
    skewness = stats.skewness
    kurtosis = stats.kurtosis
    
    cwt_summary_stats = np.hstack((mean, variance, skewness, kurtosis))
    
    return cwt_summary_stats


def calculate_mfccs(y_frame):
    """Compute MFCCs and deltas for each window in the frame.
    Average over the number of windows in the frame.
    
    Returns: (1 x (n_mfccs + n_deltas + n_delta_deltas))"""
    
    # Calculate MFCCs (n_time_windows x n_mfccs)
    mfccs = librosa.feature.mfcc(y=y_frame,
                                 sr=s.SR,
                                 n_mfcc=s.N_MFCC,
                                 n_fft=s.N_FFT,
                                 hop_length=s.STFT_HOP,
                                 win_length=s.STFT_LEN,
                                 n_mels=s.N_MELS,
                                 center=True,
                                 fmin=s.FMIN,
                                 fmax=s.FMAX).T
    
    mfcc_deltas = librosa.feature.delta(mfccs, width=s.DELTA_WIDTH, order=1,
                                        axis=0, mode='interp')
    
    
    mfcc_delta_deltas = librosa.feature.delta(mfccs, width=s.DELTA_WIDTH,
                                              order=2, axis=0, mode='interp')
    
    # Combine mfccs and deltas (n_time_windows x n_combined_mfcc_features)
    mfcc_features = np.hstack((mfccs, mfcc_deltas, mfcc_delta_deltas))
    
    # Average along time dimension
    mfcc_features = np.mean(mfcc_features, axis=0)
    
    return mfcc_features

