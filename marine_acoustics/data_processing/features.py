"""
Frame audio and extract features.

"""


import librosa
import pywt
import numpy as np
from scipy.stats import describe
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
        
    elif s.FEATURES == 'STFT_STATS':
        y_features = stft_stats(y)   # Summary stats for STFT
        
    elif s.FEATURES == 'CWT_STATS':
        y_features = cwt_stats(y)    # Summary stats for CWT
      
    # 2D Features
    elif s.FEATURES == 'STFT':
        y_features = calculate_stft(y)    # Calculate STFT frames
        
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
    
    # Calculate MFCCs (n_frames x n_mfccs)
    mfccs = librosa.feature.mfcc(y=y,
                                 sr=s.SR,
                                 n_mfcc=s.N_MFCC,
                                 n_fft=s.FRAME_LENGTH,
                                 hop_length=s.HOP_LENGTH,
                                 n_mels=s.N_MELS,
                                 center=False,
                                 fmin=s.FMIN,
                                 fmax=s.FMAX).T
   
    mfcc_deltas = librosa.feature.delta(mfccs, width=s.DELTA_WIDTH, order=1,
                                        axis=1, mode='interp')
    
    
    mfcc_delta_deltas = librosa.feature.delta(mfccs, width=s.DELTA_WIDTH,
                                              order=2, axis=1, mode='interp')

    mfcc_features = np.hstack((mfccs, mfcc_deltas, mfcc_delta_deltas))
    
    return mfcc_features
  

def calculate_cwt_avg(y):
    """Return the avg. coeffs for each cwt frame."""
    
    cwt_strided_frames = calculate_cwt(y)  
    cwt_avg = np.mean(cwt_strided_frames, axis=1)
    
    return cwt_avg


def stft_stats(y):
    """Return summary statistics along the time dimension
    for each STFT spectrogram frame.
    
    STFT is (time x freq)
    
    Return (1 x freq)
    """
    
    # (n_frames x n_windows x n_freq_bins)
    stft_frames = calculate_stft(y)
    
    
    stats = describe(stft_frames, axis=1)
    mean = stats.mean
    variance = stats.variance
    skewness = stats.skewness
    kurtosis = stats.kurtosis
    
    stft_summary_stats = np.hstack((mean, variance, skewness, kurtosis))
    
    return stft_summary_stats


def cwt_stats(y):
    """Return summary statistics along the time dimension
    for each CWT scalogram frame.
    
    CWT is (time x freq)
    
    Return (1 x freq)
    """
    
    # (n_frames x n_windows x n_freq_bins)
    cwt_frames = calculate_cwt(y)
    
    
    stats = describe(cwt_frames, axis=1)
    mean = stats.mean
    variance = stats.variance
    skewness = stats.skewness
    kurtosis = stats.kurtosis
    
    cwt_summary_stats = np.hstack((mean, variance, skewness, kurtosis))
    
    return cwt_summary_stats
    


def calculate_stft(y):
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


def calculate_cwt(y):
    """Compute cwt scalogram for each frame.
    
    Matches the number of dimensions in STFT in both time and frequency axes.
    
    Returns: (n_frames x n_coefs_per_frame x n_freq_bins)
        
    """
    
    # Compute n_scales to match n_STFT_freq_bins
    fmin_idx, fmax_idx = utils.get_stft_freq_indexes(s.N_FFT)
    n_scales = 1+fmax_idx-fmin_idx
    
    # Choose wavelet scales (pywt freq2scale uses normalised frequency)
    desired_freqs = np.linspace(s.FMIN, s.FMAX+1, num=n_scales)
    scales = pywt.frequency2scale(s.WAVELET, desired_freqs/s.SR)

    # Compute continuous wavelet transform (n_samples x n_freq_bins)
    wavelet_coeffs, wavelet_freqs = pywt.cwt(y, scales, s.WAVELET,
                                             sampling_period=1/s.SR)
    cwt = librosa.amplitude_to_db(np.abs(wavelet_coeffs), ref=np.max).T

    # Frame cwt along sample axis (n_frames x n_coefs_per_frame x n_freq_bins)
    cwt_frames = librosa.util.frame(cwt,
                                    frame_length=s.FRAME_LENGTH,
                                    hop_length=s.HOP_LENGTH, axis=0)

    # Subsample coefficients axis to equal hop length of STFT
    cwt_strided_frames = cwt_frames[:, ::s.STFT_HOP, :]
    
    return cwt_strided_frames

