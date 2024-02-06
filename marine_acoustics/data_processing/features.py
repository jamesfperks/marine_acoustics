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
        y_features = calculate_stft(y)    # Calculate STFT
        
    elif s.FEATURES == 'MEL':
        y_features = calculate_melspectrogram(y)  # Calculate mel-spectrogram
        
    elif s.FEATURES == 'CWT':
        y_features = calculate_cwt(y)   # Calculate cwt
        
    elif s.FEATURES == 'SMILE':
        y_features = calculate_smile(y)   # extract openSMILE features
    
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
    desired_freqs = np.arange(s.FMIN, s.FMAX+1, s.CWT_FREQ_RES)
    scales = frequency2scale(desired_freqs)
    
    # Compute continuous wavelet transform
    wavelet_coeffs, wavelet_freqs = apply_cwt(y, scales)
    cwt = librosa.amplitude_to_db(np.abs(wavelet_coeffs), ref=np.max)
    
    # Pad cwt to match librosa frame offset in STFT/MFCC etc.
    size = cwt.shape[1] + (s.FRAME_LENGTH//2)*2
    cwt_padded = librosa.util.pad_center(cwt, size=size, axis=1)
      
    # Split coefficients into frames along sample axis
    cwt_framed = librosa.util.frame(cwt_padded,
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


def calculate_smile(y):
    """Frame data and extract openSMILE features for each frame."""
    
    # Pad cwt to match librosa frame offset in STFT/MFCC etc.
    size = y.shape[0] + (s.FRAME_LENGTH//2)*2
    y_padded = librosa.util.pad_center(y, size=size, axis=0)
    
    # Split into frames
    y_frames = librosa.util.frame(y_padded,
                                  frame_length=s.FRAME_LENGTH,
                                  hop_length=s.HOP_LENGTH, axis=0)    

    smile_features = np.apply_along_axis(extract_smile_feature_vector, 1, 
                                         y_frames)
   
    return smile_features


def extract_smile_feature_vector(y_frame):
    """Extract smile feature vector for a single frame."""
    
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    feature_vector = smile.process_signal(y_frame, s.SR).iloc[0].to_numpy()
    
    return feature_vector

