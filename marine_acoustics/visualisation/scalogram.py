"""
Plot scalogram given a raw audio file.
"""


import pywt
import librosa
import numpy as np
import matplotlib.pyplot as plt
from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import feature_utils


def plot_scalogram(y, wavelet=s.WAVELET, ylim=[s.FMIN, s.FMAX],
                   hop_length=s.STFT_HOP, n_scales=None, colorbar=True,
                   ax=None, title="Wavelet Scalogram"):
    """Plot the wavelet scalogram. Default to main script settings."""

    if n_scales==None:
        # Compute n_scales to match n_STFT_freq_bins
        fmin_idx, fmax_idx = feature_utils.get_stft_freq_indexes(s.N_FFT)
        n_scales = 1+fmax_idx-fmin_idx
    
    # Choose wavelet scales (pywt freq2scale uses normalised frequency)
    desired_freqs = np.linspace(ylim[0], ylim[1]+1, num=n_scales)
    scales = pywt.frequency2scale(wavelet, desired_freqs/s.SR)

    # Compute continuous wavelet transform (n_freq_bins x n_samples)
    wavelet_coeffs, wavelet_freqs = pywt.cwt(y, scales, wavelet,
                                             sampling_period=1/s.SR)
    cwt = librosa.amplitude_to_db(np.abs(wavelet_coeffs), ref=np.max)
    
    # Subsample coefficients axis to equal hop length of STFT
    cwt_strided = cwt[:, ::hop_length]

    # Compute time array
    t = np.linspace(0, len(y)/s.SR, num=cwt_strided.shape[1])
    
    # Plot the wavelet scalogram.
    if ax==None:
        ax = plt.axes()
    im = ax.pcolormesh(t, wavelet_freqs, cwt_strided, cmap='magma')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pseudo-frequency (Hz)")
    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, format="%+2.f dB")
        
