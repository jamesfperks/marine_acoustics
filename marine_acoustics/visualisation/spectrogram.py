"""
Plot spectrogram given a raw audio file.
"""


import librosa
import numpy as np
import matplotlib.pyplot as plt
from marine_acoustics.configuration import settings as s


def plot_spectrogram(y, n_fft=s.N_FFT, hop_length=s.STFT_HOP,
                     win_length=s.STFT_LEN, ylim=[s.FMIN, s.FMAX],
                     colorbar=True):
    """
    Plot the linear-frequency power spectrogram.
    
    Defaults to using the current settings for STFT.
    
    Limits frequency axis to ylim = [fmin, fmax] in Hz.
    """
    
    # STFT of y
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length)

    # STFT in dB
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plot
    librosa.display.specshow(S_db, sr=s.SR, hop_length=hop_length,
                             x_axis='s', y_axis='linear')
    plt.title('Linear-frequency Power Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(ylim)
    if colorbar:
        plt.colorbar(format="%+2.f dB")
    
    