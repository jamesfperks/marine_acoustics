"""
Plot the waveform and spectrogram of a humpback whale .wav file.

"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.io import wavfile


# Close any open figures
plt.close()


# File path to .wav file
audio_file_path = "humpback.wav"


# Read the .wav file
sample_freq, signal_data = wavfile.read(audio_file_path)


# Calculate time in seconds
time = np.linspace(0, len(signal_data) / sample_freq, num=len(signal_data))


# Plot waveform
plt.subplot(211)
plt.title('Humpback Whale Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitdue')
plt.plot(time, signal_data)


# Plot spectrogram
plt.subplot(212)
plt.title('Humpback Whale Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.specgram(signal_data, Fs=sample_freq)


# Figure adjustments
plt.tight_layout()

