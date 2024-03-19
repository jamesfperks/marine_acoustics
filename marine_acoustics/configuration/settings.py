"""
Settings used in the main script.

Select recording sites and call types for train/test sets,
the model, and the feature extraction method used.

All global constants are defined here.

"""


# GENERAL CONSTANTS
# -----------------------------------------------------------------------------
DATA_FILEPATH = 'data/raw/AcousticTrends_BlueFinLibrary'
SAVE_DATA_FILEPATH = 'data/processed'
SEED = 12345         # Set random seed


# RECORDING SITE SELECTION
# -----------------------------------------------------------------------------
TRAIN_SITES = [10]         # Sites for training (E.g. [1,4])
 
TEST_SITES = [7]           # Sites for testing.
                           # [] defaults to all sites not used in training.
                                 

# CALL TYPE SELECTION
# -----------------------------------------------------------------------------
TRAIN_CALL_TYPES = [3]        # Call types for training

TEST_CALL_TYPES = []          # Call types for testing
                              # [] defaults to same call type as trained on.


# BINARY CLASSIFICATION
# -----------------------------------------------------------------------------
# Select call type for the negative class. [] defaults to background.

TRAIN_NEGATIVE_CLASS = []       # Negative class for training
TEST_NEGATIVE_CLASS = []         # Negative class for testing

IS_TEST_BALANCED = True         # Balance the positive and negative class
                                # in the test set (True/False)


# MODEL
# -----------------------------------------------------------------------------
MODEL = 'HGB'        # [HGB, CNN]


# FEATURE EXTRACTION METHOD
# -----------------------------------------------------------------------------
FEATURES = 'STFT'  # [MFCC, STFT, MEL, CWT, SMILE]
                         # [STFT_FRAME, CWT_FRAME]


# FRAME DURATION AND OVERLAP
# -----------------------------------------------------------------------------
FRAME_DURATION = 3000    # Frame duration in milliseconds
FRAME_ADVANCE = 1500     # Frame advance in milliseconds

STFT_DURATION = 1024     # STFT window duration 
STFT_ADVANCE = 100       # STFT window advance in milliseconds
       

# FREQUENCY RANGE
# -----------------------------------------------------------------------------
FMAX = 29               # Frequency lower bound used in MFCC, STFT features
FMIN = 15               # Frequency upper bound used in MFCC, STFT features


# MFCC CONSTANTS
# -----------------------------------------------------------------------------
N_MFCC = 12             # no. of mfccs to calculate


# MEL CONSTANTS
# -----------------------------------------------------------------------------
N_MELS = 16             # no. Mel bands used in mfcc calc (default 128)


# WAVELET CONSTANTS
# -----------------------------------------------------------------------------
WAVELET = 'shan0.07-0.8'     # select wavelet
CWT_FREQ_RES = 1          # Frequency resolution of cwt


"""
Examples of good wavelet choices:
    
'cmor25-2.0'           complex morlet (bandwidth = 25 centre freq = 2.0)
'shan0.07-0.8'         shannon (bandwidth = 0.07 centre freq = 0.8)
"""


# EVALUATION CONSTANTS
# -----------------------------------------------------------------------------
MEDIAN_FILTER_SIZE = 3    # Size of 1D median filter kernel


# PRINTOUT CONSTANTS
# -----------------------------------------------------------------------------
HEADER_LEN = 50
SUBHEADER_LEN = 30


# OTHER CONSTANTS
# -----------------------------------------------------------------------------
SR = 250             # Resample rate in Hz (Do not change from 250 Hz)


#------------------------------------------------------------------------------
# CALCULATED CONSTANTS (DO NOT CHANGE)
#------------------------------------------------------------------------------
FRAME_LENGTH = round(SR*FRAME_DURATION/1000)    # frame length (samples)
HOP_LENGTH = round(SR*FRAME_ADVANCE/1000)       # hop length (samples)

STFT_LEN = round(SR*STFT_DURATION/1000)         # stft window length (samples)
STFT_HOP = round(SR*STFT_ADVANCE/1000)          # stft hop (samples)
N_FFT = STFT_LEN*2       # Pad STFT window with zeros to increase n_freq_bins

#------------------------------------------------------------------------------

