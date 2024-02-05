"""
Settings used in the main script.

Select recording sites and call types for train/test sets
and the feature extraction method used.

All global constants are defined here.

"""


# GENERAL CONSTANTS
# -----------------------------------------------------------------------------
DATA_FILEPATH = 'data/AcousticTrends_BlueFinLibrary'
SEED = 12345         # Set random seed
SR = 250             # Resample rate in Hz


# TRAIN/TEST SET SELECTION
# -----------------------------------------------------------------------------
TRAINING_SITES = [3]         # Indexes of sites for training (E.g. [1,4])

TRAINING_CALL_TYPES = [0]        # Indexes of call types for training
 
TEST_SITES = [8]                 # Indexes of sites for testing

TEST_CALL_TYPES = []             # Indexes of call types for testing

IS_TEST_BALANCED = True         # Balance the test sample set (True/False)

"""
Note for test set selection:
    - [] defaults to using all sites not used in training
    - [] defaults to using the same call type as trained on
"""


# FEATURE EXTRACTION METHOD
# -----------------------------------------------------------------------------
FEATURES = 'CWT'        # [MFCC, STFT, MEL, CWT]


# FRAME DURATION AND OVERLAP
# -----------------------------------------------------------------------------
FRAME_DURATION = 1000    # Frame duration in milliseconds
FRAME_OVERLAP = 50       # Frame overlap (%)


# FREQUENCY RANGE
# -----------------------------------------------------------------------------
FMAX = 30               # Frequency lower bound used in MFCC, STFT features
FMIN = 20               # Frequency upper bound used in MFCC, STFT features


# MFCC CONSTANTS
# -----------------------------------------------------------------------------
N_MFCC = 12             # no. of mfccs to calculate


# MEL CONSTANTS
# -----------------------------------------------------------------------------
N_MELS = 32             # no. Mel bands used in mfcc calc (default 128)


# WAVELET CONSTANTS
# -----------------------------------------------------------------------------
WAVELET = 'shan0.07-0.8'        # select wavelet
CWT_FREQ_RES = 0.5          # Frequency resolution of cwt


"""
Examples of good wavelet choices:
    
'cmor25-2.0'           complex morlet (bandwidth = 25 centre freq = 2.0)
'shan0.07-0.8'         shannon (bandwidth = 0.07 centre freq = 0.8)
"""


# EVALUATION CONSTANTS
# -----------------------------------------------------------------------------
MEDIAN_FILTER_SIZE = 7    # Size of 1D median filter kernel


# PRINTOUT CONSTANTS
# -----------------------------------------------------------------------------
HEADER_LEN = 50
SUBHEADER_LEN = 30


#------------------------------------------------------------------------------
# CALCULATED CONSTANTS (DO NOT CHANGE)
#------------------------------------------------------------------------------
FRAME_LENGTH = round(SR*FRAME_DURATION/1000)    # frame length (samples)
HOP_LENGTH = round(FRAME_LENGTH *(100-FRAME_OVERLAP)/100) # hoplength (samples)

#------------------------------------------------------------------------------

