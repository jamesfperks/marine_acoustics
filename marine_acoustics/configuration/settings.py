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
TRAINING_SITES = [1,2,3]         # Indexes of sites for training (E.g. [1,4])

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
FEATURES = 'STFT'        # [MFCC, STFT, MEL, CWT]


# FRAME DURATION AND OVERLAP
# -----------------------------------------------------------------------------
FRAME_DURATION = 1000    # Frame duration in milliseconds
FRAME_OVERLAP = 50       # Frame overlap (%)


# FREQUENCY RANGE
# -----------------------------------------------------------------------------
FMAX = 35               # Frequency lower bound used in MFCC, STFT features
FMIN = 18               # Frequency upper bound used in MFCC, STFT features


# MFCC CONSTANTS
# -----------------------------------------------------------------------------
N_MFCC = 12             # no. of mfccs to calculate


# MEL CONSTANTS
# -----------------------------------------------------------------------------
N_MELS = 32             # no. Mel bands used in mfcc calc (default 128)


# WAVELET CONSTANTS
# -----------------------------------------------------------------------------
WAVELET = 'morl'        # wavelet type: morlet


# EVALUATION CONSTANTS
# -----------------------------------------------------------------------------
MEDIAN_FILTER_SIZE = 5    # Size of 1D median filter kernel


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

