import os
import time
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report




"""
Read the Australian Antarctic Data Centre Annotated Library of 
Anartctic Blue and Fin Whale sounds.
"""


# CONSTANTS
# -----------------------------------------------------------------------------
DATA_FILEPATH = 'data/AcousticTrends_BlueFinLibrary'
SR = 250                # Resample rate in Hz
FRAME_DURATION = 1000    # Frame duration in milliseconds
FRAME_OVERLAP = 50      # Frame overlap (%)
FMAX = 35
FMIN = 10
STFT_WINDOW_DURATION = 200   # STFT window duration in milliseconds
STFT_OVERLAP = 75      # STFT window overlap (%)
N_MFCC = 12             # no. of mfccs to calculate
N_MELS = 32              # no. Mel bands used in mfcc calc (default 128)
SEED = 12345            # Set random seed

# Indexes of sites for training
TRAINING_SITES = [1,2,3]

# Indexes of call types for training
TRAINING_CALL_TYPES = [0]    
 
# Indexes of sites for testing
# [] empty brace defaults to using all sites not used in training
TEST_SITES = [6]

# Indexes of call types for testing
# [] empty brace defaults to using the same call type as trained on
TEST_CALL_TYPES = []

#------------------------------------------------------------------------------
# CALCULATED CONSTANTS (DO NOT CHANGE)
#------------------------------------------------------------------------------
FRAME_LENGTH = round(SR * FRAME_DURATION / 1000)    # frame length (samples)
HOP_LENGTH = round(FRAME_LENGTH *(100-FRAME_OVERLAP)/100) # hoplength (samples)

#------------------------------------------------------------------------------


def get_folder_structure():
    """Read the folder structure csv file and save as a pd dataframe."""
    
    csv_filepath = DATA_FILEPATH + '/01-Documentation/folderStructure.csv'
    df = pd.read_csv(csv_filepath, index_col=0)
    df.index.name = None
      
    return df


def print_recording_sites(df_folder_structure):
    """Print list of recording sites with indexes."""
    
    df = df_folder_structure
    print('Recording sites:\n' + '-'*20)
    for i in range(df.shape[0]):
        print(f' {i}' + ' '*(4-len(str(i))) + df.index[i])
        
        
def get_call_types():
    """Create dataframe of call types."""
    
    call_types = ['Bm-A', 'Bm-B', 'Bm-Z', 'Bm-D',
                       'Bp-20', 'Bp-20+', 'Bp-Downsweep', 'Unidentified']
    
    return call_types
    
 
def print_call_types():
    """Print all call types and the corresponding index."""

    call_types = get_call_types()
    print('\nCall types:\n' + '-'*20)
    for i in range(len(call_types)):
        print(f' {i}  ' + call_types[i])


def get_total_annotation_count(df_folder_structure):
    """
    Return a dataframe containing the total number of annotations
    for each site and call type.
    """
    
    call_types = get_call_types()
    annotation_dict = {}
    
    for call_type in call_types:
        annotation_counts = []
        
        for site in df_folder_structure.index:
            annotation_counts.append(count_annotations(df_folder_structure,
                                                      site, call_type))
          
        annotation_dict[call_type] = annotation_counts
    
    return pd.DataFrame(annotation_dict, index=df_folder_structure.index)
    

def get_log_filepath(site, call_type, df_folder_structure):
    """Return filepath to a log file given a site name and call type."""
    
    log_header = call_type_2_log_header(call_type)
    rel_filepath = df_folder_structure.loc[site, ['Folder', log_header]].str.cat()
    log_filepath = DATA_FILEPATH + '/' + rel_filepath
    
    return log_filepath


def count_annotations(df_folder_structure, site, call_type):
    """Count the number of call annotations for a given site and call type."""
    
    log_filepath = get_log_filepath(site, call_type, df_folder_structure)
    
    with open(log_filepath, "rb") as f:
        n_annotations = sum(1 for line in f) - 1

    return n_annotations
    

def call_type_2_log_header(call_type):
    """Convert call-type string to the RavenFile name in the log headings."""
    
    call_2_log = {'Bm-A': 'Abw-A_RavenFile',
                  'Bm-B': 'Abw-B_RavenFile',
                  'Bm-Z': 'Abw-Z_RavenFile',
                  'Bm-D': 'BmD_RavenFile',
                  'Bp-20': 'Bp20_RavenFile',
                  'Bp-20+': 'Bp20Plus_RavenFile',
                  'Bp-Downsweep': 'BpDownsweep_RavenFile',
                  'Unidentified': 'UnidentifiedCalls_RavenFile'
                  }
    
    return call_2_log[call_type]


def read_log(site, call_type, df_folder_structure):
    """Read log .txt file into a dataframe."""
    
    log_filepath = get_log_filepath(site, call_type, df_folder_structure)
    
    
    fields = ['Begin File', 'End File','Begin Time (s)', 'End Time (s)',
              'Beg File Samp (samples)', 'End File Samp (samples)',
              'Begin Date Time', 'Delta Time (s)', 'Low Freq (Hz)',
              'High Freq (Hz)', 'Dur 90% (s)', 'Freq 5% (Hz)', 'Freq 95% (Hz)']
    
    df_log = pd.read_csv(log_filepath, sep='\t', usecols=fields)
    
    return df_log


def read_audio(site, wav_filename, df_folder_structure):
    """Read, resample and normalise the given .wav file. Return the resampled
    audio along with the default sample rate."""
    
    # Truncated redundant .wav filenames to match the G64 2015 folder
    if site == 'G64 2015':
        wav_filename = wav_filename[:-21] + '.wav'
    
    
    # File path to .wav file
    site_folder = df_folder_structure.loc[site, 'Folder'][:-1]
    wav_filepath = DATA_FILEPATH + '/' + site_folder + '/wav/' + wav_filename
    
    # Read entire mono .wav file and resample to preset global sample rate
    y, sr = librosa.load(wav_filepath, sr=SR)
    
    # Normalise to [-1, 1]
    y = librosa.util.normalize(y)
    
    # Store the default sample rate
    sr_default = librosa.get_samplerate(wav_filepath)
    
    return y, sr_default


def calculate_mfccs(y):
    """Split data into DFT windows and compute MFCCs for each window."""
    
    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(y=y,
                                 sr=SR,
                                 n_mfcc=N_MFCC,
                                 n_fft=FRAME_LENGTH,
                                 hop_length=HOP_LENGTH,
                                 n_mels=N_MELS,
                                 fmin=FMIN,
                                 fmax=FMAX).T
    
    return mfccs
    

def calculate_stft(y, sr):
    """Compute STFT and split data into frames."""
    
    # Calculate frame size and overlap in samples
    FRAME_LENGTH = sr * FRAME_DURATION // 1000
    HOP_LENGTH = FRAME_LENGTH * (100-FRAME_OVERLAP) // 100
    STFT_WINDOW_LENGTH = sr * STFT_WINDOW_DURATION // 1000
    STFT_HOP_LENGTH = STFT_WINDOW_LENGTH * (100-FRAME_OVERLAP) // 100
    
    # STFT of y
    D = librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    # STFT in dB
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max).T
    
                
    return S_db


def extract_features(y):
    """Frame data and extract features for each frame."""
    
    # Calculate MFCCs
    mfccs = calculate_mfccs(y)
    
    # Calculate STFT and split into frame
    #stft = calculate_stft(y, sr)
    
    # Return feature vectors for each frame
    y_features = mfccs
       
    return y_features


def index2frame(time_indexes):
    """Convert time log indexes to frame indexes."""
    
    # Frame index of the last frame that the sample is in
    frame_indexes = np.apply_along_axis(librosa.samples_to_frames,
                                        axis=0,
                                        arr=time_indexes,
                                        hop_length=HOP_LENGTH,
                                        n_fft=FRAME_LENGTH)
    
    # Deal with negative indexes caused by librosa n_fft offset
    frame_indexes[frame_indexes<0] = 0
    
    # Check
    for idx in frame_indexes.flatten():
        if idx < 0:
            raise ValueError('Negative frame index calculated during sample '
                             'index to frame index conversion.')
    
    return frame_indexes


def get_time_indexes(logs, sr_default):
    """Read the call log time stamps and calculate the index of each call
    for the resampled audio."""
    
    unsampled_indexes = logs[['Beg File Samp (samples)',
                                   'End File Samp (samples)']].to_numpy() - 1
    
    time_indexes = np.rint(SR*unsampled_indexes/sr_default)
    
    return time_indexes


def label_features(y_features, logs, sr_default):
    """Label feature vectors as 1 for 'whale' or 0 for 'background'."""
    
    # Get the sample indexes of start/end of whale calls
    time_indexes = get_time_indexes(logs, sr_default)
    
    # Convert time indexes to frame indexes
    frame_indexes = index2frame(time_indexes)
    
    # For each annotation, label corresponding frames 1 if "whale", else 0 
    feature_labels = np.zeros(y_features.shape[0])  
    for start, end in frame_indexes:  
        whale_indexes = np.arange(start, end) # Non-inclusive end frame idx
        feature_labels[whale_indexes] = np.ones(len(whale_indexes))
    
    y_labelled_features = np.column_stack((y_features, feature_labels))
    
    return y_labelled_features


def balance_dataset(samples):
    """Sub-sample the majority class to balance the dataset."""
    
    # Split into classes
    whale_samples = samples[samples[:,-1] == 1]
    background_samples = samples[samples[:,-1] == 0]
    
    # Randomise sample order
    np.random.seed(SEED)
    np.random.shuffle(whale_samples)
    np.random.shuffle(background_samples)
    
    # Subsample the majority class
    n_minority = min(len(whale_samples), len(background_samples))
    balanced_whale = whale_samples[0:n_minority, :]
    balanced_background = background_samples[0:n_minority, :]
    
    # Recombine and randomise samples from each class
    balanced_samples = np.vstack((balanced_whale, balanced_background))
    np.random.shuffle(balanced_samples)
  
    return balanced_samples


def split_sample_vector(samples):
    """Split samples into vectors X, y."""
    
    X = samples[:,0:-1]
    y = samples[:,-1]
    
    return X, y


def extract_samples(df_data_summary, df_folder_structure):
    """Extract labelled samples from .wav files."""
    
    sample_set = []
    
    for site in df_data_summary.index:
        
        logs = []
        
        for call_type in df_data_summary.columns:
            
            df_log = read_log(site, call_type, df_folder_structure)
            
            if not df_log.empty:
                logs.append(df_log)
        
        # concatenate all logs into one DF
        if len(logs) == 0:
            continue
        df_logs = pd.concat(logs)

        # Groupby .wav filename
        gb_wavfile = df_logs.groupby('Begin File')
        
        
        # For .wav in groupby object
        for wavfile, logs in gb_wavfile:
            
            # Read in audio
            y, sr_default = read_audio(site, wavfile, df_folder_structure)
            
            # Frame and extract features
            y_features = extract_features(y)
            
            # Label features
            y_labelled_features = label_features(y_features, logs, sr_default)
            
            # add samples to sample set
            sample_set.extend(y_labelled_features)
    
    # Create sample vector
    samples = np.vstack(sample_set)
    
    # Balance and randomise samples
    balanced_samples = balance_dataset(samples)
    
    # Split sample vector
    X, y = split_sample_vector(balanced_samples)
    
    return X, y


def select_training_set(df_annotations):
    """Select sites and call types to use for training."""
    
    # Training set annotation summary
    df_trainset = df_annotations.iloc[TRAINING_SITES, TRAINING_CALL_TYPES]
    
    # Raise error if no annotations exist
    if not df_trainset.any(axis=None):
        sites = df_annotations.index[TRAINING_SITES].to_list()
        calls = df_annotations.columns[TRAINING_CALL_TYPES].to_list()
        raise ValueError('Chosen sites and call-types '
                          'contain zero annotations.', sites, calls)
    
    return df_trainset
       

def get_training_samples(df_trainset, df_folder_structure):
    """Return extracted trainings samples and print time taken."""
    
    print('\n'*2 + '-'*50 + '\nTRAINING PROGRESS\n' + '-'*50 + 
          '\n  - Extracting trainings samples...', end='')
    

    start = time.time()
    X_train, y_train = extract_samples(df_trainset, df_folder_structure)
    end = time.time()
    
    print(f'100% ({end-start:.1f} s)')
    
    return X_train, y_train


def select_test_set(df_annotations):
    """Select sites and call types to use for testing."""
    
    test_sites = TEST_SITES
    test_call_types = TEST_CALL_TYPES
    
    # Default to using all non-training sites if unspecified []
    if len(test_sites) == 0:
        test_sites = list(range(0, 11))  
        for site_idx in TRAINING_SITES:   
            test_sites.remove(site_idx)
    
    # Default to using training call types if unspecified []
    if len(test_call_types) == 0:
        test_call_types = TRAINING_CALL_TYPES
            
    # Test set summary
    df_testset = df_annotations.iloc[test_sites, test_call_types]
    
    # Raise error if no annotations exist
    if not df_testset.any(axis=None):
        sites = df_annotations.index[test_sites].to_list()
        calls = df_annotations.columns[test_call_types].to_list()
        raise ValueError('Chosen sites and call-types '
                          'contain zero annotations.', sites, calls)
    
    return df_testset


def get_test_samples(df_testset, df_folder_structure):
    """Return extracted test samples and print time taken."""
    
    print('  - Extracting test samples...', end='')
    
    start = time.time()
    X_test, y_test = extract_samples(df_testset, df_folder_structure)
    end = time.time()
    
    print(f'100% ({end-start:.1f} s)')
    
    return X_test, y_test


def print_dataset_summary(df_trainset, df_testset):
    """Print a summary of the number of whale call annotations for the
    training set and tesst set."""
    
    # Train/test ratio
    train_tot = df_trainset.to_numpy().sum()
    test_tot = df_testset.to_numpy().sum()
    train_percent = round(100*train_tot/(train_tot + test_tot))
    test_percent = round(100*test_tot/(train_tot + test_tot))
    
    # Dataset summary
    print('\n'*2 + '-'*50 + '\nDATASET SUMMARY\n' + '-'*50)
    
    # Print training set summary
    print('\n' + f'\nTraining set: ({train_tot})\n'
          + '-'*30 + f'\n{df_trainset}')
    
    # Print testset summary
    print('\n'*2 + f'\nTest set: ({test_tot})\n'
          + '-'*30 + f'\n{df_testset}')
    
    # Train/test ratio printout
    print('\n'*2 + 'Percentage split train/test is '
          f'{train_percent}/{test_percent}.\n')
    
    
def train_classifier(X_train, y_train):
    """Train classifier."""
    
    print('  - Training model...', end='')
    
    start = time.time()
    clf = GradientBoostingClassifier().fit(X_train, y_train)
    end = time.time()
    
    print(f'100% ({end-start:.1f} s)')
   
    return clf


def get_results(clf, X_train, y_train, X_test, y_test):
    """Calculate and print classification results."""
    
    # Accuracy
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print('\n'*2 + '-'*50 + '\nRESULTS\n' + '-'*50)
    print(f'\n  - Training: {train_score:.3f}\n  - Testing: {test_score:.3f}')
    
    # Confusion matrix
    tn, fp, fn, tp = calculate_confusion_matrix(X_test, y_test, clf)
    
    
def calculate_confusion_matrix(X_test, y_test, clf):
    """Caclulate and print confusion matrix."""
    
    y_pred = clf.predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred)
    
    # Printout
    ref = np.array([['TN', 'FP'], ['FN', 'TP']])
                  
    print('\n' + '-'*0 + '\nConfusion Matrix:\n' + '-'*30 + 
          f'\n   {ref[0]}' + '-'*3 + f'{c_matrix[0]}' + 
          f'\n   {ref[1]}' + '-'*3 + f'{c_matrix[1]}')
        
    tn, fp, fn, tp = c_matrix.ravel()
    
    return tn, fp, fn, tp


def run():
    """Executes script."""
    
    # Get folder structure
    df_folder_structure = get_folder_structure()
    
    # Print sites
    print_recording_sites(df_folder_structure)
    
    # Print call types
    print_call_types()
    
    # Get total annotation count
    df_annotations = get_total_annotation_count(df_folder_structure)
    
    # Select training set
    df_trainset = select_training_set(df_annotations)
    
    # Select test set
    df_testset = select_test_set(df_annotations)
    
    # Print training and test set summary
    print_dataset_summary(df_trainset, df_testset)
    
    # Get training samples
    X_train, y_train = get_training_samples(df_trainset, df_folder_structure)
    
    # Get test samples
    X_test, y_test = get_test_samples(df_testset, df_folder_structure)
    
    # Train model
    clf = train_classifier(X_train, y_train)
    
    # Print results
    get_results(clf, X_train, y_train, X_test, y_test)
    
    

def main():
    # Start of script
    print('-'*40 + f'\nRunning {os.path.basename(__file__)}\n' + '-'*40 + '\n')
    print('An annotated library of Antarctic Blue and Fin Whale sounds.\n')

    # Run and time script
    start = time.time()
    run()
    end = time.time()
    
    # End of script
    print('\n'*2 + f'Total runtime: {end-start:0.1f} seconds.\n' + '-'*47 + 'End')
          
          
          #'\n'*2 + 'End' + '-'*47)
    
if __name__ == '__main__':
    main()
    