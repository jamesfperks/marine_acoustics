import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report




"""
Read the Australian Antarctic Data Centre Annotated Library of 
Anartctic Blue and Fin Whale sounds.
"""


# CONSTANTS
# ----------------------------------------------------------------------------
DATA_FILEPATH = 'data/AcousticTrends_BlueFinLibrary'
FRAME_DURATION = 750    # Frame duration in milliseconds
FRAME_OVERLAP = 50      # Frame overlap (%)
N_MFCC = 12             # no. of mfccs to calculate
N_MELS = 32             # no. Mel bands used in mfcc calc (default 128)
#SEED = 12345            # Set random seed

# Indexes of sites for training
TRAINING_SITES = [0,1,2,3,4,5,6] 

# Indexes of call types for training
TRAINING_CALL_TYPES = [3]    
 
# Indexes of sites for testng
# [] empty brace defaults to using all sites not used in training
TEST_SITES = []

# Indexes of call types for testing
# [] empty brace defaults to using the same call type as trained on
TEST_CALL_TYPES = []

#--------------------------------------------------------------


def get_log_filenames():
    """Read the folder structure csv file and save as a pd dataframe."""
    
    csv_filepath = DATA_FILEPATH + '/01-Documentation/folderStructure.csv'
    df = pd.read_csv(csv_filepath, index_col=0)
    df.index.name = None
      
    return df


def print_recording_sites(df_log_filenames):
    """Print list of recording sites with indexes."""
    
    df = df_log_filenames
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


def get_total_annotation_count(df_log_filenames):
    """
    Return a dataframe containing the total number of annotations
    for each site and call type.
    """
    
    call_types = get_call_types()
    annotation_dict = {}
    
    for call_type in call_types:
        annotation_counts = []
        
        for site in df_log_filenames.index:
            annotation_counts.append(count_annotations(df_log_filenames,
                                                      site, call_type))
          
        annotation_dict[call_type] = annotation_counts
    
    return pd.DataFrame(annotation_dict, index=df_log_filenames.index)
    

def get_log_filepath(site, call_type, df_log_filenames):
    """Return filepath to a log file given a site name and call type."""
    
    log_header = call_type_2_log_header(call_type)
    rel_filepath = df_log_filenames.loc[site, ['Folder', log_header]].str.cat()
    log_filepath = DATA_FILEPATH + '/' + rel_filepath
    
    return log_filepath


def count_annotations(df_log_filenames, site, call_type):
    """Count the number of call annotations for a given site and call type."""
    
    log_filepath = get_log_filepath(site, call_type, df_log_filenames)
    
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


def read_log(site, call_type, df_log_filenames):
    """Read log .txt file into a dataframe."""
    
    log_filepath = get_log_filepath(site, call_type, df_log_filenames)
    
    
    fields = ['Begin File', 'End File','Begin Time (s)', 'End Time (s)',
              'Beg File Samp (samples)', 'End File Samp (samples)',
              'Begin Date Time', 'Delta Time (s)', 'Low Freq (Hz)',
              'High Freq (Hz)', 'Dur 90% (s)', 'Freq 5% (Hz)', 'Freq 95% (Hz)']
    
    df_log = pd.read_csv(log_filepath, sep='\t', usecols=fields)
    
    return df_log


def read_audio(site, wav_filename, df_log_filenames):
    """Read audio and sample rate and normalise given a .wav filename."""
    
    # Truncated redundant .wav filenames to match the G64 2015 folder
    if site == 'G64 2015':
        wav_filename = wav_filename[:-21] + '.wav'
    
    
    # File path to .wav file
    site_folder = df_log_filenames.loc[site, 'Folder'][:-1]
    wav_filepath = DATA_FILEPATH + '/' + site_folder + '/wav/' + wav_filename
    
    # Read entire mono .wav file using default sampling rate
    y, sr = librosa.load(wav_filepath, sr=None, duration=None)
    
    # Normalise to [-1, 1]
    y = librosa.util.normalize(y)
    
    return y, sr


def calculate_mfccs(y, sr):
    """Split data into DFT windows and compute MFCCs for each window."""
    
    # Calculate frame size and overlap in samples
    FRAME_LENGTH = sr * FRAME_DURATION // 1000
    HOP_LENGTH = FRAME_LENGTH * (100-FRAME_OVERLAP) // 100
    
    # Calculate MFCCs
    mfccs = np.transpose(librosa.feature.mfcc(y=y,
                                              sr=sr,
                                              n_mfcc=N_MFCC,
                                              n_fft=FRAME_LENGTH,
                                              hop_length=HOP_LENGTH,
                                              n_mels=N_MELS))
    
    return mfccs
    

def extract_features(y, sr):
    """Frame data and extract features for each frame."""
    
    # Calculate MFCCs
    mfccs = calculate_mfccs(y, sr)
    
    # Return feature vectors for each frame
    y_features = mfccs
       
    return y_features


def index2frame(time_indexes, sr):
    """Convert time log indexes to frame indexes."""

    # Calculate frame size and overlap in samples
    FRAME_LENGTH = sr * FRAME_DURATION // 1000
    HOP_LENGTH = FRAME_LENGTH * (100-FRAME_OVERLAP) // 100
    
    # Frame index of the last frame that the sample is in
    frame_indexes = np.apply_along_axis(librosa.samples_to_frames,
                                        axis=0,
                                        arr=time_indexes,
                                        hop_length=HOP_LENGTH,
                                        n_fft=FRAME_LENGTH)
    
    # Deal with negative indexes caused by librosa n_fft offset
    frame_indexes[frame_indexes<-0] = 0
    
    # Check
    for idx in frame_indexes.flatten():
        if idx < 0:
            raise ValueError('Negative frame index calculated during sample '
                             'index to frame index conversion.')
    
    return frame_indexes


def label_features(y_features, sr, logs):
    """Label feature vectors as 1 for 'whale' or 0 for 'background'."""
    
    time_indexes = logs[['Beg File Samp (samples)',
                          'End File Samp (samples)']].to_numpy() - 1
    
    
    # Convert time indexes to frame indexes
    frame_indexes = index2frame(time_indexes, sr)
    
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


def extract_samples(df_data_summary, df_log_filenames):
    """Extract labelled samples from .wav files."""
    
    sample_set = []
    
    for site in df_data_summary.index:
        
        logs = []
        
        for call_type in df_data_summary.columns:
            
            df_log = read_log(site, call_type, df_log_filenames)
            
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
            y, sr = read_audio(site, wavfile, df_log_filenames)
            
            # Frame and extract features
            y_features = extract_features(y, sr)
            
            # Label features
            y_labelled_features = label_features(y_features, sr, logs)
            
            # add samples to sample set
            sample_set.extend(y_labelled_features)
    
    # Create sample vector
    samples = np.vstack(sample_set)
    
    # Balance and randomise samples
    balanced_samples = balance_dataset(samples)
    
    # Split sample vector
    X, y = split_sample_vector(balanced_samples)
    
    return X, y


def get_training_set(df_annotations, df_log_filenames):
    """Extract training samples."""
    
    # Training set summary
    df_trainset = df_annotations.iloc[TRAINING_SITES, TRAINING_CALL_TYPES]
    #df_trainset.loc['Total'] = df_trainset.sum(numeric_only=True, axis=0)
    
    # Raise error if no annotations exist
    if not df_trainset.any(axis=None):
        sites = df_annotations.index[TRAINING_SITES].to_list()
        calls = df_annotations.columns[TRAINING_CALL_TYPES].to_list()
        raise ValueError('Chosen sites and call-types '
                          'contain zero annotations.', sites, calls)
    
    # Print summary
    print('\n' + '-'*50 + '\nTraining Set:\n' + '-'*50 + f'\n{df_trainset}')
    
    # Extract samples to use for training
    X_train, y_train = extract_samples(df_trainset, df_log_filenames)
    
    return X_train, y_train
    
    
def get_test_set(df_annotations, df_log_filenames):
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
        
    # Print testset summary
    print('\n' + '-'*50 + '\nTest Set:\n' + '-'*50 + f'\n{df_testset}')
    
    # Extract samples to use for testing
    X_test, y_test = extract_samples(df_testset, df_log_filenames)
    
    return X_test, y_test


def train_classifier(X_train, y_train):
    """Train classifier."""
    
    clf = GradientBoostingClassifier().fit(X_train, y_train)
   
    return clf


def get_results(clf, X_train, y_train, X_test, y_test):
    """Calculate and print classification results."""
    
    # Accuracy
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print('\n' + '-'*40 + '\nClassifier Accuracy:\n' + '-'*40)
    print(f'\nTraining: {train_score}\nTesting: {test_score}')
    
    # Confusion matrix
    tn, fp, fn, tp = calculate_confusion_matrix(X_test, y_test, clf)
    
    
def calculate_confusion_matrix(X_test, y_test, clf):
    """Caclulate and print confusion matrix."""
    
    y_pred = clf.predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred)
    
    # Printout
    ref = np.array([['TN', 'FP'], ['FN', 'TP']])
                  
    print('\n' + '-'*40 + '\nConfusion Matrix:\n' + '-'*40 + 
          f'\n{ref[0]}' + '-'*3 + f'{c_matrix[0]}' + 
          f'\n{ref[1]}' + '-'*3 + f'{c_matrix[1]}')
        
    tn, fp, fn, tp = c_matrix.ravel()
    
    return tn, fp, fn, tp


def main():
    # Start of script
    print('-'*40 + f'\nRunning {os.path.basename(__file__)}\n' + '-'*40 + '\n')
    print('An annotated library of Antarctic Blue and Fin Whale sounds.\n')

    # Get log names
    df_log_filenames = get_log_filenames()
    
    # Print sites
    print_recording_sites(df_log_filenames)
    
    # Print call types
    print_call_types()
 
    # Get total annotation count
    df_annotations = get_total_annotation_count(df_log_filenames)
    
    # Select training set
    X_train, y_train = get_training_set(df_annotations, df_log_filenames)
    
    # Select test set
    X_test, y_test = get_test_set(df_annotations, df_log_filenames)
    
    # Train model
    clf = train_classifier(X_train, y_train)
    
    # Print results
    get_results(clf, X_train, y_train, X_test, y_test)

    # End of script
    print('\nEnd' + '-'*40)
    
    
if __name__ == '__main__':
    main()