import glob
import os
import librosa
import numpy as np
import scipy.io.wavfile
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from playsound import playsound


"""Detect Southern Right Whale Vocalisations."""


# CONSTANTS
# ----------------------------------------------------------------------------
DATA_FILEPATH = 'data/Moby_non_toothed/SouthernRightWhale001-v1'
FILE_SPLIT = (3,1,1)    # Number of train/val/test files
SR = 8000               # Sample rate in samples/sec
FRAME_DURATION = 100    # Frame duration in milliseconds
FRAME_OVERLAP = 50      # Frame overlap (%)
N_MFCC = 3              # no. of mfccs to calculate
N_MELS = 64             # no. Mel bands used in mfcc calc (default 128)


# ----------------------------------------------------------------------------
# RUNTIME CONSTANTS (DO NOT CHANGE)
FRAME_LENGTH = SR * FRAME_DURATION // 1000   # frame size in samples
HOP_LENGTH = FRAME_LENGTH * (100-FRAME_OVERLAP) // 100   # stride in samples
# ----------------------------------------------------------------------------


def get_files():
    """
    Find all .wav and log files and return as a list of tuples:
        
    Returns: [(.log1, .wav1), (.log2, .wav2), ...]
    
    """
    
    # List all .log files in cwd
    log_files = glob.glob('*.log', root_dir=DATA_FILEPATH)
    
    # List all .wav files in cwd
    wav_files = glob.glob('*.wav', root_dir=DATA_FILEPATH)
    
    # Check files are valid for classification
    test_files(log_files, wav_files)
    
    
    files = list(zip(log_files, wav_files))
    data_dir = DATA_FILEPATH.split('/')[-1]
    
    print(f'Data directory: {data_dir}\n')
    print(f'Files retrieved: {len(files)} pairs of .wav and log files\n')
    
    return files


def test_files(log_files, wav_files):
    """Check files are valid for classification"""
    
    # Check for matching number of log and .wav pairs
    if len(log_files) != len(wav_files):
        raise FileNotFoundError('Mismatched number of .wav and log files.')
    
    files = list(zip(log_files, wav_files))
    for log, wav in files:
        
        # Check names match
        if log[:-4] != wav[:-4]:
            raise NameError('Name of log file and .wav file do not match:\n' +
                            f'{log}, {wav}')
            
        # Test log times are sorted in order with no overlap
        time_log = get_time_log(log)
        time_indexes = time2index(time_log).flatten()
        if np.array_equal(np.sort(time_indexes), time_indexes) == False:    
            raise IndexError(f'Log file: {log}\n Time values overlap.')
     

def split_files(files):
    """Split files for train, val, test"""
    
    i = FILE_SPLIT[0]
    j = i + FILE_SPLIT[1]
    
    # Files used for train/val/test 
    train_files = files[0:i]
    val_files = files[i:j]
    test_files = files[j:]
    
    print('Training files:')
    print_files(train_files)
    print('\nValidation files:')
    print_files(val_files)
    print('\nTest files:')
    print_files(test_files)
    
    return train_files, val_files, test_files


def print_files(fileset):
    """Print all .wav files from a list of tuples: (log, .wav)"""

    wav_files = [wav for log, wav in fileset]
    for file in wav_files:
        print('   - ' + file)
        

def extract_samples(fileset):
    """Extract feature vector X and labels y from a given fileset"""
    
    
    """
    Test files results:
        
    Using slice_data to generate frames gives
    Whale frames: (505, 800)
    Noise frames: (6430, 800)
    
    Feature vector Should have same no. of frames but with row size of N_MFCC.
    Instead, gives:
        
    Whale features: (541, 3)
    Noise features: (6466, 3)
    
    CHECK WHY THIS IS...
    
    CHECK THAT ZERO SECTION DATA IS FIXED...
    IMPORTANT THIS MAY HAVE CHANGED THE ORDER OF WHALE/NOISE SECTIONS
    THE ZERO ARRAY IN TRACK 7 APPEARS IN THE NOISE SECTIONS....
    i.e. the fact that some sections are empty lists [] due to dodgy
    log values does not matter
    
    """
    
    # Combine sections from all recordings
    all_whale_sections = []
    all_noise_sections = []
    
    for log_filename, wav_filename in fileset:
        whale_sections, noise_sections = separate_class_data(log_filename,
                                                             wav_filename)
        
        all_whale_sections.extend(whale_sections)
        all_noise_sections.extend(noise_sections)
      
    
    # Extract features
    all_whale_features = np.vstack(tuple(extract_features(section) for
                                         section in all_whale_sections
                                         if section.size > 0))
    
    all_noise_features = np.vstack(tuple(extract_features(section) for
                                         section in all_noise_sections
                                         if section.size > 0))
    
    
    #all_whale_frames = np.vstack(tuple(slice_data(section) for section in all_whale_sections if section.size > 0))
    #all_noise_frames = np.vstack(tuple(slice_data(section) for section in all_noise_sections if section.size > 0))
    
    #print(all_whale_features.shape)
    #print(all_noise_features.shape)
    #print(all_whale_frames.shape)
    #print(all_noise_frames.shape)
    
    labelled_whale_features = np.c_[all_whale_features, np.ones(all_whale_features.shape[0])]
    labelled_noise_features = np.c_[all_noise_features, np.zeros(all_noise_features.shape[0])]
    
    all_samples = np.vstack((labelled_whale_features, labelled_noise_features))
    
    np.random.shuffle(all_samples)
    
    X = all_samples[:,0:N_MFCC]
    
    y = all_samples[:,-1]

    
    """
    LEGACY
    # Format: [(["whale1"], ["noise1"]), (["whale2"], ["noise2"]), ...]
    all_frames = [separate_class_data(log_filename, wav_filename) for
                  log_filename, wav_filename in fileset]
    
    # Format: [(["whale1"], ["whale2"], ...), (["noise1"], ["noise2"], ...)]
    all_frames_by_class = list(zip(*all_frames))
    
    
    all_whale_frames = np.vstack(all_frames_by_class[0])
    all_noise_frames = np.vstack(all_frames_by_class[1])
    
    print(all_whale_frames.shape)
    print(all_noise_frames.shape)
    """
    
    return X, y


def separate_class_data(log_filename, wav_filename):
    """Extract "whale" and "noise" sections from a given log and .wav file"""
    
    # Read in log and audio
    time_log = get_time_log(log_filename)
    y = read_audio(wav_filename)
    
    # Separate into "whale" and "noise" sections
    time_log_indexes = time2index(time_log)
    whale_sections, noise_sections = extract_labelled_sections(y, time_log_indexes)
    
    # Create "whale" and "noise" frames (legacy)
    #whale_frames = generate_frames(whale_sections)
    #noise_frames = generate_frames(noise_sections)
    
    return whale_sections, noise_sections


def extract_labelled_sections(y, time_log_indexes):
    """Extract sections of "whale" and "noise" given time-indexes"""
    
    # Split samples into alternating sections.
    # Format: [["noise"], ["whale"], ["noise"], ["whale"]...]
    sections = np.split(y, time_log_indexes.flatten())
    
    
    # Separate into whale and noise sections
    whale_sections = sections[1::2]
    noise_sections = sections[::2]
    
    return whale_sections, noise_sections


def extract_features(data):
    """Extarct feature vector given a 1D array of data"""
    
    feature_vector = calculate_mfccs(data)
    
    return feature_vector


def calculate_mfccs(y):
    # Calculate MFCCs
    mfccs = np.transpose(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH,
                                 hop_length=HOP_LENGTH, n_mels=N_MELS))
    
    return mfccs


def get_time_log(log_filename):
    """Read log file with columns: start-time, end-time"""
    
    log_filepath = DATA_FILEPATH + '/' + log_filename
    time_log = np.loadtxt(log_filepath, usecols=(0,1))
    
    return time_log


def time2index(time_log):
    """
    Calculate start-index, end-index given start-time, end-time in seconds.
    
    Format: start index (inclusive), end index (exclusive).
    
    
    Note:
    
    Start index is the index of the sample where "whale" begins (inclusive)
    (i.e start index sample should be included in the "whale" slice).
    
    End index is the index of the sample where "whale" ends (exclusive)
    (i.e end index sample should be included in the next "noise" slice).
    
    """
    #print(np.sum(time_log[:,1]-time_log[:,0]))
    # Convert seconds to sample number
    time_log_samples = time_log * SR
    
    # Round down to nearest sample for start sample no. of "whale" slice.
    # Index of this sample is one less than sample no.
    start_indexes = np.floor(time_log_samples[:,0]).astype(int) - 1
    
    
    # Round up to nearest sample for end sample no. of "whale" slice.
    # Add one for sample no. where "noise" begins
    # Index is one less than sample no. so net zero
    end_indexes = np.ceil(time_log_samples[:,1]).astype(int)
    
    time_log_indexes = np.column_stack((start_indexes, end_indexes))
    
    return time_log_indexes


def read_audio(wav_filename):
    
    # File path to .wav file
    audio_file_path = DATA_FILEPATH + '/' + wav_filename
    #filename = audio_file_path.split('/')[-1]
    
    # Read entire mono .wav file using default sampling rate
    y, sr = librosa.load(audio_file_path, sr=None, duration=None)
    #print(f'\nLoaded file: {filename}\n' + '-'*40 + '\n')
    #print(f'Duration: {y.size/sr} seconds\n' + '-'*40 + '\n')
    #print(f'Sample rate: {sr} Hz\n' + '-'*40)
    return y


def generate_frames(audio_sections):    # Legacy?
    """
    Create frames given a series of continuous audio sections
    
    Returns: 2D array of shape (no_frames, FRAME_LENGTH)
    
    LEGACY FUNCTION?
    """
    
    # Check if y.size > 0 to prevent error from slicing empty frames
    frames_tuple = tuple(slice_data(y) for y in audio_sections if y.size > 0)
    frames = np.vstack(frames_tuple)
    
    return frames


def slice_data(data):    # Legacy?
    """Slice 1D array into frames with a given overlap"""
    
    frame_view = librosa.util.frame(data, frame_length=FRAME_LENGTH,
                                    hop_length=HOP_LENGTH, axis=0)
    
    return frame_view


def play_audio(y):
    """Play audio given sample data and sampling rate"""
    
    # Possibly use playsound==1.2.2 to resolve relative path issue?
    
    scipy.io.wavfile.write('sample.wav', SR, y)
    sample_audio_filepath = os.getcwd() + '\sample.wav'
    
    print('\nPlaying sample .wav file...')
    playsound(sample_audio_filepath)
    print('\nAudio finished.\n')
   

def train_classifier(X_train, y_train):
    """Train classifier.
    
    Note from sklearn:
    All decision trees use np.float32 arrays internally.
    If training data is not in this format, a copy of the dataset will be made.
    """
    
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1,
                                     random_state=0).fit(X_train, y_train)
    
    return clf


def calculate_all_scores(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    
    train_score = clf.score(X_train, y_train)
    val_score = clf.score(X_val, y_val)
    test_score = clf.score(X_test, y_test)
    
    
    print('\n' + '-'*40 + '\nClassifier Accuracy:\n' + '-'*40)
    print(f'\nTraining: {train_score}\n\nValidation: {val_score}\n\n'
          f'Testing: {test_score}')
    
    
def calculate_confusion_matrix(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred)
    print('\n' + '-'*40 + '\nConfusion Matrix:\n' + '-'*40)
    print(c_matrix)
    
    tn, fp, fn, tp = c_matrix.ravel()
    
    return tn, fp, fn, tp


def main():

    # Start of script
    print('-'*40 + f'\nRunning {os.path.basename(__file__)}\n' + '-'*40 + '\n')


    # Read in files
    files = get_files()
    train_files, val_files, test_files = split_files(files)


    # Create train/val/test samples
    X_train, y_train = extract_samples(train_files)
    X_val, y_val = extract_samples(val_files)
    X_test, y_test = extract_samples(test_files)


    # Train model
    clf = train_classifier(X_train, y_train)
    
    
    # Predict
    y_test_pred = clf.predict(X_test)

    # Results
    calculate_all_scores(clf, X_train, y_train, X_val, y_val, X_test, y_test)
    tn, fp, fn, tp = calculate_confusion_matrix(y_test, y_test_pred)
    

    # End of script
    print('\nEnd' + '-'*40)
    


if __name__ == '__main__':
    main()



 