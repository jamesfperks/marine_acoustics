import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from playsound import playsound


"""Detect Southern Right Whale Vocalisations."""


# CONSTANTS
# ----------------------------------------------------------------------------
DATA_FILEPATH = 'data/Moby_non_toothed/SouthernRightWhale001-v1'
FILE_SPLIT = (3,1,1)    # Number of train/val/test files
SR = 8000               # Sample rate in samples/sec
FRAME_DURATION = 100    # Frame duration in milliseconds
FRAME_OVERLAP = 50      # Frame overlap (%)
N_MFCC = 12             # no. of mfccs to calculate
N_MELS = 64             # no. Mel bands used in mfcc calc (default 128)
SEED = 12345            # Set random seed
MIN_SAMPLES = 300       # Set minimum no. of class samples in a fileset

# ----------------------------------------------------------------------------
# RUNTIME CONSTANTS (DO NOT CHANGE)
FRAME_LENGTH = SR * FRAME_DURATION // 1000   # frame size in samples
HOP_LENGTH = FRAME_LENGTH * (100-FRAME_OVERLAP) // 100 # stride in samples
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
        

def extract_samples(fileset, is_balanced=True):
    """Extract feature vector X and labels y from a given fileset"""
    
    
    all_whale_features = []
    all_noise_features = []
    for log_filename, wav_filename in fileset:
    
        # Read in log and audio
        time_log = get_time_log(log_filename)
        y = read_audio(wav_filename)
        
        
        # Calculate features for entire recording
        y_features = extract_features(y)
        
        
        # Split features into whale and noise
        whale_features, noise_features = separate_features(y_features,
                                                           time_log)
        
        # Combine whale and noise features for all recordings
        all_whale_features.append(whale_features)
        all_noise_features.append(noise_features)
        
    
    # Concatenate features from all recordings
    all_whale_features = np.concatenate(all_whale_features)
    all_noise_features = np.concatenate(all_noise_features)
    
    
    # Balance dataset
    all_noise_features = balance_dataset(all_whale_features,
                                         all_noise_features,
                                         is_balanced)
    
    
    # Label samples
    labelled_samples = label_samples(all_whale_features, all_noise_features)
    
    # Collect and randomise sample set
    X, y = get_sample_vectors(labelled_samples)

    return X, y


def separate_features(features, time_log):
    """Separate features into whale and noise features."""
    
    
    # Convert times to sample indexes
    time_log_indexes = time2index(time_log)
    
    # Convert to frame indexes
    frame_indexes = index2frame(time_log_indexes)
    
    # Extract labelled features
    whale_features, noise_features = extract_labelled_features(features,
                                                               frame_indexes)
    
    return whale_features, noise_features 


def index2frame(time_log_indexes):
    """Convert time log indexes to frame indexes."""

    frame_indexes = np.apply_along_axis(librosa.samples_to_frames, 0,
                                        time_log_indexes,
                                        hop_length=HOP_LENGTH,
                                        n_fft=FRAME_LENGTH)
    
    for idx in frame_indexes.flatten():
        if idx < 0:
            raise ValueError('Negative frame index calculated during sample '
                             'index to frame index conversion.')
    
    return frame_indexes


def extract_labelled_features(features, frame_indexes):
    """Extract "whale" and "noise" features given frame-indexes"""
    
    # Split features into alternating sections.
    # Format: [["noise"], ["whale"], ["noise"], ["whale"]...]
    sections = np.split(features, frame_indexes.flatten())
    
    # Separate into whale and noise features
    whale_features = np.concatenate(sections[1::2])
    noise_features = np.concatenate(sections[::2])
    
    return whale_features, noise_features


def extract_features(data):
    """Extarct feature vector given a 1D array of data"""
    
    feature_vector = calculate_mfccs(data)
    
    return feature_vector


def calculate_mfccs(y):
    # Calculate MFCCs
    mfccs = np.transpose(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH,
                                 hop_length=HOP_LENGTH, n_mels=N_MELS))
    
    return mfccs


def balance_dataset(whale_features, noise_features, is_balanced=True):
    """Sub-sample the majority class to balance the dataset."""
    
    if is_balanced == True:
        no_whale_samples = whale_features.shape[0]
        
        np.random.shuffle(noise_features)
        
        noise_features = noise_features[0:no_whale_samples, :]
  
    
    return noise_features


def label_samples(whale_features, noise_features):
    """Label sample set. Whale = 1. Noise = 0."""
    
    labelled_whale_features = np.c_[whale_features,
                                    np.ones(whale_features.shape[0])]
    
    labelled_noise_features = np.c_[noise_features,
                                    np.zeros(noise_features.shape[0])]
    
    labelled_samples = np.vstack((labelled_whale_features,
                                  labelled_noise_features))
    
    return labelled_samples


def get_sample_vectors(samples):
    """Create sample vectors X, y."""
    
    np.random.shuffle(samples)
    
    feature_vector_len = N_MFCC
    
    X = samples[:,0:feature_vector_len]
    y = samples[:,-1]
    
    return X, y


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
    
    # Read entire mono .wav file using default sampling rate
    y, sr = librosa.load(audio_file_path, sr=None, duration=None)
    #print(f'\nLoaded file: {wav_filename}\n' + '-'*40 + '\n')
    #print(f'Duration: {y.size/sr} seconds\n' + '-'*40 + '\n')
    #print(f'Sample rate: {sr} Hz\n' + '-'*40)
    
    # Normalise to [-1, 1]
    y_norm = librosa.util.normalize(y)
    
    return y_norm
    """Slice 1D array into frames with a given overlap"""
    
    frame_view = librosa.util.frame(data, frame_length=FRAME_LENGTH,
                                    hop_length=HOP_LENGTH, axis=0)
    
    return frame_view


def play_audio(y):
    """Play audio given sample data and sampling rate"""
    
    # Possibly use playsound==1.2.2 to resolve relative path issue?
    
    scipy.io.wavfile.write('sample.wav', SR, y)
    sample_audio_filepath = os.getcwd() + '/scripts/sample.wav'
    
    print('\nPlaying sample .wav file...')
    playsound(sample_audio_filepath)
    print('\nAudio finished.\n')
   

def plot_waveform(y, axis='s', offset=0.0, title='Audio Waveform',
                  xlabel='Time (s)', ylabel='Amplitude'):
    """Plot the signal waveform in the time domain"""

    # Plot
    plt.figure()
    librosa.display.waveshow(y, sr=SR, max_points=SR//2, axis=axis, offset=offset)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def train_classifier(X_train, y_train):
    """Train classifier.
    
    Note from sklearn:
    All decision trees use np.float32 arrays internally.
    If training data is not in this format, a copy of the dataset will be made.
    """
    
    parameters = {
    "learning_rate": [0.001, 0.01, 0.1, 1],
    "max_depth":[1,2,3],
    "n_estimators":[10, 100, 1000]
    }
    
    #clf = GridSearchCV(GradientBoostingClassifier(), parameters, n_jobs=-1)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=2,
                                     random_state=SEED).fit(X_train, y_train)
    #clf.fit(X_train, y_train)
    #print(clf.best_params_)
    
    return clf


def calculate_all_scores(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    """Calculate train, val, test scores and print results."""
    
    train_score = clf.score(X_train, y_train)
    val_score = clf.score(X_val, y_val)
    test_score = clf.score(X_test, y_test)
    
    
    print('\n' + '-'*40 + '\nClassifier Accuracy:\n' + '-'*40)
    print(f'\nTraining: {train_score}\n\nValidation: {val_score}\n\n'
          f'Testing: {test_score}')
    
    
def calculate_confusion_matrix(y_true, y_pred):
    """Caclulate and print confusion matrix."""
    
    c_matrix = confusion_matrix(y_true, y_pred)
    ref = np.array([['TN', 'FP'], ['FN', 'TP']])
                    
    print('\n' + '-'*40 + '\nConfusion Matrix:\n' + '-'*40 + 
          f'\n{ref[0]}' + '-'*3 + f'{c_matrix[0]}' + 
          f'\n{ref[1]}' + '-'*3 + f'{c_matrix[1]}')
        
    tn, fp, fn, tp = c_matrix.ravel()
    
    return tn, fp, fn, tp


def get_classification_report(y_true, y_pred):
    """Print the scikit learn classification report."""
    
    targets = ['nosie', 'whale']
    
    print('\n' + '-'*40 + '\nClassification Report:\n' + '-'*40 + 
          f'\n{classification_report(y_true, y_pred,target_names=targets)}'
          '\nNB: For binary classification, recall of the positive class'
          ' is known as “sensitivity”; recall of the negative class is '
          '“specificity”.\n')
  

def plot_3_mfccs(wav_filename):
    """Plot first 3 mfccs for a given .wav file."""
    
    y = read_audio(wav_filename)
    mfccs = calculate_mfccs(y)
    
    # Plot
    x = np.arange(mfccs.shape[0])
    y1 = mfccs[:,0]
    y2 = mfccs[:,1]
    y3 = mfccs[:,2]
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    
    plt.title('First 3 MFCCs')
    plt.xlabel('Frame Number')
    plt.ylabel('MFCC Value')
    plt.legend(['1st Coeff', '2nd Coeff', '3rd Coeff'])
    


def main():

    # Start of script
    print('-'*40 + f'\nRunning {os.path.basename(__file__)}\n' + '-'*40 + '\n')
    plt.close()

    # Set random seed
    np.random.seed(seed=SEED)

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
    get_classification_report(y_test, y_test_pred)
    
    # Plot MFCCs
    plot_3_mfccs('sar98_trk1a_8000.wav')
    
    # End of script
    print('\nEnd' + '-'*40)
    
    
if __name__ == '__main__':
    main()



 