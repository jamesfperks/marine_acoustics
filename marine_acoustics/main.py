"""
Automated detection of Antarctic Blue and Fin Whale sounds using 
the Australian Antarctic Data Centre Annotated Library.

Author: James Perks
Email: jamesperks@outlook.com

"""


import time
import librosa
import pywt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

from marine_acoustics import intro, info
from marine_acoustics import settings as s


def read_log(site, call_type, df_folder_structure):
    """Read log .txt file into a dataframe."""
    
    log_filepath = info.get_log_filepath(site, call_type, df_folder_structure)
    
    
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
    wav_filepath = s.DATA_FILEPATH + '/' + site_folder + '/wav/' + wav_filename
    
    # Read entire mono .wav file and resample to preset global sample rate
    y, sr = librosa.load(wav_filepath, sr=s.SR)
    
    # Normalise to [-1, 1]
    y = librosa.util.normalize(y)
    
    # Store the default sample rate
    sr_default = librosa.get_samplerate(wav_filepath)
    
    return y, sr_default


def calculate_mfccs(y):
    """Split data into DFT windows and compute MFCCs for each window."""
    
    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(y=y,
                                 sr=s.SR,
                                 n_mfcc=s.N_MFCC,
                                 n_fft=s.FRAME_LENGTH,
                                 hop_length=s.HOP_LENGTH,
                                 n_mels=s.N_MELS,
                                 fmin=s.FMIN,
                                 fmax=s.FMAX).T
    
    return mfccs
    

def calculate_stft(y):
    """Compute STFT and split data into frames."""
    
    # STFT of y
    D = librosa.stft(y, n_fft=s.FRAME_LENGTH, hop_length=s.HOP_LENGTH)
    
    # STFT in dB
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max).T
    
    # Select frequency bin range for STFT
    stft_fmin_idx, stft_fmax_idx = get_stft_freq_range()
    S_db = S_db[:,stft_fmin_idx:stft_fmax_idx+1]
           
    return S_db


def get_stft_freq_range():
    """Return indexes of STFT frequency bins which correspond to FMIN, FMAX"""
    
    # Calculate frequency bins
    freqs = librosa.fft_frequencies(sr=s.SR, n_fft=s.FRAME_LENGTH)
    
    # Find index of closest frequency bin to FMIN, FMAX
    stft_fmin_idx = np.argmin(np.abs(freqs-s.FMIN))
    stft_fmax_idx = np.argmin(np.abs(freqs-s.FMAX))
    
    return stft_fmin_idx, stft_fmax_idx


def calculate_melspectrogram(y):
    """Compute the mel-spectrogram and return a vector for each frame."""
    
    # mel-power spectrogram of y
    D = librosa.feature.melspectrogram(y=y,
                                       sr=s.SR,
                                       n_fft=s.FRAME_LENGTH,
                                       hop_length=s.HOP_LENGTH,
                                       n_mels=s.N_MELS,
                                       fmin=s.FMIN,
                                       fmax=s.FMAX)
    
    
    # mel-power spectrogram in dB
    S_db = librosa.power_to_db(D, ref=np.max).T
    
    return S_db


def calculate_cwt(y):
    """Compute the cwt and return a vector for each frame."""
    
    # Choose wavelet pseudo frequencies
    desired_freqs = np.arange(s.FMIN, s.FMAX+1, 1)
    scales = frequency2scale(desired_freqs)
    
    # Compute continuous wavelet transform
    wavelet_coeffs, wavelet_freqs = apply_cwt(y, scales)
    
    cwt = frame_data(wavelet_coeffs.T)
    
    cwt = np.mean(cwt, axis=1)
    
    return cwt


def frame_data(data):
    """
    Slice 1D array into frames with a given overlap: (n_frames x frame_length)
    """
    
    frame_view = librosa.util.frame(data,
                                    frame_length=s.FRAME_LENGTH,
                                    hop_length=s.HOP_LENGTH,
                                    axis=0)
    
    return frame_view


def apply_cwt(y, scales):
    """Apply cwt to a 1D array"""
    
    # Compute continuous wavelet transform
    wavelet_coeffs, wavelet_freqs = pywt.cwt(y,
                                             scales,
                                             s.WAVELET,
                                             sampling_period=1/s.SR)

    return wavelet_coeffs, wavelet_freqs
    

def scale2frequency(scales):
    """Convert from cwt scale to to pseudo-frequency"""

    # pywt function returns normalised frequency so need to multiply by sr
    freqs = pywt.scale2frequency(s.WAVELET, scales) * s.SR

    return freqs


def frequency2scale(desired_freqs):
    """Convert from desired frequencies to a cwt scale"""

    # pywt function input is normalised frequency so need to normalise by sr
    normalised_freqs = desired_freqs / s.SR
    
    freqs = pywt.scale2frequency(s.WAVELET, normalised_freqs)

    return freqs


def extract_features(y):
    """Frame data and extract features for each frame. (FRAMES X s.FEATURES)"""
    

    if s.FEATURES == 'MFCC':
        y_features = calculate_mfccs(y)   # Calculate MFCCs
        
    elif s.FEATURES == 'STFT':
        y_features = calculate_stft(y)    # Calculate STFT
        
    elif s.FEATURES == 'MEL':
        y_features = calculate_melspectrogram(y)  # Calculate mel-spectrogram
        
    elif s.FEATURES == 'CWT':
        y_features = calculate_cwt(y)   # Calculate cwt
    
    else:
        raise NotImplementedError('Feature representation chosen ' 
                                        'is not implemented', s.FEATURES)
       
    return y_features


def index2frame(time_indexes):
    """Convert time log indexes to frame indexes."""
    
    # Frame index of the last frame that the sample is in
    frame_indexes = np.apply_along_axis(librosa.samples_to_frames,
                                        axis=0,
                                        arr=time_indexes,
                                        hop_length=s.HOP_LENGTH,
                                        n_fft=s.FRAME_LENGTH)
    
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
    
    time_indexes = np.rint(s.SR*unsampled_indexes/sr_default)
    
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
    print(f'\nNumber of whale call samples: {whale_samples.shape}\nNumber of background samples: {background_samples.shape}\n')
    
    # Randomise sample order
    np.random.seed(s.SEED)
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
    print('\n',balanced_samples.shape, '\n')
    
    # Split sample vector
    X, y = split_sample_vector(balanced_samples)
    
    return X, y


def select_training_set(df_annotations):
    """Select sites and call types to use for training."""
    
    # Training set annotation summary
    df_trainset = df_annotations.iloc[s.TRAINING_SITES, s.TRAINING_CALL_TYPES]
    
    # Raise error if no annotations exist
    if not df_trainset.any(axis=None):
        sites = df_annotations.index[s.TRAINING_SITES].to_list()
        calls = df_annotations.columns[s.TRAINING_CALL_TYPES].to_list()
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
    
    test_sites = s.TEST_SITES
    test_call_types = s.TEST_CALL_TYPES
    
    # Default to using all non-training sites if unspecified []
    if len(test_sites) == 0:
        test_sites = list(range(0, 11))  
        for site_idx in s.TRAINING_SITES:   
            test_sites.remove(site_idx)
    
    # Default to using training call types if unspecified []
    if len(test_call_types) == 0:
        test_call_types = s.TRAINING_CALL_TYPES
            
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
    
    # Train/test header
    print('\n'*2 + '-'*50 + '\nTRAINING AND TEST SET SELECTION\n' + '-'*50)
    
    # Print training set summary
    print('\n' + f'\nTraining set: ({train_tot})\n'
          + '-'*30 + f'\n{df_trainset}')
    
    # Print testset summary
    print('\n'*2 + f'\nTest set: ({test_tot})\n'
          + '-'*30 + f'\n{df_testset}')
    
    # Dataset summary header
    print('\n'*2 + '-'*50 + '\nDATASET SUMMARY\n' + '-'*50)
    
    # Train/test ratio printout
    print('\n'*2 + '  - Percentage split train/test is ' + 
          f'{train_percent}/{test_percent}.\n')
    
    # Feature extraction method
    print(f'  - Feature extraction method: {s.FEATURES}')
    
    
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
    
    # Start script
    intro.print_introduction()
    
    # Get data folder structure
    df_folder_structure = info.get_folder_structure()
    
    # Count total annotations
    df_annotations = info.get_total_annotation_count(df_folder_structure)
    
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
    
    # Run and time script
    start = time.time()
    run()
    end = time.time()
    
    # End of script
    print('\n'*2 + f'Total runtime: {end-start:0.1f} seconds.\n' + '-'*47 + 'End')
          
    
if __name__ == '__main__':
    main()
    