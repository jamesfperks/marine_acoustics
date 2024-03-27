"""
Utility functions to select call logs and audio
"""


from marine_acoustics.data_processing import info, read, sample, label
from marine_acoustics.configuration import settings as s


def get_call_audio(SITE, CALL_TYPE, WAVFILE, TIME, PADDING):
    """Return the raw audio of a selected call with padding either side."""

    site = info.get_recording_sites()[SITE-1]
    call_types = info.get_call_types()
    call_types = [call_types[CALL_TYPE-1]]
    wavfile = WAVFILE + '.wav'
    
    # Folder structure
    df_folder_structure = info.get_folder_structure()
    
    # Read audio
    y_raw, sr_default = read.read_audio(site, wavfile, df_folder_structure)
    
    # Get all site logs for the call type
    df_logs = sample.concat_call_logs(site, call_types, df_folder_structure)
    
    # Get log corresponding to selected call
    call_log = df_logs.loc[df_logs['Begin Date Time'] == TIME]
    
    # Get the sample indexes of start/end of whale call
    call_indexes = label.get_call_indexes(call_log, sr_default)[0]
    
    # Add padding to start and end of call
    start_idx = int(call_indexes[0] - PADDING*s.SR)
    end_idx = int(call_indexes[1] + PADDING*s.SR)
    
    # Select call audio to plot
    y = y_raw[start_idx:end_idx+1]

    return y


def get_call_annotations(SITE, CALLS):
    """Return groupby object of call annotations for a given site and list of
    call types."""

    # Get site and call types
    site = info.get_recording_sites()[SITE-1]
    call_types = info.get_call_types()
    call_types = [call_types[i-1] for i in CALLS]
    print(site, '\n', call_types)
    
    # Folder structure
    df_folder_structure = info.get_folder_structure()
    
    # Combine all call-type logs and select fields
    df_logs = sample.concat_call_logs(site, call_types, df_folder_structure)
    fields = ['Begin File', 'Beg File Samp (samples)',
              'End File Samp (samples)', 'Call Label', 'Begin Date Time',
              'Delta Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)',
              'Dur 90% (s)', 'Freq 5% (Hz)', 'Freq 95% (Hz)']
    df_logs = df_logs[fields]
    
    # Get default sample rate used at site
    _, sr_default = read.read_audio(site, df_logs['Begin File'].iloc[0],
                                    df_folder_structure)
    
    # Convert samples to time (s)
    df_logs = df_logs.rename(columns={'Beg File Samp (samples)': "Begin (s)",
                                      'End File Samp (samples)': "End (s)"})
    df_logs['Begin (s)'] = round(df_logs['Begin (s)']/sr_default,1)
    df_logs['End (s)'] = round(df_logs['End (s)']/sr_default,1)
    
    
    # Sort by annotation start time
    df_logs = df_logs.sort_values("Begin (s)")
    
    # Groupby .wav filename and display
    gb_wavfile = df_logs.groupby('Begin File')

    return gb_wavfile

