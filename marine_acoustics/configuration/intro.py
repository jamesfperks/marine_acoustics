"""
Begin the script and print an introduciton.

"""


from marine_acoustics.configuration import settings as s
from marine_acoustics.data_processing import info



def print_introduction():
    """Print the introduction."""
    
    # Project overview
    print_project_info()
    
    # Dataset overview
    print_dataset_info()
    
    # Settings overview
    print_settings_info()


def print_project_info():
    """Print project title and description."""
    
    # Script starting header
    print('\n'*2 + '-'*s.HEADER_LEN + 
          '\nMARINE BIOACOUSTICS: 4TH YEAR ENGINEERING PROJECT\n' +
          '-'*s.HEADER_LEN)
    
    # Print project description
    print('\nAn applied machine learning project to detect\n'
          'Antarctic Blue and Fin Whale sounds.')
    

def print_dataset_info():
    """Print description of dataset used."""
    
    # Print header
    print('\n' + '-'*s.HEADER_LEN + '\nDATASET SUMMARY\n' + '-'*s.HEADER_LEN)
    
    # Print all recording sites
    print_recording_sites()
    
    # Print all call types
    print_call_types()
    
  
def print_recording_sites():
    """Print all recording sites and the corresponding index."""
    
    sites = info.get_recording_sites()
    
    print('\nRecording sites:\n' + '-'*s.SUBHEADER_LEN)
    
    for i in range(len(sites)):
        print(f' {i+1}' + ' '*(4-len(str(i+1))) + sites[i])

 
def print_call_types():
    """Print all call types and the corresponding index."""

    call_types = info.get_call_types()
    
    print('\nCall types:\n' + '-'*s.SUBHEADER_LEN)
    
    for i in range(len(call_types)):
        print(f' {i+1}  ' + call_types[i])
  

def print_settings_info():
    """Print overview of main settings."""
    
    # Print header
    print('\n' + '-'*s.HEADER_LEN + '\nSETTINGS OVERVIEW\n' + '-'*s.HEADER_LEN)
    
    
    # Classifier
    print(f'\n  - Classifier: {s.MODEL}')
    
    # Feature extraction method
    print(f'\n  - Feature extraction: {s.FEATURES}', end='')
    
    