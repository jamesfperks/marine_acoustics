"""
Begin the script and print an introduciton.

"""


from marine_acoustics import info
from marine_acoustics import settings as s


def print_introduction():
    """Print the introduction."""
    
    print('\nAn automated detector of Antarctic Blue and Fin Whale sounds.')
    
    # Print all recording sites
    print_recording_sites()
    
    # Print all call types
    print_call_types()
    

def print_recording_sites():
    """Print all recording sites and the corresponding index."""
    
    sites = info.get_recording_sites()
    
    print('\nRecording sites:\n' + '-'*s.SUBHEADER_LEN)
    
    for i in range(len(sites)):
        print(f' {i}' + ' '*(4-len(str(i))) + sites[i])

 
def print_call_types():
    """Print all call types and the corresponding index."""

    call_types = info.get_call_types()
    
    print('\nCall types:\n' + '-'*s.SUBHEADER_LEN)
    
    for i in range(len(call_types)):
        print(f' {i}  ' + call_types[i])
  
