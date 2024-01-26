"""
Retrieve summary information from the Australian Antarctic Data Centre 
Annotated Library of Antarctic Blue and Fin Whale sounds.

"""


import pandas as pd
from marine_acoustics import settings as s
from marine_acoustics import read

def get_folder_structure():
    """Read the folder structure csv file and save as a pd dataframe."""
    
    csv_filepath = s.DATA_FILEPATH + '/01-Documentation/folderStructure.csv'
    df = pd.read_csv(csv_filepath, index_col=0)
    df.index.name = None
      
    return df


def get_recording_sites():
    """Return a list of all recording sites."""
    
    recording_sites = ['Maud Rise 2014', 'G64 2015', 'S Kerguelen 2005',
                       'S Kerguelen 2014', 'S Kerguelen 2015', 'Casey 2014',
                       'Casey 2017', 'Ross sea 2014', 'Balleny Islands 2015',
                       'Elephant Island 2013', 'Elephant Island 2014']
    
    return recording_sites


def get_call_types():
    """Return a list of all call types."""
    
    call_types = ['Bm-A', 'Bm-B', 'Bm-Z', 'Bm-D',
                       'Bp-20', 'Bp-20+', 'Bp-Downsweep', 'Unidentified']
    
    return call_types


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
        

def count_annotations(df_folder_structure, site, call_type):
    """Count the number of call annotations for a given site and call type."""
    
    log_filepath = read.get_log_filepath(site, call_type, df_folder_structure)
    
    with open(log_filepath, "rb") as f:
        n_annotations = sum(1 for line in f) - 1

    return n_annotations

