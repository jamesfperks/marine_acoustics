"""
Functions to read in the AADC dataset.

"""


from marine_acoustics import settings as s


def get_log_filepath(site, call_type, df_folder_structure):
    """Return filepath to a log file given a site name and call type."""
    
    log_header = call_type_2_log_header(call_type)
    rel_filepath = df_folder_structure.loc[site, ['Folder', log_header]].str.cat()
    log_filepath = s.DATA_FILEPATH + '/' + rel_filepath
    
    return log_filepath


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

