"""
Pipeline
"""

import numpy as np
from marine_acoustics.pipeline import config, process_audio
from marine_acoustics.configuration import settings as s

    
def main():
    
    # Setup
    config.pipeline_setup()
    gb_wavfile, site, df_folder_structure = config.get_groupby()
    model = config.load_model()
    
    # thresholds
    thresholds = np.linspace(1, 0, num=2000)
    
    # results per threshold (TP, FP, FN, TN)
    cmatrix = np.zeros((thresholds.size, 4))
    
    it = 0
    for wavfile, logs in gb_wavfile:
        
        y, y_proba, true_call_times = process_audio.read_wav(site, wavfile,
                                            logs, df_folder_structure, model)
        
        # FOR EACH THRESHOLD ---------------------------------------------
        
        for i in range(thresholds.size):
            threshold = thresholds[i]
            
            TP, FP, FN, TN = process_audio.get_cmatrix(y, y_proba,
                                        threshold, true_call_times)
            
            cmatrix[i,:] += TP, FP, FN, TN
          
        if it == 1:
           break
        else:
            it+=1
            
    # Save results


if __name__ == '__main__':
    main()
    