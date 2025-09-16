import pandas as pd
import os
import glob
import json
import numpy as np
from screening_sample_ssd import get_data_folders
def numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

def combine_peaks_data(root_path):
    # Base directory
    #base_dir = '/Volumes/WD/ladder/500mV_100bp_1ngmkl_10MHz_boost_240830164444/'
    #base_dir = '/Volumes/WD/ladder/500mV_500bp_1ngmkl_20MHz_boost_240830171659/'
    #base_dir = '/Volumes/WD/ladder/500mV_200bp_1ngmkl_10MHz_boost_240830165006/'
    #base_dir = '/Volumes/WD/NBX/Raw_data_20241125_WSU_6_b1_171/sample2_spikein_241125191717/'
    #base_dir = '/Volumes/WD/NBX/Raw_data_20241125_WSU_6_b1_171/sample2_spikein_5mhz_241125192921/'
    #base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_300mV_300pM_sample4_250113155944/' 
    #base_dir = '/Volumes/Nanopore/DATA/Database100-200-500/500mV_100bp_1ngmkl_10MHz_boost_240830164444/'
    #base_dir = '/Volumes/Nanopore/DATA/Database100-200-500/500mV_100bp_1ngmkl_20MHz_boost_240830164159/'
    #base_dir = '/Volumes/Nanopore/DATA/Database100-200-500/500mV_200bp_1ngmkl_10MHz_boost_240830165006/'
    #base_dir = '/Volumes/Nanopore/DATA/Database100-200-500/500mV_500bp_1ngmkl_10MHz_boost_240830171035/'
    #base_dir = '/Volumes/Nanopore/DATA/Database100-200-500/500mV_500bp_1ngmkl_20MHz_boost_240830171659/'
    #base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_300mV_300pM_sample4_250113155944/'
    #base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_300mV_300pM_sample4_250113160623/'
    #base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_300mV_300pM_sample4_250113161058/'
    #base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_300mV_baseline_250113153808/'
    #base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_500mV_300pM_sample4_250113161718/'
    # List to store all peaks data
    all_peaks = []

    # Helper function to convert numpy arrays to lists for JSON serialization
    
    #root_path = '/media/alextusnin/Nanopore/DATA/plasmid/'

    base_dirs = get_data_folders(root_path)
    # Walk through all subdirectories

    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            if 'peaks_data.json' in files:
                # Read the JSON file
                with open(os.path.join(root, 'peaks_data.json'), 'r') as f:
                    peaks_data = json.load(f)
                
                # Add source file information if not already present
                for peak in peaks_data:
                    if 'source_file' not in peak:
                        peak['source_file'] = os.path.basename(root)
                
                # Append to our list
                all_peaks.extend(peaks_data)
                print(f"Processed: {os.path.basename(root)}")

        # Save combined data as JSON
        if all_peaks:
            output_file = os.path.join(base_dir, 'combined_peaks_data.json')
            with open(output_file, 'w') as f:
                json.dump(all_peaks, f, indent=2)
            print(f"\nCombined data saved to: {output_file}")
            print(f"Total peaks found: {len(all_peaks)}")
        else:
            print("No peaks_data.json files found!")
def __main__():
    root_path = '/Users/hugo/MOLECL/Molecl_data_H'
    combine_peaks_data(root_path)

if __name__ == '__main__':
    __main__()