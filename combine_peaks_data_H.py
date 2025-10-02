import os, json
import numpy as np
import pandas as pd  
from screening_sample_ssd import get_data_folders

def numpy_to_list(obj):
    return obj.tolist() if isinstance(obj, np.ndarray) else obj

def combine_peaks_data(root_path):
    base_dirs = get_data_folders(root_path)

    for base_dir in base_dirs:
        all_peaks = []

        for root, dirs, files in os.walk(base_dir):
            if 'peaks_data.json' in files:
                peaks_path = os.path.join(root, 'peaks_data.json')
                with open(peaks_path, 'r') as f:
                    peaks_data = json.load(f)

                # Ensure list
                if isinstance(peaks_data, dict):
                    peaks_iter = peaks_data.values()
                else:
                    peaks_iter = peaks_data

                # Tag each peak with a stable source (the sample folder)
                for peak in peaks_iter:
                    if isinstance(peak, dict):
                        # Prefer full sample dir to avoid collisions
                        peak.setdefault('source_file', base_dir)
                        # (Optional) convert any numpy arrays to lists
                        for k, v in list(peak.items()):
                            peak[k] = numpy_to_list(v)
                all_peaks.extend(peaks_iter)
                print(f"Processed: {os.path.relpath(root, base_dir)}")

        # Save per-folder combined JSON (only that folder’s peaks)
        output_file = os.path.join(base_dir, 'combined_peaks_data.json')
        if all_peaks:
            with open(output_file, 'w') as f:
                json.dump(all_peaks, f, indent=2)
            print(f"\nCombined data saved to: {output_file}")
            print(f"Total peaks found in this folder: {len(all_peaks)}\n")
        else:
            # If nothing found in this folder, remove any stale combined JSON
            if os.path.exists(output_file):
                os.remove(output_file)
            print(f"No peaks_data.json files found under: {base_dir}\n")

def __main__():
    root_path = '/Users/hugo/MOLECL/Molecl_data_H'
    combine_peaks_data(root_path)

if __name__ == '__main__':
    __main__()
