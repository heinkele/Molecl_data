#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:17:54 2024

@author: alexandertusnin
"""

import pyabf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, filtfilt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
import glob
from scipy.signal import butter, lfilter
import pandas as pd
import os
import json  # Add this import

def NormalizeTransmission(sig, averaging_range, polyorder = 2):
    
    norm = savgol_filter(sig,averaging_range,polyorder=polyorder)
    return norm

def SmoothenTransmission(sig, averaging_range, polyorder = 3):
    result = savgol_filter(sig,averaging_range,polyorder=polyorder)
    return result

def SignalFunc_old(x, A, b, x1, c, x2, offset):
    
    return A*(np.tanh(b*x+x1) - np.tanh(c*x+x2)) -offset

def SignalFunc(x, *popt):
    res = np.zeros_like(x)
    n_func = (len(popt)-1)//5
    #params = np.zeros([n_func,5])
    for jj in range(n_func):
        res += popt[jj*5+0]*(np.tanh(popt[jj*5+1]*x+popt[jj*5+2]) - np.tanh(popt[jj*5+3]*x+popt[jj*5+4])) 
    res-=popt[-1]
    return res


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y

# Add this function to create directories if they don't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_sudden_changes(signal, threshold=0.1, min_segment_length=1000):
    """Find indices where sudden changes occur in the signal.
    Returns segment boundaries including start and end of signal."""
    # Calculate normalized gradient
    grad = np.gradient(signal)
    grad_norm = grad / np.max(np.abs(grad))
    
    # Find significant changes
    change_points = np.where(np.abs(grad_norm) > threshold)[0]
    
    # Add start and end points
    all_points = np.concatenate(([0], change_points, [len(signal)-1]))
    
    # Filter out segments that are too short
    segments = []
    for i in range(len(all_points)-1):
        if all_points[i+1] - all_points[i] >= min_segment_length:
            segments.append((all_points[i], all_points[i+1]))
    
    return segments

# Add this function before the main processing loop
def get_data_folders(root_path):
    """
    Find all relevant data folders in the given root path.
    Excludes folders containing 'baseline' in their name.
    """
    folders = []
    for item in os.listdir(root_path):
        full_path = os.path.join(root_path, item)
        #if os.path.isdir(full_path) and '3MLiCl' in item and 'baseline' not in item:
        #if os.path.isdir(full_path) and '100pM' in item and 'baseline' not in item:
        #if os.path.isdir(full_path) and '2023' in item and 'baseline' not in item:
        #if os.path.isdir(full_path) and '500bp' in item and 'baseline' not in item:
        if os.path.isdir(full_path) and 'baseline' not in item :
            folders.append(full_path)
    return sorted(folders)

def load_processing_status(root_path):
    """Load the processing status from JSON file."""
    status_file = os.path.join(root_path, 'processing_status.json')
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    return {}

def save_processing_status(root_path, status):
    """Save the processing status to JSON file."""
    status_file = os.path.join(root_path, 'processing_status.json')
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

def mark_file_as_processed(root_path, data_folder, file_path):
    """Mark a file as processed in the status JSON."""
    status = load_processing_status(root_path)
    if data_folder not in status:
        status[data_folder] = []
    status[data_folder].append(file_path)
    save_processing_status(root_path, status)

def is_file_processed(root_path, data_folder, file_path):
    """Check if a file has been processed."""
    status = load_processing_status(root_path)
    return data_folder in status and file_path in status[data_folder]

#%% Main processing loop
def process_data(root_path):
    window_size = 2500
    signals = {}
    signals_but = {}
    raw_signals = {}
    raw_signal_not_norm = {}
    t_signal = {}
    i_grad = {}
    i_grad_sig = {}
    norm_signals = {}
    n_func = {}
    base_line = {}


    #root_path = '/Volumes/Nanopore/DATA/NewBioSample6/' 
    #root_path = '/Volumes/Nanopore/DATA/Database100-200-500/'
    #root_path = '/media/alextusnin/Nanopore/DATA/plasmid/'

    data_folders = get_data_folders(root_path)
    print(data_folders)
    for data_path in data_folders:
        print(f"Processing folder: {data_path}")
    
        n_signal = 0
        print('files:',glob.glob(os.path.join(data_path, '*CH001_[0-9][0-9][0-9].abf')))
        for file in glob.glob(os.path.join(data_path, '*CH001_[0-9][0-9][0-9].abf')):
            # Check if file has already been processed
            if is_file_processed(root_path, data_path, file):
                print(f"Skipping already processed file: {file}")
                continue
                
            print('Processing file:', file)

            # Reset peak_data for each new file
            
            peak_data = []
            signals[file] = {}
            signals_but[file] = {}
            raw_signals[file] = {}
            raw_signal_not_norm[file] = {}
            t_signal[file] = {}
            i_grad[file] = {}
            i_grad_sig[file] = {}
            norm_signals[file] = {}
            n_func[file] = {}
            base_line[file] = {}
            
            # Create output directories
            base_filename = os.path.splitext(os.path.basename(file))[0]
            output_dir = os.path.join(os.path.dirname(file), base_filename)
            figures_dir = os.path.join(output_dir, 'figures')
            ensure_dir(output_dir)
            ensure_dir(figures_dir)
            
            try:
                df = pyabf.ABF(file)
            except:
                continue
            df.setSweep(0)
            I = np.array(df.sweepY)
            t = np.array(df.sweepX)
            order = 6
            fs = 1/abs(t[1]-t[0])       # sample rate, Hz
            cutoff = 5*1e5  

            median = np.median(I)
            I_norm = butter_lowpass_filter(I, 1e3, fs, 4)
            
            # Find segments between sudden changes
            segments = find_sudden_changes(I_norm, threshold=0.3, min_segment_length=50000)
            variation = np.std(I_norm)/np.median(I_norm)
            fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=200)
            ax.plot(t[::1000],I[::1000],c='blue') 
            ax.plot(t[::1000],I_norm[::1000],c='orange')
            ax.set_title('Baseline variation:' + str(np.round(variation,4)))

            plt.savefig(os.path.join(output_dir, f'background.png'))
            plt.close()
            
            global_peak_counter = 0  # Add this counter to track total peaks across segments
            print('Number of segments:', len(segments))
            for seg_start, seg_end in segments:
                # Analyze each segment independently
                I_segment = I[seg_start:seg_end]
                I_norm_segment = I_norm[seg_start:seg_end]
                t_segment = t[seg_start:seg_end]
                
                # Calculate local variation
                variation = np.std(I_norm_segment)/np.median(I_norm_segment)
                #print(variation)
                
                if variation <= 5e-3:  # Process segments with acceptable variation
                    I_filtered_segment = butter_lowpass_filter(1-I_segment/I_norm_segment, cutoff, fs, order)
                    peak_args = find_peaks(I_filtered_segment, prominence=0.2, distance=1000)[0]
                    
                    # Adjust peak indices to global coordinates
                    peak_args = peak_args + seg_start
                    
                    if len(peak_args) > 0:
                        # Process peaks using global_peak_counter instead of jj
                        for local_idx in range(len(peak_args)):
                            try:
                                # Use global_peak_counter for dictionary indexing
                                peak_idx = global_peak_counter + local_idx
                                
                                # Make sure to check array bounds when using window_size
                                start_idx = max(peak_args[local_idx]-window_size, 0)
                                end_idx = min(peak_args[local_idx]+window_size, len(I))
                                
                                # Calculate SNR for this slice
                                slice_signal = I_filtered_segment[start_idx-seg_start:end_idx-seg_start]
                                peak_amplitude = np.max(np.abs(slice_signal))
                                noise_level = np.std(slice_signal)  # Using standard deviation as noise measure
                                snr_db = 10 * np.log10(peak_amplitude / noise_level)
                                
                                # Only process peaks with SNR > 2 dB
                                if 1:
                                    # Update all dictionary references to use peak_idx instead of jj
                                    base_line[file][peak_idx] = np.median(I_norm[start_idx:end_idx])/median
                                    norm_signals[file][peak_idx] = I_norm[start_idx:end_idx]/median
                                    signals[file][peak_idx] = I_filtered_segment[start_idx-seg_start:end_idx-seg_start]
                                    signals_but[file][peak_idx] = I_filtered_segment[start_idx-seg_start:end_idx-seg_start]
                                    
                                    raw_signal_not_norm[file][peak_idx] = I[start_idx:end_idx]
                                    raw_signals[file][peak_idx] = I[start_idx:end_idx]/I_norm[start_idx:end_idx]
                                    t_signal[file][peak_idx] = t[start_idx:end_idx]*1e6-t[peak_args[local_idx]]*1e6
                                    
                                    # Convert NumPy types to native Python types
                                    peak_dict = {
                                        'peak_index': int(peak_idx),
                                        'segment_index': int(local_idx),
                                        'segment_start': int(seg_start),
                                        'segment_end': int(seg_end),
                                        't_start': float(t_signal[file][peak_idx][0]),
                                        't_end': float(t_signal[file][peak_idx][-1]),
                                        'dt': float(t_signal[file][peak_idx][1] - t_signal[file][peak_idx][0]),
                                        'raw_signal': [float(x) for x in raw_signals[file][peak_idx].tolist()],
                                        'raw_signal_not_norm': [float(x) for x in raw_signal_not_norm[file][peak_idx].tolist()],
                                        'norm_signal': [float(x) for x in norm_signals[file][peak_idx].tolist()],
                                        'filtered_signal': [float(x) for x in signals_but[file][peak_idx].tolist()],
                                        'baseline': float(base_line[file][peak_idx]),
                                        'snr_db': float(snr_db)  # Add SNR to the output
                                    }
                                    peak_data.append(peak_dict)
                                    peak_dict={}
                                    
                                    # Update figure title to show global peak index and SNR
                                    fig, ax = plt.subplots(1, 1, figsize=(6,6), dpi=200)
                                    ax.plot(t_signal[file][peak_idx], signals_but[file][peak_idx], c='blue')
                                    ax.plot(t_signal[file][peak_idx], raw_signals[file][peak_idx], c='orange')
                                    ax.set_title(f'Peak {peak_idx} (Segment Peak {local_idx}, SNR: {snr_db:.1f} dB)')
                                    plt.savefig(os.path.join(figures_dir, f'peak_{peak_idx}.png'))
                                    plt.close()
                                    
                                    # Increment global peak counter only for accepted peaks
                                    global_peak_counter += 1
                                
                            except Exception as e:
                                print(f"Error processing peak {peak_idx} in file {file}: {str(e)}")
                                continue
                        
                        print(f'Processed {base_filename}: found {global_peak_counter} peaks with SNR > 2 dB in segment')
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=200)
                    ax.plot(t_segment[::100], I_segment[::100], c='blue')
                    ax.plot(t_segment[::100], I_norm_segment[::100], c='orange')
                    ax.set_title(f'Segment variation: {np.round(variation,4)}')
                    plt.savefig(os.path.join(output_dir, f'segment_{seg_start}_{seg_end}.png'))
                    plt.close()
                # Save peaks data to JSON instead of CSV
                with open(os.path.join(output_dir, 'peaks_data.json'), 'w') as f:
                    json.dump(peak_data, f, indent=2)
                
                # Mark file as processed after successful completion
                mark_file_as_processed(root_path, data_path, file)
            
        
def __main__():
    root_path = '/media/alextusnin/Nanopore/DATA/plasmid/'
    process_data(root_path)
    
if __name__ == '__main__':
    __main__()
    