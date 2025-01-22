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
import re
import json  

def process_batches(signal, start, end, depth=0, variance_threshold=5e-3, max_recursions=3, batch_number=100):
    """
    Recursively processes a segment of the signal to identify high-quality segments based on variance.
    Compute the variance of the batch (size : from start to end) and return it if variance below to threshold or split it in two 
    batches if variance higher than threshold and the process happens again (until max_recursion)

    Parameters:
        signal (numpy.ndarray): The full signal to process.
        start (int): The starting index of the batch.
        end (int): The ending index of the batch.
        depth (int): The current recursion depth (used to limit splitting).

    Returns:
        list: A list of tuples [(start1, end1), (start2, end2), ...] representing high-quality segments.
    """
    if depth > max_recursions:
        return []  # Stop further splitting if max recursion depth is reached
    batch_size = len(signal) / batch_number
    batch = signal[start:end]
    variance = np.std(batch)/np.median(batch)

    if variance <= variance_threshold:
        return [(start, end)]  # Return as valid segment
    elif end - start <= batch_size:
        return []  # Batch too small to split further

    # Split batch into two and process recursively
    mid = (start + end) // 2
    return process_batches(signal, start, mid, depth + 1) + process_batches(signal, mid, end, depth + 1)

def merge_segments(segments):
    """
    Merges adjacent high-quality segments into larger continuous segments.

    Parameters:
        segments (list): A list of tuples [(start1, end1), (start2, end2), ...] of valid signal.

    Returns:
        list: A list of merged tuples [(start1, end1), ...].
    """
    merged = []
    for seg in segments:
        # Add segment if merged is empty or if the end of the segment to add is different from the start of the existing segment
        # as they cannot be merged (not continuous)
        if not merged or merged[-1][1] != seg[0]:
            merged.append(seg)
        # If end of new segment equal start of previous segment, merge them by updating the end term with the one of the new segment
        else:
            merged[-1] = (merged[-1][0], seg[1])
    return merged

def save_segments_to_json(segments, output_file):
    """
    Saves the high-quality segments to a JSON file.

    Parameters:
        segments (list): A list of tuples [(start1, end1), ...] representing high-quality segments.
        output_file (str): The path to the output JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(segments, f, indent=2)
    print(f"High-quality segments saved to: {output_file}")


#####
##### Plots are here to check conformity, not useful otherwise and computationaly expensive...
#####
def plot_signal_with_segments(t, I, I_norm, segments, output_dir, base_filename, variation):
    """
    Plots the original signal, the normalized signal, and overlays the high-quality segments.

    Parameters:
        t (numpy.ndarray): Time array for the signal.
        I (numpy.ndarray): Original signal data.
        I_norm (numpy.ndarray): Normalized signal data (baseline removed).
        segments (list): List of tuples [(start1, end1), ...] representing high-quality segments.
        output_dir (str): Directory to save the plot.
        base_filename (str): Base name of the file being processed.
        variation (float): Baseline variation value for the signal.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=200)

    ax.plot(t[::1000], I[::1000], c='blue', label='Original Signal')  
    ax.plot(t[::1000], I_norm[::1000], c='orange', label='Normalized Signal (Baseline)')
    
    # Plot high-quality segments with a single label
    high_quality_label_added = False
    for start, end in segments:
        if not high_quality_label_added:
            ax.plot(t[start:end:1000], I_norm[start:end:1000], c='green', label='High-Quality Segment', alpha=0.8)
            high_quality_label_added = True
        else:
            ax.plot(t[start:end:1000], I_norm[start:end:1000], c='green', alpha=0.8)
    
    ax.set_title(f'Baseline Variation: {np.round(variation, 4)}')
    ax.legend()
    
    output_file = os.path.join(output_dir, f'{base_filename}_segments.png')
    plt.savefig(output_file)
    plt.close()
    print(f'Plot saved to: {output_file}')

def plot_individual_segments(t, I_norm, segments, output_dir, base_filename):
    """
    Plots each high-quality segment individually.

    Parameters:
        t (numpy.ndarray): Time array for the signal.
        I_norm (numpy.ndarray): Normalized signal data (baseline removed).
        segments (list): List of tuples [(start1, end1), ...] representing high-quality segments.
        output_dir (str): Directory to save the plots.
        base_filename (str): Base name of the file being processed.
    """
    for idx, (start, end) in enumerate(segments):
        segment_variation = np.var(I_norm[start:end])

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
        ax.plot(t[start:end], I_norm[start:end], c='green', label=f'Segment {idx + 1}')
        
        ax.set_title(f'High-Quality Segment {idx + 1}')
        ax.legend()

        segment_center = (t[start] + t[end - 1]) / 2  
        ax.text(segment_center, np.max(I_norm[start:end]), f"Var: {segment_variation:.4f}",
            color='red', fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='red', facecolor='white', alpha=0.6))
        
        output_file = os.path.join(output_dir, f'{base_filename}_segment_{idx + 1}.png')
        plt.savefig(output_file)
        plt.close()
        print(f'Segment {idx + 1} plot saved to: {output_file}')

#####
#####
#####

def SmoothenTransmission(sig, averaging_range, polyorder = 3):
    result = savgol_filter(sig,averaging_range,polyorder=polyorder)
    return result

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


window_size = 500
variation_threshold = 5e-3
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
max_recursions = 3  
batch_number = 100

#data_path = '/Volumes/WD/NBX/Raw_data_20241125_WSU_6_b1_171/sample2_spikein_241125191717/'
#data_path = '/Volumes/WD/NBX/Raw_data_20241125_WSU_6_b1_171/sample2_spikein_5mhz_241125192921/'
#data_path = '/Volumes/WD/ladder/500mV_200bp_1ngmkl_10MHz_boost_240830165006/'
#data_path = '/Volumes/WD/ladder/500mV_500bp_1ngmkl_20MHz_boost_240830171659/'
#data_path = '/Volumes/WD/ladder/500mV_500bp_1ngmkl_10MHz_boost_240830171035/'
#data_path = '/Volumes/WD/ladder/500mV_100bp_1ngmkl_20MHz_boost_240830164159/'
#data_path = '/Volumes/WD/ladder/500mV_100bp_1ngmkl_10MHz_boost_240830164444/'
data_path = r'E:\ladder\500mV_500bp_1ngmkl_20MHz_boost_240830171659'
output_base_dir = r'C:\Users\utilisateur\Desktop\MOLECL\Code'

# Retrieve all .abf files in the directory
all_files = glob.glob(os.path.join(data_path, '*.abf'))
pattern = re.compile(r'.*_CH001_0(0[0-9]|[1-5][0-9])\.abf$')
matching_files = [file for file in all_files if pattern.match(file)]

test_files = matching_files[:3] #Test only with 3 first files so not too long to compute

n_signal = 0
for file in (test_files):
    # Create output directories
    base_filename = os.path.splitext(os.path.basename(file))[0]
    output_dir = os.path.join(output_base_dir, base_filename)
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
    fs = 1/abs(t[1]-t[0])       # sample rate, Hz 

    median = np.median(I)
    I_norm = butter_lowpass_filter(I, 1e4, fs, 4) # baseline normalization. 10kHz cutoff, 4th 

    variation = np.std(I_norm)/np.median(I_norm)

    fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=200)
    ax.plot(t[::1000],I[::1000],c='blue') 
    ax.plot(t[::1000],I_norm[::1000],c='orange')
    ax.set_title('Baseline variation:' + str(np.round(variation,4)))

    plt.savefig(os.path.join(output_dir, f'background.png'))
    plt.close()
    print(f'Processed {base_filename}: baseline variation {variation}')

    if variation > variation_threshold :

        high_quality_segments = []
        batch_size = len(I_norm) // batch_number

        for i in range(batch_number):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(I_norm))
            high_quality_segments += process_batches(I_norm, start, end)

        merged_segments = merge_segments(high_quality_segments)

        # Plot the signal with high-quality segments
        plot_signal_with_segments(
        I = np.array(df.sweepY),
        t = np.array(df.sweepX),              
        I_norm=I_norm,              
        segments=merged_segments,   
        output_dir=output_dir,     
        base_filename=os.path.basename(file).replace('.abf', ''),  
        variation=variation         
    )
        
        # Plot individual high-quality segments
        plot_individual_segments(
        t = np.array(df.sweepX),               
        I_norm=I_norm,              
        segments=merged_segments,   
        output_dir=output_dir,     
        base_filename=os.path.basename(file).replace('.abf', '')  
    )