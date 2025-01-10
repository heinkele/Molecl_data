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

#%% Main processing loop
window_size = 500
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
#data_path = '/Volumes/WD/NBX/Raw_data_20241125_WSU_6_b1_171/sample2_spikein_241125191717/'
data_path = '/Volumes/WD/NBX/Raw_data_20241125_WSU_6_b1_171/sample2_spikein_5mhz_241125192921/'
#data_path = '/Volumes/WD/ladder/500mV_200bp_1ngmkl_10MHz_boost_240830165006/'
#data_path = '/Volumes/WD/ladder/500mV_500bp_1ngmkl_20MHz_boost_240830171659/'
#data_path = '/Volumes/WD/ladder/500mV_500bp_1ngmkl_10MHz_boost_240830171035/'
#data_path = '/Volumes/WD/ladder/500mV_100bp_1ngmkl_20MHz_boost_240830164159/'
#data_path = '/Volumes/WD/ladder/500mV_100bp_1ngmkl_10MHz_boost_240830164444/'
n_signal = 0
for file in glob.glob(data_path+'*CH001_0[5][4-9].abf'):
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
    I_norm = butter_lowpass_filter(I, 1e4, fs, 4) # baseline normalization. 10kHz cutoff, 4th 
    I_filtered = butter_lowpass_filter(1-I/I_norm, cutoff, fs, order)
    I_smooth = SmoothenTransmission(1-I/I_norm, 40,polyorder=2)
    order = 6
    fs = 1/abs(t[1]-t[0])       # sample rate, Hz
    cutoff = 5*1e5  
    I_filtered = butter_lowpass_filter(1-I/I_norm, cutoff, fs, order)
    #I_filtered = butter_lowpass_filter(I_smooth, cutoff, fs, order)
    median = np.median(I)

    variation = np.std(I_norm)/np.median(I_norm)

    if variation > 5e-3:
        fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=200)
        ax.plot(t[::1000],I[::1000],c='blue') 
        ax.plot(t[::1000],I_norm[::1000],c='orange')
        ax.set_title('Baseline variation:' + str(np.round(variation,4)))

        plt.savefig(os.path.join(output_dir, f'background.png'))
        plt.close()
        print(f'Processed {base_filename}: baseline variation {variation}')

        continue
    peak_args = find_peaks(I_filtered, prominence=0.5,distance = 1000)[0]
    n_signal+=len(peak_args)
    
    if len(peak_args) >0 :
        peak_data = []  # Initialize the list here
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
        
        for jj in range(0,len(peak_args)):
            
            base_line[file][jj] = np.median(I_norm[peak_args[jj]-window_size:peak_args[jj]+window_size])/median
            norm_signals[file][jj] = I_norm[peak_args[jj]-window_size:peak_args[jj]+window_size]/median
            signals[file][jj] = I_smooth[peak_args[jj]-window_size:peak_args[jj]+window_size]
            signals_but[file][jj] = I_filtered[peak_args[jj]-window_size:peak_args[jj]+window_size]
            
            raw_signal_not_norm[file][jj] = I[peak_args[jj]-window_size:peak_args[jj]+window_size]
            raw_signals[file][jj] = I[peak_args[jj]-window_size:peak_args[jj]+window_size]/I_norm[peak_args[jj]-window_size:peak_args[jj]+window_size]
            t_signal[file][jj] = t[peak_args[jj]-window_size:peak_args[jj]+window_size]*1e6-t[peak_args[jj]]*1e6
            i_grad[file][jj] = np.gradient(signals_but[file][jj])
            i_grad_sig[file][jj] = (i_grad[file][jj])*np.abs(signals_but[file][jj])#**2
            i_grad_sig[file][jj][i_grad_sig[file][jj]>0] = abs(i_grad_sig[file][jj][i_grad_sig[file][jj]>0])**0.7
            i_grad_sig[file][jj][i_grad_sig[file][jj]<0] = -abs(i_grad_sig[file][jj][i_grad_sig[file][jj]<0])**0.7
            i_grad_sig[file][jj]/=np.max(i_grad_sig[file][jj])
            peak_args_grad = find_peaks(i_grad_sig[file][jj], prominence=0.1, height=0.23)[0]
            n_func[file][jj] = len(peak_args_grad)    
            
            # Store peak data in a dictionary
            peak_dict = {
                'peak_index': jj,
                'time': t_signal[file][jj].tolist(),
                'raw_signal': raw_signals[file][jj].tolist(),
                'raw_signal_not_norm': raw_signal_not_norm[file][jj].tolist(),
                'norm_signal': norm_signals[file][jj].tolist(),
                'filtered_signal': signals_but[file][jj].tolist(),
                'gradient': i_grad_sig[file][jj].tolist(),
                'baseline': float(base_line[file][jj]),
                'n_gradient_peaks': int(len(peak_args_grad))
            }
            peak_data.append(peak_dict)
            
            # Create and save figure
            fig, ax = plt.subplots(2, 1, figsize=(10,10), dpi=200)
            ax[0].plot(t_signal[file][jj],np.median(raw_signals[file][jj])-raw_signals[file][jj],alpha=0.3,label='Raw signal')
            
            ax[0].plot(t_signal[file][jj],np.median(norm_signals[file][jj])-norm_signals[file][jj],alpha=0.5,label='Normalization signal')
            
            #ax[0].plot(t_signal[file][jj],signals[file][jj],c='r',label='Savgol filtered')
            
            ax[0].plot(t_signal[file][jj],signals_but[file][jj],c='k',label='LP filter, 0.1 MHz')
            
            
            ax[1].plot(t_signal[file][jj],i_grad_sig[file][jj],label='Gradient (LP)')
            peak_args_grad = find_peaks(i_grad_sig[file][jj], prominence=0.1, height=0.23)[0]
            ax[1].scatter(t_signal[file][jj][peak_args_grad],i_grad_sig[file][jj][peak_args_grad],label='Gradient (LP)')
            
            ax[0].set_ylabel(r'$I$, arb. units')
            ax[1].set_ylabel(R'$dI/dt$, arb. units')
            ax[1].set_xlabel(R'$t$, $\mu$s')
            norm_signals[jj] = I_norm[peak_args[jj]-window_size:peak_args[jj]+window_size]/median
            
            fig.suptitle(f'Peak {jj}')
            plt.savefig(os.path.join(figures_dir, f'peak_{jj}.png'))
            plt.close()
        
        # Save peaks data to JSON instead of CSV
        with open(os.path.join(output_dir, 'peaks_data.json'), 'w') as f:
            json.dump(peak_data, f, indent=2)
    
    print(f'Processed {base_filename}: found {len(peak_args)} peaks')