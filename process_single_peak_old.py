import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d, CubicSpline
from scipy.ndimage import convolve1d
from scipy.signal import find_peaks, peak_widths
import pdb
import json

def left_sigmoid_function(x, max_value, min_value, center, width):
    return (max_value-min_value)/(1+np.exp(-(x-center)/width))+min_value

def right_sigmoid_function(x, max_value, min_value, center, width):
    return (max_value-min_value)/(1+np.exp((x-center)/width))+min_value

def gaussian_function(x, max_value, min_value, center, width, sharpness):
    return max_value*np.exp(-(x-center)**2/(2*width**2)**sharpness) + min_value

def linear_function(x,slope, offset):
    return slope*x + offset

def str_to_array(s):
    try:
        # Remove any whitespace and split by spaces
        if isinstance(s, str):
            # Clean up the string: remove brackets and split by whitespace
            clean_str = s.strip('[]').strip()
            # Convert to numpy array of floats
            return np.array([float(x) for x in clean_str.split()])
        return s
    except Exception as e:
        print(f"Error converting string to array: {e}")
        print(f"Problematic string: {s}")
        return None
    
def process_single_peak(combined_df, rolling_window = 50):
    #base_dir = '/media/alextusnin/Nanopore/DATA/plasmid/109_1MKCl_500mV_sample2_100xdilution_dilutw_10uLKCl_10MHz_250429152930/109_1MKCl_500mV_sample2_100xdilution_dilutw_10uLKCl_10MHz_250429152930_CH001_000'

    #with open(os.path.join(base_dir, 'peaks_data.json'), 'r') as f:
    #    peaks_data = json.load(f)

    # Convert to DataFrame for easier processing
    #combined_df = pd.DataFrame(peaks_data)

    #print(combined_df.columns)
    # Convert lists back to numpy arrays
    #array_columns = ['time', 'filtered_signal', 'gradient']
    array_columns = ['t_start', 't_end', 'dt', 'filtered_signal', 'norm_signal', 'raw_signal_not_norm']
    for column in array_columns:
        combined_df[column] = combined_df[column].apply(np.array)

    #signal_index = 4
    results = []
    
    # rolling_window = 50
    for signal_index in range(len(combined_df)):
        #norm_signal = combined_df.iloc[signal_index]['norm_signal']
        signal = combined_df.iloc[signal_index]['filtered_signal']
        signal = np.array(signal, dtype=float)
        t_signal = np.linspace(combined_df.iloc[signal_index]['t_start'], combined_df.iloc[signal_index]['t_end'], len(signal))
        #raw_signal_not_norm = combined_df.iloc[signal_index]['raw_signal']
        signal -=np.median(signal)

        #convolved_signal = convolve1d(signal, np.ones(rolling_window), mode='constant')

        #derivative_convolved = np.gradient(convolved_signal)
        derivative = np.gradient(pd.Series(signal).rolling(window=rolling_window,center=True).mean())
        derivative = pd.Series(derivative).rolling(window=rolling_window,center=True).mean()
        derivative = derivative/np.max(abs(derivative))

        rolling_std = pd.Series(signal).rolling(window=rolling_window,center=True).std()
        rolling_mean = pd.Series(signal).rolling(window=rolling_window,center=True).mean()
        rolling_std_of_rolling_mean = pd.Series(rolling_mean).rolling(window=rolling_window,center=True).std()

        change_signature = rolling_std*rolling_std_of_rolling_mean
        change_signature = change_signature/np.max(change_signature)


        #print('1*np.median(change_signature)', change_signature.median())
        peaks, _ = find_peaks(change_signature, distance = rolling_window/2, height=10*change_signature.median())
        #print('peaks', peaks)
        widths = peak_widths(change_signature, peaks, rel_height=0.5)

        #print('widths', widths)

        #fig, ax = plt.subplots(figsize=(10, 5))
        #ax.plot(t_signal,1-np.array(combined_df.iloc[signal_index]['raw_signal'],dtype=float))
        popt_array = []
        times_array = np.array([])
        signal_array = np.nan*np.ones(np.size(t_signal))

        t_signal_rise = 0
        t_signal_fall = 0
        dt_signal = 0
        left_baseline = 0
        signal_type = ''
        if peaks[0]+widths[0][0]*1 - (peaks[-1]-widths[0][-1]*1) > 0:
            p0=[np.max(signal), np.min(signal[0]), t_signal[peaks[0]], widths[0][0]/2, 1]
            bounds=([p0[0]-abs(p0[0]/2), p0[1]-np.abs(p0[1])/2,p0[2]-widths[0][0]/2, p0[3]/10,0.05], [p0[0]+abs(p0[0]), p0[1]+np.abs(p0[1]), p0[2]+widths[0][0]/2, p0[3]*2,2])
            try:
                popt, pcov = curve_fit(gaussian_function, t_signal, signal, p0=p0,bounds=bounds,maxfev=10000)
                #popt, pcov = curve_fit(gaussian_function, t_signal, 1-np.array(combined_df.iloc[signal_index]['raw_signal'],dtype=float), p0=p0,bounds=bounds,maxfev=10000)
                dt_signal = popt[3]
            except Exception as e:
                print(f"Error fitting gaussian function: {e}")
                print(f"Problematic signal: {signal}")
                print(f"Problematic t_signal: {t_signal}")
                print(f"Problematic p0: {p0}")
                print(f"Problematic bounds: {bounds}")
                continue
            #print(popt)
            #ax.plot(t_signal, gaussian_function(t_signal, *popt), color='green')
            popt_array.append(popt)
            signal_array = gaussian_function(t_signal, *popt)
            signal_type = 'gaussian'
        else:
            for ii in range(len(peaks)):
                window_width = np.int16(widths[0][ii]*2)
                #print('window_width', window_width)
                window_start = np.int16(peaks[ii] - window_width*1)
                window_end = np.int16(peaks[ii] + window_width*1)
                #print('window_start', window_start)
                #print('window_end', window_end)
                window_signal = signal[window_start:window_end]
                #window_signal = 1-np.array(combined_df.iloc[signal_index]['raw_signal'],dtype=float)[window_start:window_end]
                window_time = t_signal[window_start:window_end]
                #times_array.append([window_start, window_end])
                times_array = np.append(times_array, window_time)
                if derivative[peaks[ii]] > 0:
                    p0=[window_signal[-1], window_signal[0], t_signal[peaks[ii]], window_width/10]
                    #print('p0', p0)
                    bounds=([p0[0]-abs(p0[0]/2), p0[1]-np.abs(p0[1])/2,p0[2]-window_width/2, p0[3]/10], [p0[0]+abs(p0[0]), p0[1]+np.abs(p0[1]), p0[2]+window_width/2, p0[3]*2])
                    #print('bounds', bounds)
                    try:
                        popt, pcov = curve_fit(left_sigmoid_function, window_time, window_signal, p0,bounds=bounds,maxfev=10000)
                    except Exception as e:
                        print(f"Error fitting left sigmoid function: {e}")
                        print(f"Problematic window_signal: {window_signal}")
                        print(f"Problematic window_time: {window_time}")
                        print(f"Problematic p0: {p0}")
                        print(f"Problematic bounds: {bounds}")
                        continue
                    if ii == 0:
                        left_baseline = popt[1]
                    #print(popt)
                    #ax.plot(window_time, left_sigmoid_function(window_time, *popt), color='green')
                    popt_array.append(popt)
                    signal_array[window_start:window_end] = left_sigmoid_function(window_time, *popt)
                    if ii == 0:
                        signal_array[0:window_start] = left_sigmoid_function(t_signal[0:window_start], *popt)
                        t_signal_rise = popt[2]
                else:
                    p0=[window_signal[0], window_signal[-1], t_signal[peaks[ii]], window_width/10]
                    
                    
                    bounds=([p0[0]-abs(p0[0]/2), p0[1]-np.abs(p0[1])/2,p0[2]-window_width/2, p0[3]/10], [p0[0]+abs(p0[0]), p0[1]+np.abs(p0[1]), p0[2]+window_width/2, p0[3]*2])
                    #print('p0', p0)
                    #print('bounds', bounds)
                    try:
                        popt, pcov = curve_fit(right_sigmoid_function, window_time, window_signal, p0,bounds=bounds,maxfev=10000)
                    except Exception as e:
                        print(f"Error fitting right sigmoid function: {e}")
                        print(f"Problematic window_signal: {window_signal}")
                        print(f"Problematic window_time: {window_time}")
                        print(f"Problematic p0: {p0}")
                        print(f"Problematic bounds: {bounds}")
                        continue
                    if ii == len(peaks)-1:
                        #print('left_baseline', left_baseline)
                        #print('popt[1]', popt[1])
                        popt[1] = left_baseline
                    #print(popt)
                    #ax.plot(window_time, right_sigmoid_function(window_time, *popt), color='red')
                    popt_array.append(popt)
                    signal_array[window_start:window_end] = right_sigmoid_function(window_time, *popt)
                    if ii == len(peaks)-1:
                        signal_array[window_end:] = right_sigmoid_function(t_signal[window_end:], *popt)
                        t_signal_fall = popt[2]
            signal_type = 'sigmoid'
        if signal_type == 'sigmoid':
            #print('t_signal_rise', t_signal_rise)
            #print('t_signal_fall', t_signal_fall)
            dt_signal = abs(t_signal_fall - t_signal_rise)
        else:
            dt_signal = popt[-2]
            #print('dt_signal', dt_signal)
    
        mask = np.isnan(signal_array)       
        result_array = CubicSpline(t_signal[~mask], signal_array[~mask])
        
        norm_factor = np.mean(np.array(combined_df.iloc[signal_index]['raw_signal_not_norm'])/np.array(combined_df.iloc[signal_index]['raw_signal']))
        #ax.plot(t_signal, result_array(t_signal)-np.min(result_array(t_signal)), color='red')
        #ax.set_xlim(-dt_signal*5, dt_signal*5)
        #plt.show()
        results.append({
            'result_array': result_array(t_signal)-np.min(result_array(t_signal)),
            't_signal': t_signal, 
            'dt_signal': dt_signal, 
            't_start': combined_df.iloc[signal_index]['t_start'], 
            't_end': combined_df.iloc[signal_index]['t_end'],
            'dt': combined_df.iloc[signal_index]['dt'], 
            'norm_factor': norm_factor, 
            'popt_array': popt_array, 
            'times_array': times_array, 
            'signal_array': signal, 
            'raw_signal_not_norm': combined_df.iloc[signal_index]['raw_signal_not_norm'],
            'signal_type': signal_type,
            'signal_type_index': signal_index})

    return results


    

def __main__():
    base_dir = '/media/alextusnin/Nanopore/DATA/plasmid/109_1MKCl_500mV_sample2_100xdilution_dilutw_10uLKCl_10MHz_250429152930'
    with open(os.path.join(base_dir, 'combined_peaks_data.json'), 'r') as f:
        peaks_data = json.load(f)
    combined_df = pd.DataFrame(peaks_data)
    results = process_single_peak(combined_df,rolling_window=50)
    #print(result_array(t_signal))
    #print(popt_array)
    #print(times_array)

if __name__ == '__main__':
    __main__()