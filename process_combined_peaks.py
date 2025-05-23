import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import pdb
import json

#print("Script is starting...")


# Helper function to safely convert string to array
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

# Load the combined data


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
base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_300mV_300pM_sample4_250113155944/'
#base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_300mV_300pM_sample4_250113160623/'
#base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_300mV_300pM_sample4_250113161058/'
#base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_300mV_baseline_250113153808/'
#base_dir = '/Volumes/Nanopore/DATA/NewBioSample4/160_3MLiCl_1MHz_500mV_300pM_sample4_250113161718/'


with open(os.path.join(base_dir, 'combined_peaks_data.json'), 'r') as f:
    peaks_data = json.load(f)

# Convert to DataFrame for easier processing
combined_df = pd.DataFrame(peaks_data)

# Convert lists back to numpy arrays
#array_columns = ['time', 'filtered_signal', 'gradient']
array_columns = ['t_start', 't_end', 'dt', 'filtered_signal']
for column in array_columns:
    combined_df[column] = combined_df[column].apply(np.array)

#pdb.set_trace()  # Debugging breakpoint

# Verify conversion
#print("Data types after conversion:")
#print(f"Time type: {type(combined_df.iloc[0]['time'])}")
#print(f"Time sample: {combined_df.iloc[0]['time'][:5]}")  # Show first 5 elements

#pdb.set_trace()  # Alternative to breakpoint()

print('The combined data script is runningD')

# Define a fitting function (example: you might want to adjust this)
def peak_function(x, amplitude, center, width, baseline):
    """
    Gaussian peak function with baseline
    Parameters:
        x: time points
        amplitude: peak height
        center: peak center position
        width: peak width (sigma)
        baseline: baseline offset
    """
    return amplitude * np.exp(-((x - center)**2 / (2 * width**2))**2) + baseline

# Create directories for results and histograms
results_dir = os.path.join(base_dir, 'peak_fits')
histogram_dir = os.path.join(results_dir, 'histograms')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(histogram_dir):
    os.makedirs(histogram_dir)

# Lists to store fitting results
fit_results = []

# Calculate FWHM
def calculate_fwhm(time, fitted_curve):
    """Calculate Full Width at Half Maximum"""
    # Find the maximum
    peak_height = np.max(fitted_curve)
    half_max = peak_height / 2
    
    # Create interpolation function
    f = interp1d(time, fitted_curve - half_max, kind='cubic')
    
    # Find roots (where curve crosses half max)
    t_dense = np.linspace(time[0], time[-1], 1000)
    roots = []
    for i in range(len(t_dense)-1):
        if f(t_dense[i]) * f(t_dense[i+1]) <= 0:
            roots.append(t_dense[i])
    
    if len(roots) >= 2:
        return abs(roots[-1] - roots[0])
    else:
        return np.nan

# Function to calculate area under the curve using trapezoidal rule
def calculate_area(time, signal):
    return np.trapz(signal, time)

# Process each peak
for idx, row in combined_df.iterrows():
    try:
        # Extract peak data
        #time = row['time']
        t_start = row['t_start']
        t_end = row['t_end']
        dt = row['dt']
        #time = np.arange(t_start,t_end+dt,dt)
        signal = row['filtered_signal']
        n_t = len(signal)
        time = np.linspace(t_start,t_end,n_t)
        raw_signal = row['raw_signal']
        raw_signal_not_norm = row['raw_signal_not_norm']
        norm_factor = np.mean(np.array(raw_signal_not_norm)/np.array(raw_signal))
        #gradient = row['gradient']
        gradient = np.gradient(signal)
        gradient *= np.abs(signal)
        gradient[gradient>0] = abs(gradient[gradient>0])**0.7
        gradient[gradient<0] = -abs(gradient[gradient<0])**0.7
        gradient/=np.max(gradient)
        
        # Calculate area under the curve
        
        
        t_rise_arg = np.argmax(gradient)
        dt_coarse = abs(time[t_rise_arg]*2)
        
        # Initial parameter guess
        p0 = [
            np.max(signal) - np.min(signal),  # amplitude
            time[np.argmax(signal)],          # center
            dt_coarse,          # width
            np.min(signal)                    # baseline
        ]
        #print(p0)
        # Define bounds for the parameters
        lower_bounds = [
            0.05,                    # min amplitude
            -20,               # min center (start of time range)
            0,                     # min width
            -5               # min baseline
        ]
        
        upper_bounds = [
            0.9,                    # max amplitude
            20,             # max center (end of time range)
            dt_coarse * 2,        # max width
            5                # max baseline
        ]
        
        # Fit the peak with bounds
        popt, pcov = curve_fit(peak_function, time, signal, p0=p0, 
                             bounds=(lower_bounds, upper_bounds),method='dogbox')
        
        # Calculate fit quality
        fitted_curve = peak_function(time, *popt)
        r_squared = 1 - np.sum((signal - fitted_curve)**2) / np.sum((signal - np.mean(signal))**2)
        
        # Calculate FWHM
        fwhm = calculate_fwhm(time, fitted_curve)
        #print(fwhm)
        area = calculate_area(time, fitted_curve-popt[3])*norm_factor
        # Add area to fit results
        if popt[0] < 0.9 and popt[0]>0.15:
            fit_results.append({
                'source_file': row['source_file'],
                'peak_index': row['peak_index'],
                'amplitude': popt[0],
                'center': popt[1],
                'width': popt[2],
                'baseline': popt[3],
                'r_squared': r_squared,
                'fwhm': fwhm,
                'area': area,
                'norm_factor': norm_factor
            })
        
            # Plot the result
            plt.figure(figsize=(10, 6))
            plt.plot(time, signal*norm_factor, 'b-', label='Data')
            plt.plot(time, (fitted_curve-popt[3])*norm_factor, 'r--', label='Fit')
            plt.title(f"Peak fit - {row['source_file']} - Peak {row['peak_index']}")
            plt.xlabel('Time')
            plt.ylabel('Signal')
            plt.legend()
            plt.savefig(os.path.join(results_dir, f"fit_{row['source_file']}_peak_{row['peak_index']}.png"))
            plt.close()
            
        if idx % 10 == 0:  # Progress update every 10 peaks
            print(f"Processed {idx}/{len(combined_df)} peaks")
            
    except Exception as e:
        print(f"Error processing peak {idx} from {row['source_file']}: {str(e)}")
        continue

# First create the DataFrame from fit_results
fit_results_df = pd.DataFrame(fit_results)
fit_results_df.to_csv(os.path.join(results_dir, 'fit_results.csv'), index=False)



# Make sure we have FWHM values before plotting
if 'fwhm' in fit_results_df.columns:
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(fit_results_df['fwhm'].dropna(), bins=50, edgecolor='black')
    plt.title('Histogram of Peak FWHM')
    plt.xlabel('FWHM (time units)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(histogram_dir, 'fwhm_histogram.png'))
    plt.close()

    # Save statistics to a text file
    with open(os.path.join(histogram_dir, 'fwhm_statistics.txt'), 'w') as f:
        f.write("FWHM Statistics:\n")
        f.write(f"Mean FWHM: {fit_results_df['fwhm'].mean():.3f}\n")
        f.write(f"Std FWHM: {fit_results_df['fwhm'].std():.3f}\n")
        f.write(f"Median FWHM: {fit_results_df['fwhm'].median():.3f}\n")

    # Print statistics to console as well
    print("\nFWHM Statistics:")
    print(f"Mean FWHM: {fit_results_df['fwhm'].mean():.3f}")
    print(f"Std FWHM: {fit_results_df['fwhm'].std():.3f}")
    print(f"Median FWHM: {fit_results_df['fwhm'].median():.3f}")
else:
    print("\nWarning: 'fwhm' column not found in results!")
    print("Available columns:", fit_results_df.columns.tolist())

areas = [result['area'] for result in fit_results]


# Save area statistics
with open(os.path.join(histogram_dir, 'area_statistics.txt'), 'w') as f:
    f.write("Area Statistics:\n")
    f.write(f"Mean Area: {np.mean(areas):.3f}\n")
    f.write(f"Std Area: {np.std(areas):.3f}\n")
    f.write(f"Median Area: {np.median(areas):.3f}\n")

# Print statistics to console
print("\nArea Statistics:")
print(f"Mean Area: {np.mean(areas):.3f}")
print(f"Std Area: {np.std(areas):.3f}")
print(f"Median Area: {np.median(areas):.3f}")

print("\nProcessing complete!")
print(f"Results saved in: {results_dir}") 

# When saving results, use JSON instead of CSV
if fit_results:
    # Convert numpy values to Python native types for JSON serialization
    for result in fit_results:
        for key, value in result.items():
            if isinstance(value, np.number):
                result[key] = float(value)
    
    # Save results as JSON
    with open(os.path.join(results_dir, 'fit_results.json'), 'w') as f:
        json.dump(fit_results, f, indent=2) 

# Create area histogram
plt.figure(figsize=(10, 6))

plt.hist(areas, bins=300, edgecolor='black')
plt.xlim(0, 20.5)
plt.title('Histogram of Peak Areas')
plt.xlabel('Area (arbitrary units)')
plt.ylabel('Count')
plt.savefig(os.path.join(histogram_dir, 'area_histogram.png'))
#plt.close()