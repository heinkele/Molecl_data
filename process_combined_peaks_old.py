import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import pdb
import json

from screening_sample_ssd_old import get_data_folders
from process_single_peak_old import process_single_peak
from screening_sample_ssd_old import is_file_processed
from screening_sample_ssd_old import mark_file_as_processed


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



def process_combined_peaks(root_path):
    data_folders = get_data_folders(root_path)
    print(data_folders)
    for base_dir in data_folders:
        # if is_file_processed(root_path, base_dir, 'combined_peaks_data.json'):
        #     print(f"Skipping root_path processed file: ")
        #     continue
        with open(os.path.join(base_dir, 'combined_peaks_data.json'), 'r') as f:
            peaks_data = json.load(f)

        # Convert to DataFrame for easier processing
        combined_df = pd.DataFrame(peaks_data)

        


        print('The combined data script is running')

        print('combined_df.columns', combined_df.columns)
        
        # Create directories for results and histograms
        results_dir = os.path.join(base_dir, 'peak_fits')
        histogram_dir = os.path.join(results_dir, 'histograms')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(histogram_dir):
            os.makedirs(histogram_dir)

        # Lists to store fitting results
        fit_results = []

        fitted_results = process_single_peak(combined_df)
        array_columns = ['t_start', 't_end', 'dt', 'filtered_signal']
        for column in array_columns:
            combined_df[column] = combined_df[column].apply(np.array)

        for idx in range(len(fitted_results)):
            try:
                # Extract peak data
                #time = row['time']
                t_start = fitted_results[idx]['t_start']
                t_end = fitted_results[idx]['t_end']
                dt = fitted_results[idx]['dt']
                fitted_signals = fitted_results[idx]['result_array']
                t_signal = fitted_results[idx]['t_signal']
                dt_signal = fitted_results[idx]['dt_signal']
                popt_array = fitted_results[idx]['popt_array']
                times_array = fitted_results[idx]['times_array']
                norm_factor = fitted_results[idx]['norm_factor']
                filtered_signal = fitted_results[idx]['signal_array']
                raw_signal_not_norm = fitted_results[idx]['raw_signal_not_norm']
                signal_type = fitted_results[idx]['signal_type']
                
            
                
                t_event = t_start + t_signal[np.argmax(fitted_signals-np.min(fitted_signals))]
                


                integral = np.trapz(fitted_signals-np.min(fitted_signals), t_signal)*norm_factor
                max_displacement = np.max(fitted_signals-np.min(fitted_signals))*norm_factor

                fitted_for_plot = (fitted_signals - np.min(fitted_signals)) * norm_factor
                t = np.asarray(t_signal, dtype=float)
                raw = np.asarray(raw_signal_not_norm, dtype=float)
                filt = np.asarray(filtered_signal, dtype=float) * norm_factor
                n = min(len(t), len(raw), len(filt), len(fitted_for_plot))
                t, raw, filt, fitted_for_plot = t[:n], raw[:n], filt[:n], fitted_for_plot[:n]

                #print(integral, max_displacement)
                fit_results.append({
                    'source_file': base_dir,
                    'peak_index': idx,
                    't_event': t_event,
                    'area': integral,
                    'max_displacement': max_displacement,
                    'fwhm': dt_signal,
                    'signal_type': signal_type,
                    'trace': {
                        't': t.tolist(),
                        'raw': raw.tolist(),
                        'filtered': filt.tolist(),
                        'fitted': fitted_for_plot.tolist()
                    }
                })
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(t_signal, raw_signal_not_norm, 'k-', label='Raw Data')
                ax.plot(t_signal, filtered_signal*norm_factor, 'b-', label='Filtered Data')
                ax.plot(t_signal, (fitted_signals-np.min(fitted_signals))*norm_factor, 'r--', label='Fitted Data')
                ax.set_title(f"Peak fit - {base_dir} - Peak {idx}")
                ax.set_xlabel('Time')
                ax.set_ylabel('Signal')
                ax.legend()
                #plt.show()
                #input('Press Enter to continue')
                plt.savefig(results_dir+ f"/fit_peak_{idx}.png")
                #print(results_dir+ f"/fit_peak_{idx}.png")
                plt.close()

                #mark_file_as_processed(root_path, base_dir, 'combined_peaks_data.json')
            except Exception as e:
                print(f"Error processing peak {idx} from {base_dir}: {str(e)}")
                #input('Press Enter to continue')
                continue



        # Process each peak
        # for idx, row in combined_df.iterrows():
        #     try:
        #         # Extract peak data
        #         #time = row['time']
        #         t_start = row['t_start']
        #         t_end = row['t_end']
        #         dt = row['dt']
        #         #time = np.arange(t_start,t_end+dt,dt)
        #         signal = row['filtered_signal']
        #         n_t = len(signal)
        #         time = np.linspace(t_start,t_end,n_t)
        #         raw_signal = row['raw_signal']
        #         raw_signal_not_norm = row['raw_signal_not_norm']
        #         norm_factor = np.mean(np.array(raw_signal_not_norm)/np.array(raw_signal))
        #         #gradient = row['gradient']
        #         gradient = np.gradient(signal)
        #         gradient *= np.abs(signal)
        #         gradient[gradient>0] = abs(gradient[gradient>0])**0.7
        #         gradient[gradient<0] = -abs(gradient[gradient<0])**0.7
        #         gradient/=np.max(gradient)
                
        #         # Calculate area under the curve
                
                
        #         t_rise_arg = np.argmax(gradient)
        #         dt_coarse = abs(time[t_rise_arg]*2)
                
        #         # Initial parameter guess
        #         p0 = [
        #             np.max(signal) - np.min(signal),  # amplitude
        #             time[np.argmax(signal)],          # center
        #             dt_coarse,          # width
        #             np.min(signal)                    # baseline
        #         ]
        #         #print(p0)
        #         # Define bounds for the parameters
        #         lower_bounds = [
        #             0.05,                    # min amplitude
        #             -20,               # min center (start of time range)
        #             0,                     # min width
        #             -5               # min baseline
        #         ]
                
        #         upper_bounds = [
        #             0.9,                    # max amplitude
        #             20,             # max center (end of time range)
        #             dt_coarse * 2,        # max width
        #             5                # max baseline
        #         ]
                
        #         # Fit the peak with bounds
        #         popt, pcov = curve_fit(peak_function, time, signal, p0=p0, 
        #                             bounds=(lower_bounds, upper_bounds),method='dogbox')
                
        #         # Calculate fit quality
        #         fitted_curve = peak_function(time, *popt)
        #         r_squared = 1 - np.sum((signal - fitted_curve)**2) / np.sum((signal - np.mean(signal))**2)
                
        #         # Calculate FWHM
        #         fwhm = calculate_fwhm(time, fitted_curve)
        #         #print(fwhm)
        #         area = calculate_area(time, fitted_curve-popt[3])*norm_factor
        #         # Add area to fit results
        #         if popt[0] < 0.9 and popt[0]>0.15:
        #             fit_results.append({
        #                 'source_file': row['source_file'],
        #                 'peak_index': row['peak_index'],
        #                 'amplitude': popt[0],
        #                 'center': popt[1],
        #                 'width': popt[2],
        #                 'baseline': popt[3],
        #                 'r_squared': r_squared,
        #                 'fwhm': fwhm,
        #                 'area': area,
        #                 'norm_factor': norm_factor
        #             })
                
        #             # Plot the result
        #             plt.figure(figsize=(10, 6))
        #             plt.plot(time, signal*norm_factor, 'b-', label='Data')
        #             plt.plot(time, (fitted_curve-popt[3])*norm_factor, 'r--', label='Fit')
        #             plt.title(f"Peak fit - {row['source_file']} - Peak {row['peak_index']}")
        #             plt.xlabel('Time')
        #             plt.ylabel('Signal')
        #             plt.legend()
        #             plt.savefig(os.path.join(results_dir, f"fit_{row['source_file']}_peak_{row['peak_index']}.png"))
        #             plt.close()
                    
        #         if idx % 10 == 0:  # Progress update every 10 peaks
        #             print(f"Processed {idx}/{len(combined_df)} peaks")
                    
        #     except Exception as e:
        #         print(f"Error processing peak {idx} from {row['source_file']}: {str(e)}")
        #         continue

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

        plt.figure(figsize=(10, 6))
        plt.scatter(fit_results_df['fwhm'], fit_results_df['max_displacement'], alpha=0.5)
        plt.title('FWHM vs Max Displacement')
        plt.xlabel('FWHM (time units)')
        plt.ylabel('Max Displacement')
        plt.savefig(os.path.join(histogram_dir, 'fwhm_vs_max_displacement.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(fit_results_df['fwhm'], areas, alpha=0.5)
        plt.title('FWHM vs Area')
        plt.xlabel('FWHM (time units)')
        plt.ylabel('Area (arbitrary units)')
        plt.savefig(os.path.join(histogram_dir, 'fwhm_vs_area.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(fit_results_df['max_displacement'], areas, alpha=0.5)
        plt.title('Max Displacement vs Area')
        plt.xlabel('Max Displacement')
        plt.ylabel('Area (arbitrary units)')
        plt.savefig(os.path.join(histogram_dir, 'max_displacement_vs_area.png'))
        plt.close()

        # Add correlation statistics to the text files
        with open(os.path.join(histogram_dir, 'correlation_statistics.txt'), 'w') as f:
            f.write("Correlation Statistics:\n")
            f.write(f"FWHM vs Max Displacement correlation: {np.corrcoef(fit_results_df['fwhm'], fit_results_df['max_displacement'])[0,1]:.3f}\n")
            f.write(f"FWHM vs Area correlation: {np.corrcoef(fit_results_df['fwhm'], areas)[0,1]:.3f}\n")
            f.write(f"Max Displacement vs Area correlation: {np.corrcoef(fit_results_df['max_displacement'], areas)[0,1]:.3f}\n")

        # Print correlation statistics to console
        print("\nCorrelation Statistics:")
        print(f"FWHM vs Max Displacement correlation: {np.corrcoef(fit_results_df['fwhm'], fit_results_df['max_displacement'])[0,1]:.3f}")
        print(f"FWHM vs Area correlation: {np.corrcoef(fit_results_df['fwhm'], areas)[0,1]:.3f}")
        print(f"Max Displacement vs Area correlation: {np.corrcoef(fit_results_df['max_displacement'], areas)[0,1]:.3f}")
#plt.close()


def __main__():
    base_dir = '/Users/hugo/New data/PacBio'
    #data_folders = get_data_folders(base_dir)
    #print(data_folders)
    process_combined_peaks(base_dir)

if __name__ == '__main__':
    __main__()