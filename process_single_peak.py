import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import convolve1d  


# -------------------- Peak shape models --------------------
def left_sigmoid_function(x, max_value, min_value, center, width):
    # rising (left) logistic
    return (max_value - min_value) / (1.0 + np.exp(-(x - center) / max(width, 1e-12))) + min_value

def right_sigmoid_function(x, max_value, min_value, center, width):
    # falling (right) logistic
    return (max_value - min_value) / (1.0 + np.exp((x - center) / max(width, 1e-12))) + min_value

def gaussian_function(x, max_value, min_value, center, width, sharpness):
    # super-Gaussian if sharpness>1; Gaussian if sharpness≈1
    # NOTE: keeps your original intent: exp(-( ((x-μ)^2/(2σ^2))**sharpness ))
    sigma = max(width, 1e-12)
    core = ((x - center) ** 2) / (2.0 * sigma * sigma)
    return max_value * np.exp(-np.power(core, max(sharpness, 1e-12))) + min_value

def linear_function(x, slope, offset):
    return slope * x + offset


# -------------------- Utilities --------------------
def str_to_array(s):
    try:
        if isinstance(s, str):
            clean_str = s.strip('[]').strip()
            return np.array([float(x) for x in clean_str.split()])
        return s
    except Exception as e:
        print(f"Error converting string to array: {e}")
        print(f"Problematic string: {s}")
        return None

def _safe_rolling(series, window, center=True, func="mean"):
    """Centered rolling with NaNs handled."""
    r = getattr(series.rolling(window=window, center=center), func)()
    return np.nan_to_num(r.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

def _clip_p0_into_bounds(p0, lb, ub):
    p0 = np.asarray(p0, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    eps = 1e-12
    return np.minimum(np.maximum(p0, lb + eps), ub - eps)


# -------------------- Main function --------------------
def process_single_peak(combined_df, rolling_window=50):
    """
    Input:  combined_df = DataFrame read from combined_peaks_data.json
            It must contain (per peak) at least:
            - 'filtered_signal', 'raw_signal', 'raw_signal_not_norm'
            - 't_start', 't_end', 'dt'  (µs-based; dt can be scalar)
    Output: list of dicts with:
            'result_array', 't_signal', 'dt_signal', 't_start', 't_end', 'dt',
            'norm_factor', 'popt_array', 'times_array', 'signal_array',
            'raw_signal_not_norm', 'signal_type', 'signal_type_index'
    """
    # Ensure list-like columns are arrays
    array_columns = [
        't_start', 't_end', 'dt',
        'filtered_signal', 'norm_signal',
        'raw_signal', 'raw_signal_not_norm'
    ]
    for column in array_columns:
        if column in combined_df.columns:
            combined_df[column] = combined_df[column].apply(np.array)

    results = []

    for signal_index in range(len(combined_df)):
        # --- Pull signals ---
        signal = np.asarray(combined_df.iloc[signal_index]['filtered_signal'], dtype=float)
        if signal.size < 8:
            # too short to fit anything meaningful
            continue

        # Build time axis in µs (JSON already stored t_* in µs)
        t_start_val = combined_df.iloc[signal_index]['t_start']
        t_end_val   = combined_df.iloc[signal_index]['t_end']
        # If these came in as arrays (size 1), take scalar
        if np.ndim(t_start_val) > 0:
            t_start_val = float(np.ravel(t_start_val)[0])
        if np.ndim(t_end_val) > 0:
            t_end_val = float(np.ravel(t_end_val)[-1])

        t_signal = np.linspace(t_start_val, t_end_val, len(signal))

        # dt per sample in µs (from the constructed t_signal)
        dt_us = float(np.median(np.diff(t_signal))) if len(t_signal) > 1 else 1.0

        # Center signal (remove local median)
        signal = signal - np.median(signal)

        # --- Change signature & peak detection (robust) ---
        s = pd.Series(signal)
        # Centered rolling stats
        rolling_mean = _safe_rolling(s, rolling_window, center=True, func="mean")
        rolling_std  = _safe_rolling(s, rolling_window, center=True, func="std")
        rolling_std_of_mean = _safe_rolling(pd.Series(rolling_mean), rolling_window, center=True, func="std")

        change_signature = rolling_std * rolling_std_of_mean
        if np.max(np.abs(change_signature)) > 0:
            change_signature = change_signature / np.max(np.abs(change_signature))
        else:
            change_signature = np.zeros_like(signal)

        # Peaks on change signature
        cs_median = np.median(change_signature)
        height_thr = 10.0 * cs_median  # your original heuristic
        distance_samples = max(1, int(rolling_window / 2))
        peaks, _ = find_peaks(change_signature, distance=distance_samples, height=height_thr)

        # If nothing detected, fall back to a single broad Gaussian around max
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(np.abs(signal)))])

        # Peak widths in SAMPLES (convert to µs later)
        widths = peak_widths(change_signature, peaks, rel_height=0.5)
        width_samples_vec = widths[0] if len(widths) >= 1 else np.array([max(rolling_window, 8.0)])

        # Prepare containers
        popt_array = []
        times_array = np.array([])
        signal_array = np.full_like(t_signal, np.nan, dtype=float)

        t_signal_rise = None
        t_signal_fall = None
        dt_signal = 0.0
        left_baseline = 0.0
        signal_type = ''

        # --- Decide model: Gaussian (one broad) vs Sigmoids (composite steps) ---
        # Keep your original quick criterion but guard indices
        if len(peaks) >= 1 and len(width_samples_vec) >= 1:
            left_term  = peaks[0] + width_samples_vec[0] * 1.0
            right_term = peaks[-1] - width_samples_vec[-1] * 1.0
            use_gaussian = (left_term - right_term) > 0
        else:
            use_gaussian = True

        if use_gaussian:
            # Single Gaussian fit over full window
            width_samples = float(width_samples_vec[0] if len(width_samples_vec) > 0 else max(rolling_window, 8.0))
            width_us = max(width_samples * dt_us, dt_us)

            # p0: [max_value, min_value, center(µs), width(µs), sharpness]
            p0 = [
                float(np.max(signal)),
                float(signal[0]),
                float(t_signal[peaks[0]]),
                float(width_us / 2.0),
                1.0
            ]

            center_pad_us = max(width_us / 2.0, 5.0 * dt_us)
            width_min_us  = max(0.1 * width_us, 0.2 * dt_us)
            width_max_us  = max(3.0  * width_us, 5.0 * dt_us)

            lb = [
                p0[0] - abs(p0[0]) / 2.0,
                p0[1] - abs(p0[1]) / 2.0,
                p0[2] - center_pad_us,
                width_min_us,
                0.5  # sharpness lower
            ]
            ub = [
                p0[0] + abs(p0[0]),
                p0[1] + abs(p0[1]),
                p0[2] + center_pad_us,
                width_max_us,
                2.0  # sharpness upper
            ]

            p0 = _clip_p0_into_bounds(p0, lb, ub)

            try:
                popt, pcov = curve_fit(
                    gaussian_function, t_signal, signal,
                    p0=p0, bounds=(lb, ub), maxfev=10000
                )
                dt_signal = float(popt[3])  # width parameter in µs
            except Exception as e:
                print(f"Error fitting gaussian function (idx {signal_index}): {e}")
                print(f"  p0: {p0}\n  bounds: {(lb, ub)}")
                continue

            popt_array.append(popt)
            signal_array = gaussian_function(t_signal, *popt)
            signal_type = 'gaussian'

        else:
            # Composite rising/falling sigmoids around each detected step
            for ii in range(len(peaks)):
                window_width_samples = int(max(8, width_samples_vec[ii] * 2.0))
                window_start = int(max(0, peaks[ii] - window_width_samples))
                window_end   = int(min(len(signal), peaks[ii] + window_width_samples))

                window_signal = signal[window_start:window_end]
                window_time   = t_signal[window_start:window_end]
                if window_signal.size < 8:
                    continue

                # keep a record of the time window
                times_array = np.append(times_array, window_time)

                # Determine sign: we use the derivative of a smoothed signal
                # (rebuild a quick derivative here to avoid NaNs)
                s_mean = _safe_rolling(pd.Series(signal), rolling_window, center=True, func="mean")
                derivative = np.gradient(s_mean)
                derivative = np.nan_to_num(derivative, nan=0.0)
                rising = derivative[peaks[ii]] > 0

                # Convert window width from samples to µs for bounds
                window_width_us = max(window_width_samples * dt_us, dt_us)

                if rising:
                    # left (rising) sigmoid
                    p0 = [
                        float(window_signal[-1]),          # max_value
                        float(window_signal[0]),           # min_value
                        float(t_signal[peaks[ii]]),        # center (µs)
                        float(window_width_us / 10.0)      # width (µs)
                    ]
                    center_pad_us = max(window_width_us / 2.0, 5.0 * dt_us)
                    width_min_us  = max(0.1 * window_width_us, 0.2 * dt_us)
                    width_max_us  = max(2.0 * window_width_us, 5.0 * dt_us)

                    lb = [
                        p0[0] - abs(p0[0]) / 2.0,
                        p0[1] - abs(p0[1]) / 2.0,
                        p0[2] - center_pad_us,
                        width_min_us
                    ]
                    ub = [
                        p0[0] + abs(p0[0]),
                        p0[1] + abs(p0[1]),
                        p0[2] + center_pad_us,
                        width_max_us
                    ]

                    p0 = _clip_p0_into_bounds(p0, lb, ub)

                    try:
                        popt, pcov = curve_fit(
                            left_sigmoid_function, window_time, window_signal,
                            p0=p0, bounds=(lb, ub), maxfev=10000
                        )
                    except Exception as e:
                        print(f"Error fitting left sigmoid (idx {signal_index}, step {ii}): {e}")
                        print(f"  p0: {p0}\n  bounds: {(lb, ub)}")
                        continue

                    if ii == 0:
                        left_baseline = popt[1]
                    popt_array.append(popt)
                    signal_array[window_start:window_end] = left_sigmoid_function(window_time, *popt)
                    if ii == 0:
                        # fill before window with extension of first rise
                        signal_array[:window_start] = left_sigmoid_function(t_signal[:window_start], *popt)
                        t_signal_rise = float(popt[2])

                else:
                    # right (falling) sigmoid
                    p0 = [
                        float(window_signal[0]),           # max_value
                        float(window_signal[-1]),          # min_value
                        float(t_signal[peaks[ii]]),        # center (µs)
                        float(window_width_us / 10.0)      # width (µs)
                    ]
                    center_pad_us = max(window_width_us / 2.0, 5.0 * dt_us)
                    width_min_us  = max(0.1 * window_width_us, 0.2 * dt_us)
                    width_max_us  = max(2.0 * window_width_us, 5.0 * dt_us)

                    lb = [
                        p0[0] - abs(p0[0]) / 2.0,
                        p0[1] - abs(p0[1]) / 2.0,
                        p0[2] - center_pad_us,
                        width_min_us
                    ]
                    ub = [
                        p0[0] + abs(p0[0]),
                        p0[1] + abs(p0[1]),
                        p0[2] + center_pad_us,
                        width_max_us
                    ]

                    p0 = _clip_p0_into_bounds(p0, lb, ub)

                    try:
                        popt, pcov = curve_fit(
                            right_sigmoid_function, window_time, window_signal,
                            p0=p0, bounds=(lb, ub), maxfev=10000
                        )
                    except Exception as e:
                        print(f"Error fitting right sigmoid (idx {signal_index}, step {ii}): {e}")
                        print(f"  p0: {p0}\n  bounds: {(lb, ub)}")
                        continue

                    if ii == len(peaks) - 1:
                        # align final baseline with the first rise baseline
                        popt[1] = left_baseline
                    popt_array.append(popt)
                    signal_array[window_start:window_end] = right_sigmoid_function(window_time, *popt)
                    if ii == len(peaks) - 1:
                        # fill after last window with extension of fall
                        signal_array[window_end:] = right_sigmoid_function(t_signal[window_end:], *popt)
                        t_signal_fall = float(popt[2])

            signal_type = 'sigmoid'

            # Width measure for sigmoid composite: time between rise/fall centers
            if (t_signal_rise is not None) and (t_signal_fall is not None):
                dt_signal = abs(t_signal_fall - t_signal_rise)
            else:
                # fallback: try to estimate from occupied window
                valid_idx = np.where(~np.isnan(signal_array))[0]
                if valid_idx.size >= 2:
                    dt_signal = float(t_signal[valid_idx[-1]] - t_signal[valid_idx[0]])
                else:
                    dt_signal = float(max(width_samples_vec) * dt_us if len(width_samples_vec) > 0 else rolling_window * dt_us)

        # --- Fill any remaining gaps in the model with a spline over known points ---
        mask = np.isnan(signal_array)
        if np.all(mask):
            # If we have no modeled points (should be rare), use the raw centered signal
            signal_array_filled = signal.copy()
        else:
            # Cubic spline over fitted regions
            cs = CubicSpline(t_signal[~mask], signal_array[~mask])
            signal_array_filled = cs(t_signal)

        # Build result array baseline at zero for downstream area/fit (consistent with your earlier return)
        result_array = signal_array_filled - np.min(signal_array_filled)

        # Normalization factor: mean(raw_not_norm/raw_signal)
        raw_signal = np.asarray(combined_df.iloc[signal_index]['raw_signal'], dtype=float)
        raw_signal_not_norm = np.asarray(combined_df.iloc[signal_index]['raw_signal_not_norm'], dtype=float)
        # Guard division
        if raw_signal.size == raw_signal_not_norm.size and raw_signal.size > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.true_divide(raw_signal_not_norm, raw_signal)
                ratio = np.where(np.isfinite(ratio), ratio, np.nan)
            # robust mean (ignore NaN/inf)
            norm_factor = float(np.nanmean(ratio))
            if not np.isfinite(norm_factor):
                norm_factor = 1.0
        else:
            norm_factor = 1.0

        # Package result
        results.append({
            'result_array': result_array,                            # modeled & baseline-shifted
            't_signal': t_signal,                                   # µs
            'dt_signal': float(dt_signal),                          # width in µs
            't_start': combined_df.iloc[signal_index]['t_start'],   # as in JSON (may be array)
            't_end':   combined_df.iloc[signal_index]['t_end'],
            'dt':      combined_df.iloc[signal_index]['dt'],
            'norm_factor': float(norm_factor),
            'popt_array': popt_array,
            'times_array': times_array,                              # concatenated fit window times
            'signal_array': signal,                                  # centered filtered signal (your downstream uses this)
            'raw_signal_not_norm': raw_signal_not_norm,
            'signal_type': signal_type,
            'signal_type_index': signal_index
        })

    return results


# -------------------- Standalone test --------------------
def __main__():
    base_dir = '/path/to/a/folder/that/contains/combined_peaks_data.json'
    with open(os.path.join(base_dir, 'combined_peaks_data.json'), 'r') as f:
        peaks_data = json.load(f)
    combined_df = pd.DataFrame(peaks_data)
    results = process_single_peak(combined_df, rolling_window=50)
    print(f"Processed {len(results)} peaks.")

if __name__ == '__main__':
    __main__()
