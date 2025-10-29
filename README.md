# MOLECL Nanopore Analysis Pipeline

## Overview

This repository contains the complete data-processing and analysis pipeline for nanopore experiments performed in the MOLECL project.
The pipeline extracts, processes, and classifies individual event peaks from raw electrical signals to distinguish plasmid topologies (supercoiled, relaxed, linear, etc.).
The workflow runs in a fully modular order, with each script focusing on one well-defined step — from raw JSON extraction to supervised classification.

## Setup & Installation

### Requirements

Python ≥ 3.8
Standard libraries only (no custom dependencies)
Plus the following external packages:
pip install numpy pandas matplotlib scikit-learn
Optionally (for Jupyter notebooks):
pip install notebook

## Data Organization
All samples must be located in the same parent folder, for example:
/Users/hugo/MOLECL_test/Molecl_data_H/

├── processing_status.json

├── Sample_01/

│   ├── peaks_data.json

│   ├── peak_fits/

│   ├── events.csv

│   └── ...

├── Sample_02/

│   ├── peaks_data.json

│   └── ...

└── Sample_03/

    └── ...

## In every script, update the ROOT variable to point to your main data folder:
ROOT = "/Users/hugo/MOLECL_test/Molecl_data_H"

## Pipeline Execution Order
You must execute the scripts in the following order:

• screening_sample.py → Identify and screen valid data folders.
• combine_peaks_data_H.py → Merge peaks_data.json files from all samples.
• process_combined_peaks.py → Extract and process features from the combined peaks.
• Extract_Features.py → Aggregate all processed features into one master dataset.
• Prepare_SVM.py → Aggregate all experiments into one single csv for the models
• Supervised_process.py → Train SVM and evaluate classification.
• (optional) Unsupervised_process.py → Perform clustering, PCA, ICA, etc.
• (optional) Prepare_SVM.py & SVM.ipynb → Interactive fine-tuning and visualization. (not needed, just here to perform some tests)

Each script can be executed directly:
python screening_sample_ssd.py

## Important
If you need to test multiple time the code to adjust parameters, you must delete processing_status.json file as well as the folder you just created, otherwise the signals won't be treated

The parameters used to obtain the results have been stored in CONFIG.ipynb for both abf and dat files.
# CONFIG — Parameter Overview

## Screening_sample

---

### Paths / IO

| Parameter | Description |
|------------|--------------|
| **`ROOT_PATH`** | Root directory containing all samples (e.g. `/Users/hugo/New data/PacBio/sample1/`). |
| **`FILE_GLOB`** | List of file patterns to process (supports `["*.dat", "*.abf"]`). |
| **`SKIP_ALREADY_PROCESSED`** | If `True`, uses `processing_status.json` to skip files already analyzed. |

---

### Header Parsing (.dat only)

| Parameter | Description |
|------------|--------------|
| **`HEADER_SAMPLERATE_DIV`** | Correction factor for the samplerate declared in `.dat` headers. Keep `1` unless your sampling frequency appears incorrect. |

---

### Baseline Estimation

| Parameter | Description |
|------------|--------------|
| **`BASELINE_LP_HZ`** | Cutoff frequency (Hz) of the low-pass filter used to estimate the smooth baseline. Lower = smoother, higher = more responsive. |
| **`BASELINE_ORDER`** | Filter order (Butterworth). `4` is a stable and sharp default. |

---

### Segmentation

| Parameter | Description |
|------------|--------------|
| **`SEG_GRAD_THRESH`** | Threshold on normalized baseline gradient. Lower → more segments (more sensitive to drift), higher → fewer. |
| **`MIN_SEG_LEN`** | Minimum segment length (in samples). Rejects very short windows. |
| **`SEG_VAR_MAX`** | Maximum normalized variance allowed within a segment (rejects noisy regions). |
| **`SEGMENT_LP_HZ`** | Optional extra low-pass filtering before segmentation. `None` disables it. |

---

### Feature Extraction

| Parameter | Description |
|------------|--------------|
| **`FEATURE_MODE`** | Defines how the blockade is computed: `"ratio"` → `1 - I/I_baseline`. |
| **`EPS_DENOM`** | Small constant to prevent division by zero. |
| **`FEATURE_LP_ENABLE`** | Enable smoothing of the feature signal before peak detection. |
| **`FEATURE_LP_MODE`** | `"fixed_hz"` uses an absolute cutoff; `"frac_fs"` uses a fraction of the sampling frequency. |
| **`FEATURE_LP_FIXED_HZ`** | Cutoff frequency (Hz) for feature smoothing if `fixed_hz` mode. |
| **`FEATURE_LP_FRAC`** | Fraction of sampling rate used as cutoff if `frac_fs` mode (ignored otherwise). |
| **`FEATURE_LP_ORDER`** | Filter order for feature smoothing (higher = steeper cutoff). |

---

### Peak Detection

| Parameter | Description |
|------------|--------------|
| **`WINDOW_SIZE`** | Window size (samples) for local analysis (should exceed event duration). |
| **`PEAK_PROMINENCE`** | Minimum prominence for detected peaks (higher = stricter). |
| **`PEAK_MIN_DISTANCE`** | Minimum spacing (samples) between peaks. Prevents double counting. |
| **`SNR_REPORT`** | If `True`, compute and include the signal-to-noise ratio for each detected event. |

---

### Plotting / Debug

| Parameter | Description |
|------------|--------------|
| **`BG_PLOT_DOWNSAMPLE`** | Downsampling factor when plotting long raw traces (reduces file size). |
| **`SEG_PLOT_DOWNSAMPLE`** | Downsampling factor for segment plots (accepted/rejected). |
| **`SAVE_SEGMENT_DEBUG_WHEN_REJECTED`** | Save debug plots for rejected segments (helps tune rejection criteria). |
| **`SAVE_SEGMENT_DEBUG_WHEN_ACCEPTED`** | Save debug plots for accepted segments (useful for visual QC). |

---

### Safety / Numeric

| Parameter | Description |
|------------|--------------|
| **`MIN_BASELINE_MEDIAN`** | Minimal acceptable baseline median. Prevents division by near-zero values. |
| **`MIN_WINDOW_SAMPLES`** | Minimum number of samples required for any local analysis window. |
| **`ENFORCE_DISTANCE_GTE_WINDOW`** | Ensures `PEAK_MIN_DISTANCE ≥ WINDOW_SIZE` to avoid overlapping detections. |

---

### Quick Tuning Guide

| Category | Key Parameters to Adjust First |
|-----------|-------------------------------|
| **Data path** | `ROOT_PATH`, `FILE_GLOB` |
| **Baseline behavior** | `BASELINE_LP_HZ` |
| **Segmentation** | `SEG_GRAD_THRESH`, `SEG_VAR_MAX` |
| **Event detection** | `PEAK_PROMINENCE`, `PEAK_MIN_DISTANCE`, `WINDOW_SIZE` |

For most users, tuning only these will adapt the pipeline to new datasets.  
Other parameters should be changed only if you understand their numerical or physical implications.

---


## Process_combined_peaks

---

### IO / Structure

| Parameter | Description |
|------------|--------------|
| **`ROOT_PATH`** | Root directory containing all samples (e.g. `/Users/hugo/New data/PacBio`). |
| **`INPUT_JSON_NAME`** | Name of the combined JSON file containing all detected peaks (`combined_peaks_data.json`). |
| **`RESULTS_SUBDIR`** | Subdirectory name for saving fit results (e.g. `peak_fits/`). |
| **`HIST_SUBDIR`** | Subdirectory for histogram outputs (e.g. `histograms/`). |
| **`SKIP_IF_FIT_EXISTS`** | If `True`, skip folders where a `fit_results.json` already exists (saves time on reruns). |

---

### Plotting

| Parameter | Description |
|------------|--------------|
| **`SAVE_PLOTS`** | If `True`, save generated plots for each processed peak. |
| **`FIGSIZE`** | Figure size in inches (width, height). Default `(10, 6)`. |
| **`FIG_DPI`** | Resolution (dots per inch) for saved figures. |
| **`PLOT_RAW`** | Plot the raw signal around each detected peak. |
| **`PLOT_FILTERED`** | Plot the filtered signal (useful for debugging smoothing). |
| **`PLOT_FITTED`** | Plot the fitted curve (Gaussian or variant). |
| **`XLABEL_TIME`** | X-axis label for all plots (default: `"Time (µs)"`). |

---

### Histograms & Statistics

| Parameter | Description |
|------------|--------------|
| **`FWHM_HIST_BINS`** | Number of bins for the FWHM (Full Width Half Maximum) histogram. |
| **`AREA_HIST_BINS`** | Number of bins for the peak area histogram. |
| **`AREA_XLIM`** | X-axis limits for area histograms; set to `None` for automatic scaling. |
| **`SAVE_STATS_TXT`** | Save computed statistics (mean, std, etc.) to a text file. |
| **`SAVE_CORR`** | If `True`, save correlation matrices (e.g., between FWHM, area, amplitude). |

---

### Post-Fit Filtering (for Stats/Plots Only)

These thresholds do **not** affect fitting itself — only what is included in the final plots or statistical summaries.

| Parameter | Description |
|------------|--------------|
| **`MIN_FWHM` / `MAX_FWHM`** | Keep only events within this FWHM range (µs). |
| **`MIN_AREA` / `MAX_AREA`** | Keep only events with area within these limits. |
| **`MIN_MAX_DISPLACEMENT` / `MAX_MAX_DISPLACEMENT`** | Filter based on the displacement between raw and fitted maxima (µA or normalized units). |

All values can be set to `None` to disable filtering for that property.

---

### Fit Model & Robustness

| Parameter | Description |
|------------|--------------|
| **`FIT_MODEL`** | Model used for fitting individual peaks: `"gaussian"`, `"supergauss"`, or `"skewgauss"`. |
| **`SUPER_GAUSS_P`** | Shape parameter for the Super-Gaussian (2 = standard Gaussian). |
| **`ROBUST_LOSS`** | Type of robust loss used by `scipy.optimize.curve_fit`: `"linear"`, `"soft_l1"`, `"huber"`, `"cauchy"`. |
| **`ROBUST_F_SCALE`** | Scaling factor for robust loss; higher = less sensitive to outliers. |

---

### Parameter Bounds (Units: µs)

| Parameter | Description |
|------------|--------------|
| **`AMP_MIN` / `AMP_MAX`** | Minimum and maximum allowed peak amplitudes during fitting. |
| **`SIGMA_MIN` / `SIGMA_MAX`** | Bounds for the Gaussian width parameter (σ, in µs). |
| **`MU_PAD_US`** | Extra padding (µs) around the detected peak center when defining the fitting window. Prevents edge artifacts. |

---

### Window Refinement (Optional)

| Parameter | Description |
|------------|--------------|
| **`REFIT_USE_CENTERED_WINDOW`** | If `True`, recenters the fitting window around the local maximum before fitting. |
| **`REFIT_HALF_WINDOW_US`** | Half-width of the window (in µs) used for the centered fit region. |

---

### Quick Tuning Guide

| Category | Key Parameters to Adjust First |
|-----------|-------------------------------|
| **File handling** | `ROOT_PATH`, `INPUT_JSON_NAME` |
| **Visualization** | `SAVE_PLOTS`, `PLOT_*`, `FIG_DPI`, `FIGSIZE` |
| **Model fitting** | `FIT_MODEL`, `ROBUST_LOSS`, `SUPER_GAUSS_P` |
| **Filtering for stats** | `MIN_FWHM`, `MAX_AREA`, `MIN_AREA` |
| **Bounds / constraints** | `AMP_MIN`, `AMP_MAX`, `SIGMA_MAX` |

For most use cases, you’ll only need to tune the fit model, amplitude bounds, and histogram limits.  
More advanced users can explore robust loss functions or window refinements for complex peak shapes.

---

