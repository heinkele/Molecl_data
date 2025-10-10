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

• screening_sample_ssd.py → Identify and screen valid data folders.
• combine_peaks_data_H.py → Merge peaks_data.json files from all samples.
• process_combined_peaks_H.py → Extract and process features from the combined peaks.
• All_features.py → Aggregate all processed features into one master dataset.
• Supervised_process.py → Train SVM and evaluate classification.
• (optional) Unsupervised_process.py → Perform clustering, PCA, ICA, etc.
• (optional) Prepare_SVM.py & SVM.ipynb → Interactive fine-tuning and visualization. (not needed, just here to perform some tests)

Each script can be executed directly:
python screening_sample_ssd.py

## Important
If you need to test multiple time the code to adjust parameters, you must delete processing_status.json file as well as the folder you just created, otherwise the signals won't be treated

## Key Parameter Reference

### screening_sample_ssd.py

| **Stage**                              | **Parameter(s)**                                                                  | **Default**                                                             | **Role**                                                                                                                                                         | **Effect on Results** |
| -------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| **Signal normalization (baseline)**    | `cutoff = 1e3`, `order = 4`<br>*(used in `butter_lowpass_filter()` for `I_norm`)* | Low-pass filter at 1 kHz smooths the signal and estimates the baseline. | ↑ `cutoff` or ↑ `order`: baseline follows rapid changes but is noisier.<br>↓ values: smoother baseline, risk of oversmoothing slow variations.                   |                       |
| **Segmentation (find sudden changes)** | `threshold = 0.3`, `min_segment_length = 50000`<br>*(in `find_sudden_changes()`)* | Detects gradient jumps and retains long, stable portions.               | ↑ `threshold`: fewer segments, stricter detection.<br>↓ `threshold`: more, shorter segments.<br>↑ `min_segment_length`: keeps only long, stable segments.        |                       |
| **Segment selection**                  | `variation <= 5e-3`                                                               | Selects stable segments with low internal variability.                  | ↓ threshold: only very flat segments retained, fewer peaks.<br>↑ threshold: more noisy segments accepted.                                                        |                       |
| **Filtering before peak detection**    | `cutoff = 5e5`, `order = 6`<br>*(applied to `1 - I_segment / I_norm_segment`)*    | Smooths the normalized signal before `find_peaks`.                      | ↓ `cutoff` or ↓ `order`: less filtering, more residual noise.<br>↑ values: stronger filtering, may flatten real peaks.                                           |                       |
| **Peak extraction window**             | `window_size = 2500`                                                              | Portion of the signal kept around each detected peak.                   | ↑ window: wider context (useful for large peaks) but heavier files.<br>↓ window: may crop parts of peaks.                                                        |                       |
| **Peak detection (`find_peaks`)**      | `prominence = 0.2`, `distance = 1000`                                             | Defines how distinct a peak must be to be counted.                      | ↓ prominence: detects more peaks (incl. noise).<br>↑ prominence: keeps only strong peaks.<br>↓ distance: allows close peaks.<br>↑ distance: merges nearby peaks. |                       |

### process_combined_peaks_H.py

| **Stage**                                   | **Parameter(s)**                                        | **Default**                                                                                   | **Role**                                                                                                                                                   | **Effect on Results** |
| ------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| **Peak extraction (`process_single_peak`)** | `rolling_window = 50`                                   | Controls the smoothing window used for derivative computation and change-signature detection. | ↑ `rolling_window`: detection becomes more robust but less sensitive to rapid variations.<br>↓ `rolling_window`: captures more detail but increases noise. |                       |
| **Transition detection (`find_peaks`)**     | `distance = rolling_window / 2`, `height = 10 * median` | Detects transition points in the smoothed derivative.                                         | ↑ `distance` or ↑ `height`: fewer, more selective transitions.<br>↓ values: detects more transitions, possibly spurious.                                   |                       |
| **Signal fitting (Gaussian / Sigmoid)**     | `curve_fit(..., bounds = ..., maxfev = 10000)`          | Defines parameter bounds and maximum iterations for the curve-fitting procedure.              | Looser `bounds`: allows atypical peaks.<br>↑ `maxfev`: better convergence but slower.<br>↓ `maxfev`: faster but may fail on complex fits.                  |                       |


## Other scripts (summary)

### Script	Function

• combine_peaks_data_H.py	Collects all peaks_data.json files, adds source_file tags, merges into combined dataset.
• All_features.py	Aggregates processed features into a global table (e.g., All_features.csv).
• Supervised_process.py	Loads features, normalizes (via StandardScaler), trains and evaluates SVM/LDA.
• Unsupervised_process.py	Performs unsupervised exploration: PCA, ICA, GMM, clustering visualization.
• Prepare_SVM.py & SVM.ipynb	Used for manual testing, grid search, or visualization of decision boundaries.