Current process flow:
- screeneing_sample_ssd.py: manually enter the folder you want to analyze in `get_data_folders` (e.g. change `if os.path.isdir(full_path) and '109_1MKCl_500mV_sample2_100xdilution_10MHz_250429142949' in item:`) and run the script
- Change `base_dir` in combine_peaks_data.py to the folder you just analyzed and run the script
- Do the same with `base_dir` in process_combine_data.py and run it
The results should be written on the ssd
