import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
from matplotlib.patches import Ellipse
import matplotlib.ticker as mticker

def extract_sample_tag(path):
    """Extract sample tag (sample1, sample2, sample3) from path"""
    match = re.search(r'sample[123]', path)
    return match.group(0) if match else None

def parse_statistics_file(file_path):
    """Parse statistics from a text file"""
    stats = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':')
                    # Clean up the value string and handle empty strings
                    value = value.strip()
                    if value:  # Only try to convert if the string is not empty
                        try:
                            # Try to convert to float, handling scientific notation
                            value = float(value)
                            stats[key.strip()] = value
                        except ValueError as ve:
                            print(f"Warning: Could not convert value '{value}' to float in {file_path}")
                            continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return stats

def gather_statistics(base_dir):
    """Gather statistics from all sample directories, split by signal_type."""
    types = ['gaussian', 'sigmoid']
    area_data = {stype: {'sample1': [], 'sample2': [], 'sample3': []} for stype in types}
    fwhm_data = {stype: {'sample1': [], 'sample2': [], 'sample3': []} for stype in types}
    max_disp_data = {stype: {'sample1': [], 'sample2': [], 'sample3': []} for stype in types}
    # Add "all"
    area_data['all'] = {'sample1': [], 'sample2': [], 'sample3': []}
    fwhm_data['all'] = {'sample1': [], 'sample2': [], 'sample3': []}
    max_disp_data['all'] = {'sample1': [], 'sample2': [], 'sample3': []}
    
    area_files = glob.glob(os.path.join(base_dir, '**/plasmid/*sample*/peak_fits/histograms/area_statistics.txt'), recursive=True)
    
    for file_path in area_files:
        sample_tag = extract_sample_tag(file_path)
        if sample_tag:
            raw_data_path = file_path.replace('histograms/area_statistics.txt', 'fit_results.json')
            try:
                with open(raw_data_path, 'r') as f:
                    data = json.load(f)
                    for peak in data:
                        stype = peak.get('signal_type', 'gaussian')
                        if stype not in types:
                            continue
                        if (
                            'area' in peak and peak['area'] <= 50 and
                            'fwhm' in peak and peak['fwhm'] <= 40 and
                            'max_displacement' in peak
                        ):
                            area_data[stype][sample_tag].append(peak['area'])
                            fwhm_data[stype][sample_tag].append(peak['fwhm'])
                            max_disp_data[stype][sample_tag].append(peak['max_displacement'])
                            # Add to "all"
                            area_data['all'][sample_tag].append(peak['area'])
                            fwhm_data['all'][sample_tag].append(peak['fwhm'])
                            max_disp_data['all'][sample_tag].append(peak['max_displacement'])
            except Exception as e:
                print(f"Error reading {raw_data_path}: {e}")
    
    return area_data, fwhm_data, max_disp_data

def analyze_statistics(area_data, fwhm_data, max_disp_data, signal_type):
    """Analyze and plot the gathered statistics"""
    # Set a modern color scheme
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    plt.style.use('bmh')
    
    def calculate_bins(data, sample):
        n_points = len(data[sample])
        n_bins = max(10, int(np.sqrt(n_points)))
        return n_bins

    def calculate_statistics(data):
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'sem': np.std(data) / np.sqrt(len(data)),  # Standard error of mean
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25)
        }

    # Calculate statistics for each sample
    area_stats = {sample: calculate_statistics(data) for sample, data in area_data.items() if data}
    fwhm_stats = {sample: calculate_statistics(data) for sample, data in fwhm_data.items() if data}

    # 1. Box plots for summary statistics
    plt.figure(figsize=(15, 5))
    
    # Area box plot
    ax1 = plt.subplot(1, 2, 1)
    area_data_list = [area_data[sample] for sample in ['sample1', 'sample2', 'sample3'] if area_data[sample]]
    ax1.boxplot(area_data_list, labels=['sample1', 'sample2', 'sample3'], 
                patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax1.set_title(f'Area Distribution by Sample ({signal_type})')
    ax1.set_ylabel('Area')
    ax1.grid(True, alpha=0.3)
    stats_text = ""
    for s in ['sample1', 'sample2', 'sample3']:
        if s in area_stats:
            stats = area_stats[s]
            stats_text += (
                f"{s}: μ={stats['mean']:.2f}±{stats['sem']*1.96:.2f}\n"
                f"med={stats['median']:.2f}, n={len(area_data[s])}\n"
            )
    stats_text = f"Type: {signal_type}\n" + stats_text
    ax1.text(
        0.98, 0.5, stats_text, ha='right', va='center', transform=ax1.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
    )

    # FWHM box plot
    ax2 = plt.subplot(1, 2, 2)
    fwhm_data_list = [fwhm_data[sample] for sample in ['sample1', 'sample2', 'sample3'] if fwhm_data[sample]]
    ax2.boxplot(fwhm_data_list, labels=['sample1', 'sample2', 'sample3'],
                patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax2.set_title(f'FWHM Distribution by Sample ({signal_type})')
    ax2.set_ylabel('FWHM')
    ax2.grid(True, alpha=0.3)
    stats_text = ""
    for s in ['sample1', 'sample2', 'sample3']:
        if s in fwhm_stats:
            stats = fwhm_stats[s]
            stats_text += (
                f"{s}: μ={stats['mean']:.2f}±{stats['sem']*1.96:.2f}\n"
                f"med={stats['median']:.2f}, n={len(fwhm_data[s])}\n"
            )
    stats_text = f"Type: {signal_type}\n" + stats_text
    ax2.text(
        0.98, 0.5, stats_text, ha='right', va='center', transform=ax2.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
    )
    
    plt.tight_layout()
    plt.savefig(f'box_plots_{signal_type}.png')
    #plt.show()
    plt.close()

    # 2. Violin plots for distribution visualization
    plt.figure(figsize=(15, 5))
    
    # Area violin plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.violinplot(area_data_list, showmeans=True, showmedians=True)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['sample1', 'sample2', 'sample3'])
    ax1.set_title(f'Area Distribution (Violin Plot) ({signal_type})')
    ax1.set_ylabel('Area')
    ax1.grid(True, alpha=0.3)
    stats_text = ""
    for s in ['sample1', 'sample2', 'sample3']:
        if s in area_stats:
            stats = area_stats[s]
            stats_text += (
                f"{s}: μ={stats['mean']:.2f}±{stats['sem']*1.96:.2f}\n"
                f"med={stats['median']:.2f}, n={len(area_data[s])}\n"
            )
    stats_text = f"Type: {signal_type}\n" + stats_text
    ax1.text(
        0.98, 0.5, stats_text, ha='right', va='center', transform=ax1.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
    )

    # FWHM violin plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.violinplot(fwhm_data_list, showmeans=True, showmedians=True)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['sample1', 'sample2', 'sample3'])
    ax2.set_title(f'FWHM Distribution (Violin Plot) ({signal_type})')
    ax2.set_ylabel('FWHM')
    ax2.grid(True, alpha=0.3)
    stats_text = ""
    for s in ['sample1', 'sample2', 'sample3']:
        if s in fwhm_stats:
            stats = fwhm_stats[s]
            stats_text += (
                f"{s}: μ={stats['mean']:.2f}±{stats['sem']*1.96:.2f}\n"
                f"med={stats['median']:.2f}, n={len(fwhm_data[s])}\n"
            )
    stats_text = f"Type: {signal_type}\n" + stats_text
    ax2.text(
        0.98, 0.5, stats_text, ha='right', va='center', transform=ax2.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
    )
    
    plt.tight_layout()
    plt.savefig(f'violin_plots_{signal_type}.png')
    #plt.show()
    plt.close()

    # Box plot for max_displacement
    plt.figure(figsize=(6, 5))
    max_disp_data_list = [max_disp_data[sample] for sample in ['sample1', 'sample2', 'sample3'] if max_disp_data[sample]]
    ax = plt.subplot(1, 1, 1)
    ax.boxplot(max_disp_data_list, labels=['sample1', 'sample2', 'sample3'],
               patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.7))
    ax.set_title(f'Max Displacement Distribution by Sample ({signal_type})')
    ax.set_ylabel('Max Displacement')
    ax.grid(True, alpha=0.3)
    stats_text = ""
    for s in ['sample1', 'sample2', 'sample3']:
        if max_disp_data[s]:
            stats_text += (
                f"{s}: μ={np.mean(max_disp_data[s]):.2f}±{np.std(max_disp_data[s])/np.sqrt(len(max_disp_data[s]))*1.96:.2f}\n"
                f"med={np.median(max_disp_data[s]):.2f}, n={len(max_disp_data[s])}\n"
            )
    stats_text = f"Type: {signal_type}\n" + stats_text
    ax.text(
        0.98, 0.5, stats_text, ha='right', va='center', transform=ax.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
    )
    plt.tight_layout()
    plt.savefig(f'max_displacement_boxplot_{signal_type}.png')
    #plt.show()
    plt.close()

    # Violin plot for max_displacement
    plt.figure(figsize=(6, 5))
    ax = plt.subplot(1, 1, 1)
    ax.violinplot(max_disp_data_list, showmeans=True, showmedians=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['sample1', 'sample2', 'sample3'])
    ax.set_title(f'Max Displacement (Violin Plot) ({signal_type})')
    ax.set_ylabel('Max Displacement')
    ax.grid(True, alpha=0.3)
    stats_text = ""
    for s in ['sample1', 'sample2', 'sample3']:
        if max_disp_data[s]:
            stats_text += (
                f"{s}: μ={np.mean(max_disp_data[s]):.2f}±{np.std(max_disp_data[s])/np.sqrt(len(max_disp_data[s]))*1.96:.2f}\n"
                f"med={np.median(max_disp_data[s]):.2f}, n={len(max_disp_data[s])}\n"
            )
    stats_text = f"Type: {signal_type}\n" + stats_text
    ax.text(
        0.98, 0.5, stats_text, ha='right', va='center', transform=ax.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
    )
    plt.tight_layout()
    plt.savefig(f'max_displacement_violinplot_{signal_type}.png')
    #plt.show()
    plt.close()

    # 3. Enhanced histograms with KDE
    plt.figure(figsize=(15, 5))
    for i, sample in enumerate(['sample1', 'sample2', 'sample3'], 1):
        if area_data[sample]:
            ax = plt.subplot(1, 3, i)
            n_bins = calculate_bins(area_data, sample)
            ax.hist(area_data[sample], bins=n_bins, density=True, alpha=0.7, 
                    color=colors[i-1], label=f'{sample} (n={len(area_data[sample])})')
            
            # Add KDE
            from scipy import stats
            kde = stats.gaussian_kde(area_data[sample])
            x_range = np.linspace(min(area_data[sample]), max(area_data[sample]), 100)
            ax.plot(x_range, kde(x_range), 'k--', alpha=0.5)
            
            ax.set_title(f'Area Distribution - {sample} ({signal_type})')
            ax.set_xlim(0, 25)
            ax.set_xlabel('Area')
            ax.set_ylabel('Density')
            ax.legend()

            if sample in area_stats:
                stats = area_stats[sample]
                stats_text = (
                    f"μ={stats['mean']:.2f}±{stats['sem']*1.96:.2f}\n"
                    f"med={stats['median']:.2f}\nn={len(area_data[sample])}"
                )
                stats_text = f"Type: {signal_type}\n" + stats_text
                ax.text(
                    0.98, 0.5, stats_text, ha='right', va='center', transform=ax.transAxes,
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
                )

            if sample in area_stats and sample in fwhm_stats:
                stats = area_stats[sample]
                stats_f = fwhm_stats[sample]
                stats_text = (
                    f"μA={stats['mean']:.2f}±{stats['sem']*1.96:.2f}\n"
                    f"medA={stats['median']:.2f}\nnA={len(area_data[sample])}\n"
                    f"μF={stats_f['mean']:.2f}±{stats_f['sem']*1.96:.2f}\n"
                    f"medF={stats_f['median']:.2f}\nnF={len(fwhm_data[sample])}"
                )
                stats_text = f"Type: {signal_type}\n" + stats_text
                ax.text(
                    0.98, 0.5, stats_text, ha='right', va='center', transform=ax.transAxes,
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
                )

    stats_text = ""
    for i, s in enumerate(['sample1', 'sample2', 'sample3']):
        if s in area_stats and s in fwhm_stats:
            stats = area_stats[s]
            stats_text += (
                f"{s}: μA={area_stats[s]['mean']:.2f}, μF={fwhm_stats[s]['mean']:.2f}\n"
                f"medA={area_stats[s]['median']:.2f}, medF={fwhm_stats[s]['median']:.2f}\n"
            )
    stats_text = f"Type: {signal_type}\n" + stats_text
    plt.gca().text(
        0.98, 0.5, stats_text, ha='right', va='center', transform=plt.gca().transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig(f'area_histograms_with_kde_{signal_type}.png')
    #plt.show()
    plt.close()

    # 4. Scatter plot with confidence ellipses
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, sample in enumerate(['sample1', 'sample2', 'sample3']):
        if area_data[sample] and fwhm_data[sample]:
            ax.scatter(fwhm_data[sample], area_data[sample], 
                       alpha=0.5, color=colors[i], label=f'{sample} (n={len(area_data[sample])})')
            fwhm = np.array(fwhm_data[sample])
            area = np.array(area_data[sample])
            mean = [np.mean(fwhm), np.mean(area)]
            cov = np.cov(fwhm, area)
            eigenvals, eigenvecs = np.linalg.eig(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
            angle = np.degrees(np.arctan2(*eigenvecs[:,0][::-1]))
            width, height = 2 * np.sqrt(eigenvals)
            ellipse = Ellipse(mean, width, height, angle=angle, fill=False, color=colors[i], alpha=0.8, lw=2)
            ax.add_patch(ellipse)
    stats_text = ""
    for s in ['sample1', 'sample2', 'sample3']:
        if s in area_stats and s in fwhm_stats:
            stats = area_stats[s]
            stats_f = fwhm_stats[s]
            stats_text += (
                f"{s}: μA={stats['mean']:.2f}, μF={stats_f['mean']:.2f}\n"
                f"medA={stats['median']:.2f}, medF={stats_f['median']:.2f}\n"
            )
    stats_text = f"Type: {signal_type}\n" + stats_text
    ax.text(
        0.98, 0.5, stats_text, ha='right', va='center', transform=ax.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
    )
    ax.set_title('Area vs FWHM with Confidence Ellipses')
    ax.set_xlabel('FWHM')
    ax.set_xlim(0, 25)
    ax.set_ylabel('Area')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f'area_vs_fwhm_with_ellipses_{signal_type}.png')
    #plt.show()
    plt.close()

    # 4b. Individual scatter plots with confidence ellipses for each sample
    for i, sample in enumerate(['sample1', 'sample2', 'sample3']):
        if area_data[sample] and fwhm_data[sample]:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(fwhm_data[sample], area_data[sample], 
                       alpha=0.5, color=colors[i], label=f'{sample} (n={len(area_data[sample])})')
            fwhm = np.array(fwhm_data[sample])
            area = np.array(area_data[sample])
            mean = [np.mean(fwhm), np.mean(area)]
            cov = np.cov(fwhm, area)
            eigenvals, eigenvecs = np.linalg.eig(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
            angle = np.degrees(np.arctan2(*eigenvecs[:,0][::-1]))
            width, height = 2 * np.sqrt(eigenvals)
            ellipse = Ellipse(mean, width, height, angle=angle, fill=False, color='k', alpha=1, lw=3)
            ax.add_patch(ellipse)
            stats = area_stats[sample]
            stats_f = fwhm_stats[sample]
            stats_text = (
                f"μA={stats['mean']:.2f}±{stats['sem']*1.96:.2f}\n"
                f"medA={stats['median']:.2f}\nnA={len(area_data[sample])}\n"
                f"μF={stats_f['mean']:.2f}±{stats_f['sem']*1.96:.2f}\n"
                f"medF={stats_f['median']:.2f}\nnF={len(fwhm_data[sample])}"
            )
            stats_text = f"Type: {signal_type}\n" + stats_text
            ax.text(
                0.98, 0.5, stats_text, ha='right', va='center', transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
            )
            ax.set_title(f'Area vs FWHM with Confidence Ellipse: {sample} ({signal_type})')
            ax.set_xlabel('FWHM')
            ax.set_ylabel('Area')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'area_vs_fwhm_with_ellipse_{signal_type}_{sample}.png')
            #plt.show()
            plt.close()

    # 5. Mean and Median with 95% Confidence Intervals vs Sample
    samples = ['sample1', 'sample2', 'sample3']
    means_area = [area_stats[s]['mean'] if s in area_stats else np.nan for s in samples]
    sems_area = [area_stats[s]['sem']*1.96 if s in area_stats else np.nan for s in samples]  # 95% CI
    medians_area = [area_stats[s]['median'] if s in area_stats else np.nan for s in samples]

    means_fwhm = [fwhm_stats[s]['mean'] if s in fwhm_stats else np.nan for s in samples]
    sems_fwhm = [fwhm_stats[s]['sem']*1.96 if s in fwhm_stats else np.nan for s in samples]  # 95% CI
    medians_fwhm = [fwhm_stats[s]['median'] if s in fwhm_stats else np.nan for s in samples]

    x = np.arange(len(samples))

    plt.figure(figsize=(12, 5))

    # Area
    ax1 = plt.subplot(1, 2, 1)
    ax1.errorbar(x, means_area, yerr=sems_area, fmt='o', capsize=5, label='Mean ± 95% CI', color='tab:blue')
    ax1.scatter(x, medians_area, marker='s', color='tab:orange', label='Median')
    ax1.set_xticks(x)
    ax1.set_xticklabels(samples)
    ax1.set_ylabel('Area')
    ax1.set_title('Area: Mean/Median with 95% CI')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    stats_text = ""
    for i, s in enumerate(samples):
        if s in area_stats:
            stats = area_stats[s]
            stats_text += (
                f"{s}: μ={stats['mean']:.2f}±{stats['sem']*1.96:.2f}, "
                f"med={stats['median']:.2f}, n={len(area_data[s])}\n"
            )
    stats_text = f"Type: {signal_type}\n" + stats_text
    ax1.text(
        0.98, 0.5, stats_text, ha='right', va='center', transform=ax1.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
    )

    # FWHM
    ax2 = plt.subplot(1, 2, 2)
    ax2.errorbar(x, means_fwhm, yerr=sems_fwhm, fmt='o', capsize=5, label='Mean ± 95% CI', color='tab:blue')
    ax2.scatter(x, medians_fwhm, marker='s', color='tab:orange', label='Median')
    ax2.set_xticks(x)
    ax2.set_xticklabels(samples)
    ax2.set_ylabel('FWHM')
    ax2.set_title('FWHM: Mean/Median with 95% CI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    stats_text = ""
    for i, s in enumerate(samples):
        if s in fwhm_stats:
            stats = fwhm_stats[s]
            stats_text += (
                f"{s}: μ={stats['mean']:.2f}±{stats['sem']*1.96:.2f}, "
                f"med={stats['median']:.2f}, n={len(fwhm_data[s])}\n"
            )
    stats_text = f"Type: {signal_type}\n" + stats_text
    ax2.text(
        0.98, 0.5, stats_text, ha='right', va='center', transform=ax2.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig(f'mean_median_CI_vs_sample_{signal_type}.png')
    #plt.show()
    plt.close()

    # Save enhanced summary statistics to a text file
    with open(f'statistics_summary_enhanced_{signal_type}.txt', 'w') as f:
        f.write("Enhanced Summary Statistics\n")
        f.write("=========================\n\n")
        
        for sample in ['sample1', 'sample2', 'sample3']:
            f.write(f"\n{sample.upper()}\n")
            f.write("-" * len(sample) + "\n")
            
            if area_data[sample]:
                f.write("\nArea Statistics:\n")
                stats = area_stats[sample]
                f.write(f"Count: {len(area_data[sample])}\n")
                f.write(f"Mean: {stats['mean']:.3f} ± {stats['sem']:.3f} (SEM)\n")
                f.write(f"Median: {stats['median']:.3f}\n")
                f.write(f"Std: {stats['std']:.3f}\n")
                f.write(f"Q25: {stats['q25']:.3f}\n")
                f.write(f"Q75: {stats['q75']:.3f}\n")
                f.write(f"IQR: {stats['iqr']:.3f}\n")
                f.write(f"Min: {np.min(area_data[sample]):.3f}\n")
                f.write(f"Max: {np.max(area_data[sample]):.3f}\n")
            
            if fwhm_data[sample]:
                f.write("\nFWHM Statistics:\n")
                stats = fwhm_stats[sample]
                f.write(f"Count: {len(fwhm_data[sample])}\n")
                f.write(f"Mean: {stats['mean']:.3f} ± {stats['sem']:.3f} (SEM)\n")
                f.write(f"Median: {stats['median']:.3f}\n")
                f.write(f"Std: {stats['std']:.3f}\n")
                f.write(f"Q25: {stats['q25']:.3f}\n")
                f.write(f"Q75: {stats['q75']:.3f}\n")
                f.write(f"IQR: {stats['iqr']:.3f}\n")
                f.write(f"Min: {np.min(fwhm_data[sample]):.3f}\n")
                f.write(f"Max: {np.max(fwhm_data[sample]):.3f}\n")

    # Histogram of max_displacement for each sample
    plt.figure(figsize=(15, 5))
    for i, sample in enumerate(['sample1', 'sample2', 'sample3'], 1):
        if max_disp_data[sample]:
            ax = plt.subplot(1, 3, i)
            n_bins = max(10, int(np.sqrt(len(max_disp_data[sample]))))
            ax.hist(max_disp_data[sample], bins=n_bins, density=True, alpha=0.7, color=colors[i-1])
            ax.set_title(f'Max Displacement - {sample} ({signal_type})')
            ax.set_xlabel('Max Displacement')
            ax.set_ylabel('Density')
            stats_text = (
                f"μ={np.mean(max_disp_data[sample]):.2f}±{np.std(max_disp_data[sample])/np.sqrt(len(max_disp_data[sample]))*1.96:.2f}\n"
                f"med={np.median(max_disp_data[sample]):.2f}\nn={len(max_disp_data[sample])}"
            )
            stats_text = f"Type: {signal_type}\n" + stats_text
            ax.text(
                0.98, 0.5, stats_text, ha='right', va='center', transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
            )
    plt.tight_layout()
    plt.savefig(f'max_displacement_histograms_{signal_type}.png')
    #plt.show()
    plt.close()

    # Scatter: max_displacement vs area
    plt.figure(figsize=(15, 5))
    for i, sample in enumerate(['sample1', 'sample2', 'sample3'], 1):
        if max_disp_data[sample] and area_data[sample]:
            ax = plt.subplot(1, 3, i)
            ax.scatter(max_disp_data[sample], area_data[sample], alpha=0.5, color=colors[i-1])
            ax.set_title(f'Area vs Max Displacement - {sample} ({signal_type})')
            ax.set_xlabel('Max Displacement')
            ax.set_ylabel('Area')
            stats_text = (
                f"corr={np.corrcoef(max_disp_data[sample], area_data[sample])[0,1]:.2f}\n"
                f"n={len(area_data[sample])}"
            )
            stats_text = f"Type: {signal_type}\n" + stats_text
            ax.text(
                0.98, 0.5, stats_text, ha='right', va='center', transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
            )
    plt.tight_layout()
    plt.savefig(f'area_vs_max_displacement_scatter_{signal_type}.png')
    #plt.show()
    plt.close()

    # Scatter: max_displacement vs FWHM
    plt.figure(figsize=(15, 5))
    for i, sample in enumerate(['sample1', 'sample2', 'sample3'], 1):
        if max_disp_data[sample] and fwhm_data[sample]:
            ax = plt.subplot(1, 3, i)
            ax.scatter(max_disp_data[sample], fwhm_data[sample], alpha=0.5, color=colors[i-1])
            ax.set_title(f'FWHM vs Max Displacement - {sample} ({signal_type})')
            ax.set_xlabel('Max Displacement')
            ax.set_ylabel('FWHM')
            stats_text = (
                f"corr={np.corrcoef(max_disp_data[sample], fwhm_data[sample])[0,1]:.2f}\n"
                f"n={len(fwhm_data[sample])}"
            )
            stats_text = f"Type: {signal_type}\n" + stats_text
            ax.text(
                0.98, 0.5, stats_text, ha='right', va='center', transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.7)
            )
    plt.tight_layout()
    plt.savefig(f'fwhm_vs_max_displacement_scatter_{signal_type}.png')
    #plt.show()
    plt.close()

def main():
    base_dir = '/Users/hugo/MOLECL/Molecl_data_H'  # Adjust this path to your base directory
    area_data, fwhm_data, max_disp_data = gather_statistics(base_dir)
    for stype in ['all', 'gaussian', 'sigmoid']:
        print(f"Processing signal_type: {stype}")
        analyze_statistics(area_data[stype], fwhm_data[stype], max_disp_data[stype], stype)

if __name__ == "__main__":
    main()