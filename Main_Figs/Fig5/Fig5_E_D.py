import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.ndimage import gaussian_filter1d
import os, csv, glob

plt.style.use('default')
plt.rcParams.update({
    'svg.fonttype':       'none',
    'axes.linewidth':     2.,
    'xtick.major.size':   14, 'xtick.major.width': 2.,
    'xtick.minor.size':   14, 'xtick.minor.width': 2.,
    'ytick.major.size':   14, 'ytick.major.width': 2.,
    'ytick.minor.size':   14, 'ytick.minor.width': 2.,
    'ytick.labelsize':   28, 'xtick.labelsize':  28,
    'font.size':          28, 'font.sans-serif':   'Arial',
    'legend.title_fontsize': 14,
})

BASE_CSV_PATH = "STI1_occupancy_CSVs"

###Peak parameters 
SMOOTH_SIGMA      = 0.4  # Gaussian σ (residues) applied before peak-finding
MIN_PROMINENCE    = 17    # minimum peak height above surrounding baseline (increased to avoid false positives)
MIN_HEIGHT        = 30   # absolute minimum height for a peak (filters flat noise)
MIN_WIDTH         = 2    # minimum peak width in residues
REL_HEIGHT        = 0.3  # fraction of prominence used for width calculation
PEAK_BUFFER       = 3    # residues to add on each side of detected peaks
LOCAL_WINDOW      = 10   # residues to check on each side for local context (20 total)
STI1_COLORS       = {   
    1: 'blue',  
    2: 'green',  
}
STI1_CMAPS = {
    1: 'Oranges',
    2: 'Greens',
}
ALPHA_SHADE = 0.18

information = {
    "Dsk2_full": {"idr": [[75,145],  [223,325]]},
    "Q9UMX0":   {"idr": [[108,181],  [252,386],  [471,541]]},
    "Q9UHD9":   {"idr": [[104,177],  [248,378],  [461,576]]},
    "Q9NRR5":   {"idr": [[84,191],   [262,392],  [477,553]]},
    "Q9SII8":   {"idr": [[94,167],   [237,380],  [450,504]]},
    "Q9SII9":   {"idr": [[94,162],   [232,363],  [445,490]]},
    "Q9VWD9":   {"idr": [[80,134],   [208,321],  [402,498]]},
    "Q9JJP9":   {"idr": [[103,172],  [246,380],  [458,538]]},
    "G5EFF7":   {"idr": [[84,131],   [201,310],  [382,454]]},
    "D4A3P1":   {"idr": [[83,170],   [271,374],  [477,548]]},
    "D4AA63":   {"idr": [[104,174],  [275,381],  [483,591]]},
}

xlims = {
    "Q9VWD9":  [(80,112),  (236,290),  (420,490)],
    "Q9SII8":  [(95,150),  (265,375),  (465,495)],
    "Q9UMX0":  [(110,175), (260,385),  (475,540)],
    "Q9UHD9":  [(105,172), (255,365),  (470,575)],
    "Q9NRR5":  [(90,180),  (270,380),  (485,545)],
    "Q9SII9":  [(95,155),  (260,360),  (450,485)],
    "G5EFF7":  [(85,115),  (231,285),  (410,450)],
    "D4A3P1":  [(85,165),  (275,370),  (481,545)],
    "D4AA63":  [(110,168), (292,380),  (490,580)],
    "Q9JJP9":  [(105,155), (250,380),  (465,535)],
    "Dsk2_full":[(75,142), (235,325)],
}


def merge_overlapping_peaks(peaks):
    ##merge overlapping peaks 
    if len(peaks) <= 1:
        return peaks
    
    # Sort by start position
    sorted_peaks = sorted(peaks, key=lambda x: x['start'])
    
    merged = []
    current = sorted_peaks[0].copy()
    
    for next_peak in sorted_peaks[1:]:
        if next_peak['start'] <= current['end']:
            current['end'] = max(current['end'], next_peak['end'])
            
            if next_peak['prominence'] > current['prominence']:
                current['peak_residue'] = next_peak['peak_residue']
                current['prominence'] = next_peak['prominence']
        else:
            merged.append(current)
            current = next_peak.copy()
    
    merged.append(current)
    
    return merged


def detect_peaks(residues, values, smooth_sigma=SMOOTH_SIGMA,
                 min_prominence=MIN_PROMINENCE, min_width=MIN_WIDTH,
                 rel_height=REL_HEIGHT, min_height=MIN_HEIGHT, 
                 buffer=PEAK_BUFFER, local_window=LOCAL_WINDOW):
    
    valid = ~np.isnan(values)
    if valid.sum() < 5:
        return []

    valid_indices = np.where(valid)[0]
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    
    # Fill NaN values
    filled = values.copy()
    if not valid.all():
        interior_nan = ~valid & (np.arange(len(values)) > first_valid) & (np.arange(len(values)) < last_valid)
        if interior_nan.any():
            x_all = np.arange(len(values))
            filled[interior_nan] = np.interp(x_all[interior_nan], x_all[valid], values[valid])
        for i in range(first_valid):
            filled[i] = values[first_valid]
        for i in range(last_valid + 1, len(values)):
            filled[i] = values[last_valid]

    smoothed = gaussian_filter1d(filled, sigma=smooth_sigma)

    peak_idx_standard, props_standard = find_peaks(
        smoothed,
        prominence=min_prominence,
        width=min_width,
        rel_height=rel_height,
        height=min_height,
    )
    
    peak_idx_local, props_local = find_peaks(
        smoothed,
        prominence=min_prominence,
        height=min_height,
    )
    
    # Filter local peaks: must be elevated above local baseline
    valid_local_peaks = []
    for pi in peak_idx_local:
        # Skip if already found in standard peaks
        if pi in peak_idx_standard:
            continue
            
        # Define local window around the peak
        window_start = max(first_valid, pi - local_window)
        window_end = min(last_valid + 1, pi + local_window + 1)
        
        # Calculate local baseline (mean of lowest quartile in window)
        window_values = smoothed[window_start:window_end]
        local_baseline = np.percentile(window_values, 25)
        
        # Check if peak is sufficiently elevated above local baseline
        peak_height = smoothed[pi]
        local_prominence = peak_height - local_baseline
        
        # Accept if locally prominent
        if local_prominence >= min_prominence:
            valid_local_peaks.append(pi)
    
    # Combine both sets of peaks
    all_peak_idx = np.concatenate([peak_idx_standard, valid_local_peaks]) if valid_local_peaks else peak_idx_standard
    all_peak_idx = np.unique(all_peak_idx)  # Remove duplicates and sort
    
    if len(all_peak_idx) == 0:
        return []
    
    # Calculate widths and prominences for all peaks
    widths, _, left_ips, right_ips = peak_widths(
        smoothed, all_peak_idx, rel_height=rel_height
    )
    prominences = peak_prominences(smoothed, all_peak_idx)[0]

    results = []
    x_all = np.arange(len(values))
    for pi, li, ri, prom in zip(all_peak_idx, left_ips, right_ips, prominences):
        # Map fractional indices to residue numbers
        l_res = int(np.floor(np.interp(li, x_all, residues)))
        r_res = int(np.ceil(np.interp(ri, x_all, residues)))
        
        # Expand by buffer
        l_res = max(l_res - buffer, residues[first_valid])
        r_res = min(r_res + buffer, residues[last_valid])
        
        results.append({
            'peak_residue': residues[pi],
            'start':        l_res,
            'end':          r_res,
            'prominence':   float(prom),
        })
    
    # Merge overlapping peaks
    results = merge_overlapping_peaks(results)
    
    return results


def load_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['residue_number'])
    residues = df['residue_number'].values.astype(int)
    conditions = {}
    for col in df.columns:
        if col.endswith('_mean'):
            cond = col[:-5]
            conditions[cond] = (
                df[col].values.astype(float),
                df[f'{cond}_std'].values.astype(float),
            )
        elif col == 'mean' and 'std' in df.columns:
            # Handle CSVs with simple 'mean' and 'std' columns (no prefix)
            conditions['data'] = (
                df['mean'].values.astype(float),
                df['std'].values.astype(float),
            )
    return residues, conditions


os.makedirs('peak_plots', exist_ok=True)
os.makedirs('peak_ranges', exist_ok=True)

all_peaks_rows = []   

for protein, meta in information.items():
    idrs    = meta['idr']
    n_idrs  = len(idrs)
    fig, axes = plt.subplots(1, n_idrs, figsize=(5*n_idrs, 5), sharey=True)
    if n_idrs == 1:
        axes = [axes]

    protein_peak_rows = []
    max_value_all_idrs = 0  

    for sti1 in range(1,3):         
        for i, section in enumerate(idrs):
            ax = axes[i]
            csv_path = f"{BASE_CSV_PATH}/{protein}_IDR{i+1}_STI1{sti1}_excess_prob.csv"
            if not os.path.exists(csv_path):
                continue

            residues, conditions = load_csv(csv_path)

            for cond, (mean_vals, std_vals) in conditions.items():
                color = STI1_COLORS[sti1] 
                
                upper_bound = mean_vals + 40 
                current_max = np.nanmax(upper_bound)
                
                if current_max > max_value_all_idrs:
                    max_residue_idx = np.nanargmax(upper_bound)
                    max_value_all_idrs = current_max

                ### Plot 
                ax.plot(residues, mean_vals, color=color,
                        label=f'{cond} STI1-{sti1}', zorder=100)
                ax.scatter(residues, mean_vals, color=color, s=10, zorder=100)
                ax.fill_between(residues,
                                mean_vals - std_vals,
                                mean_vals + std_vals,
                                color=color, alpha=0.15, linewidth=0, zorder=50)

                # ---- find peaks (only for STI1-1) ----
                if sti1 == 1:
                    peaks = detect_peaks(residues, mean_vals)
                    
                    global_vmin = np.nanmin(mean_vals)
                    global_vmax = np.nanmax(mean_vals)
                    
                    cmap = plt.cm.get_cmap(STI1_CMAPS[sti1])

                    for pk in peaks:
                        peak_mask = (residues >= pk['start']) & (residues <= pk['end'])
                        peak_residues = residues[peak_mask]
                        peak_values = mean_vals[peak_mask]
                        
                        if global_vmax > global_vmin:
                            normalized_values = (peak_values - global_vmin) / ((global_vmax - global_vmin) *2)
                        else:
                            normalized_values = np.ones_like(peak_values) * 0.5
                        
                        #! uncomment to show range of peaks colored by value
                        # for res, val, norm_val in zip(peak_residues, peak_values, normalized_values):
                        #     if not np.isnan(val):
                        #         rgba_color = cmap(norm_val)
                        #         fill_color = rgba_color 
                                
                        #         ax.axvspan(res - 0.5, res + 0.5,
                        #                 color=fill_color, linewidth=0, zorder=2)

                        row = {
                            'protein':      protein,
                            'IDR':          i + 1,
                            'STI1_domain':  sti1,
                            'condition':    cond,
                            'peak_residue': pk['peak_residue'],
                            'start':        pk['start'],
                            'end':          pk['end'],
                            'prominence':   round(pk['prominence'], 2),
                        }
                        protein_peak_rows.append(row)
                        all_peaks_rows.append(row)

    if max_value_all_idrs > 0:
        y_max = np.ceil(max_value_all_idrs / 50) * 50
    else:
        y_max = None
    
    for i, ax in enumerate(axes):
        if y_max is not None:
            ax.set_ylim(0, 300)
        else:
            ax.set_ylim(0, None)
        
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        if protein in xlims and i < len(xlims[protein]):
            ax.set_xlim(*xlims[protein][i])

    if protein_peak_rows:
        pk_df = pd.DataFrame(protein_peak_rows)
        pk_df.to_csv(f'peak_ranges/{protein}_peaks.csv', index=False)

    handles = [mpatches.Patch(color=c, label=f'STI1-{cond}') for cond, c in STI1_COLORS.items()]
    fig.supylabel("Occupancy Fold Change\n($P_{\\mathrm{g}}$/$P_{\\mathrm{g,EV}}$)", x=-0.06,  horizontalalignment='center')
    fig.supxlabel("Residue Number", y=-0.1)

    plt.savefig(f'peak_plots/{protein}_peaks.svg',          bbox_inches='tight')
    plt.close()

if all_peaks_rows:
    master_df = pd.DataFrame(all_peaks_rows)
    master_df.to_csv('peak_ranges/ALL_PROTEINS_peaks_summary.csv', index=False)
