import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.ndimage import gaussian_filter1d
import os, csv, glob

# ── Styling (match your existing script) ─────────────────────────────────────
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



# ── Peak-detection parameters ─────────────────────────────────────────────────
SMOOTH_SIGMA      = 0.4  # Gaussian σ (residues) applied before peak-finding
MIN_PROMINENCE    = 17    # minimum peak height above surrounding baseline (increased to avoid false positives)
MIN_HEIGHT        = 30   # absolute minimum height for a peak (filters flat noise)
MIN_WIDTH         = 2    # minimum peak width in residues
REL_HEIGHT        = 0.3  # fraction of prominence used for width calculation
PEAK_BUFFER       = 3    # residues to add on each side of detected peaks
LOCAL_WINDOW      = 10   # residues to check on each side for local context (20 total)
STI1_COLORS       = {    # Color by STI1 domain
    1: 'blue',  # Blue for STI1-1
    2: 'green',  # Green for STI1-2
}
STI1_CMAPS = {  # Colormaps for gradient fills
    1: 'Oranges',
    2: 'Greens',
}
ALPHA_SHADE = 0.18

# ── Protein / IDR layout (copy from your main script) ────────────────────────
information = {
    # "Dsk2_full": {"idr": [[75,145],  [223,325]]},
    # "Q9UMX0":   {"idr": [[108,181],  [252,386],  [471,541]]},
    # "Q9UHD9":   {"idr": [[104,177],  [248,378],  [461,576]]},
    # "Q9NRR5":   {"idr": [[84,191],   [262,392],  [477,553]]},
    # "Q9SII8":   {"idr": [[94,167],   [237,380],  [450,504]]},
    # "Q9SII9":   {"idr": [[94,162],   [232,363],  [445,490]]},
    # "Q9VWD9":   {"idr": [[80,134],   [208,321],  [402,498]]},
    "Q9JJP9":   {"idr": [[103,172],  [246,380],  [458,538]]},
    # "G5EFF7":   {"idr": [[84,131],   [201,310],  [382,454]]},
    # "D4A3P1":   {"idr": [[83,170],   [271,374],  [477,548]]},
    # "D4AA63":   {"idr": [[104,174],  [275,381],  [483,591]]},
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def merge_overlapping_peaks(peaks):
    """
    Merge peaks with overlapping ranges.
    For overlapping peaks, keep the one with highest prominence and extend the range.
    
    Args:
        peaks: List of peak dicts with 'start', 'end', 'peak_residue', 'prominence'
    
    Returns:
        List of merged peak dicts
    """
    if len(peaks) <= 1:
        return peaks
    
    # Sort by start position
    sorted_peaks = sorted(peaks, key=lambda x: x['start'])
    
    merged = []
    current = sorted_peaks[0].copy()
    
    for next_peak in sorted_peaks[1:]:
        # Check if peaks overlap (next starts before current ends)
        if next_peak['start'] <= current['end']:
            # Peaks overlap - merge them
            # Extend the range to cover both peaks
            current['end'] = max(current['end'], next_peak['end'])
            
            # Keep the peak_residue and prominence of the more prominent peak
            if next_peak['prominence'] > current['prominence']:
                current['peak_residue'] = next_peak['peak_residue']
                current['prominence'] = next_peak['prominence']
        else:
            # No overlap - save current and move to next
            merged.append(current)
            current = next_peak.copy()
    
    # Don't forget the last peak
    merged.append(current)
    
    return merged


def detect_peaks(residues, values, smooth_sigma=SMOOTH_SIGMA,
                 min_prominence=MIN_PROMINENCE, min_width=MIN_WIDTH,
                 rel_height=REL_HEIGHT, min_height=MIN_HEIGHT, 
                 buffer=PEAK_BUFFER, local_window=LOCAL_WINDOW):
    """
    Returns a list of dicts: {peak_residue, start_residue, end_residue, prominence}
    Values may contain NaN; they are filled before smoothing.
    Uses a two-pass approach:
    1. Find peaks with standard width requirement
    2. Find additional narrow peaks that stand out in their local context
    """
    valid = ~np.isnan(values)
    if valid.sum() < 5:
        return []

    # Find valid data range to avoid extrapolating into NaN regions
    valid_indices = np.where(valid)[0]
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    
    # Fill NaN values: use interpolation for interior NaNs, forward/backward fill for edges
    filled = values.copy()
    if not valid.all():
        # Interior NaNs: interpolate
        interior_nan = ~valid & (np.arange(len(values)) > first_valid) & (np.arange(len(values)) < last_valid)
        if interior_nan.any():
            x_all = np.arange(len(values))
            filled[interior_nan] = np.interp(x_all[interior_nan], x_all[valid], values[valid])
        # Edge NaNs: forward/backward fill (use nearest valid value)
        for i in range(first_valid):
            filled[i] = values[first_valid]
        for i in range(last_valid + 1, len(values)):
            filled[i] = values[last_valid]

    smoothed = gaussian_filter1d(filled, sigma=smooth_sigma)

    # Pass 1: Standard peak detection with width requirement
    peak_idx_standard, props_standard = find_peaks(
        smoothed,
        prominence=min_prominence,
        width=min_width,
        rel_height=rel_height,
        height=min_height,
    )
    
    # Pass 2: Find narrow peaks using local window approach
    # Look for peaks that stand out within their local neighborhood
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
        # Map fractional indices → residue numbers
        l_res = int(np.floor(np.interp(li, x_all, residues)))
        r_res = int(np.ceil(np.interp(ri, x_all, residues)))
        
        # Expand by buffer, but don't go outside valid data range
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
    """Load a section CSV; returns (residues, dict of condition → (mean, std))."""
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
    return residues, conditions


# ── Main loop ─────────────────────────────────────────────────────────────────
os.makedirs('peak_plots', exist_ok=True)
os.makedirs('peak_ranges', exist_ok=True)

all_peaks_rows = []   # accumulated for master summary CSV

for protein, meta in information.items():
    idrs    = meta['idr']
    n_idrs  = len(idrs)
    fig, axes = plt.subplots(1, n_idrs, figsize=(5*n_idrs, 5), sharey=True)
    if n_idrs == 1:
        axes = [axes]

    protein_peak_rows = []
    max_value_all_idrs = 0  # Track max value across ALL IDR subplots

    for sti1 in range(1,3):           # STI1 domains 1 and 2
        for i, section in enumerate(idrs):
            ax = axes[i]
            csv_path = f"csvs/{protein}_IDR{i+1}_STI1{sti1}_excess_prob.csv"
            if not os.path.exists(csv_path):
                continue

            residues, conditions = load_csv(csv_path)

            for cond, (mean_vals, std_vals) in conditions.items():
                color = STI1_COLORS[sti1] 
                
                # Track maximum value (mean + std for upper bound) across all IDRs
                upper_bound = mean_vals +20 # std_vals
                current_max = np.nanmax(upper_bound)
                
                # Debug: print if we find a new maximum
                if current_max > max_value_all_idrs:
                    max_residue_idx = np.nanargmax(upper_bound)
                    print(f"[{protein}] New max: {current_max:.1f} in IDR{i+1}, STI1-{sti1}, "
                          f"condition={cond}, residue={residues[max_residue_idx]}, "
                          f"mean={mean_vals[max_residue_idx]:.1f}, std={std_vals[max_residue_idx]:.1f}")
                    max_value_all_idrs = current_max

                # ---- plot the data ----
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
                    
                    # Calculate global min/max across ALL data for this protein
                    global_vmin = np.nanmin(mean_vals)
                    global_vmax = np.nanmax(mean_vals)
                    
                    # Get colormap for this STI1 domain
                    cmap = plt.cm.get_cmap(STI1_CMAPS[sti1])

                    for pk in peaks:
                        # Get data within peak range
                        peak_mask = (residues >= pk['start']) & (residues <= pk['end'])
                        peak_residues = residues[peak_mask]
                        peak_values = mean_vals[peak_mask]
                        
                        # Normalize using GLOBAL min/max instead of per-peak
                        if global_vmax > global_vmin:
                            normalized_values = (peak_values - global_vmin) / ((global_vmax - global_vmin) *2)
                        else:
                            normalized_values = np.ones_like(peak_values) * 0.5
                        
                        # Create gradient fill...
                        for res, val, norm_val in zip(peak_residues, peak_values, normalized_values):
                            if not np.isnan(val):
                                rgba_color = cmap(norm_val)
                                fill_color = rgba_color #(*rgba_color[:3], ALPHA_SHADE * 2)
                                
                                ax.axvspan(res - 0.5, res + 0.5,
                                        color=fill_color, linewidth=0, zorder=2)


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

    # ---- Set y-limits: highest peak across all panels, rounded to nearest multiple of 50 ----
    if max_value_all_idrs > 0:
        # Round up to nearest multiple of 50
        y_max = np.ceil(max_value_all_idrs / 50) * 50
    else:
        y_max = None
    
    for i, ax in enumerate(axes):
        if y_max is not None:
            ax.set_ylim(0, 300)
        else:
            ax.set_ylim(0, None)
        
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        # apply xlims if defined
        if protein in xlims and i < len(xlims[protein]):
            ax.set_xlim(*xlims[protein][i])

    # ---- per-protein peak CSV ----
    if protein_peak_rows:
        pk_df = pd.DataFrame(protein_peak_rows)
        pk_df.to_csv(f'peak_ranges/{protein}_peaks.csv', index=False)
        print(f"[{protein}] {len(protein_peak_rows)} peaks found → peak_ranges/{protein}_peaks.csv")

    # ---- legend & labels ----
    # build a tidy legend from unique (condition, color) pairs
    handles = [mpatches.Patch(color=c, label=f'STI1-{cond}') for cond, c in STI1_COLORS.items()]
    # fig.legend(handles=handles, bbox_to_anchor=(0.95, 0.6))
    fig.supylabel("Occupancy Fold Change\n($P_{\\mathrm{g}}$/$P_{\\mathrm{g,EV}}$)", x=-0.06,  horizontalalignment='center')
    fig.supxlabel("Residue Number", y=-0.1)
    # fig.suptitle(protein, y=1.01)

    plt.savefig(f'peak_plots/{protein}_peaks.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'peak_plots/{protein}_peaks.svg',          bbox_inches='tight')
    plt.close()

# ---- master summary CSV ----
if all_peaks_rows:
    master_df = pd.DataFrame(all_peaks_rows)
    master_df.to_csv('peak_ranges/ALL_PROTEINS_peaks_summary.csv', index=False)
    print(f"\nMaster summary → peak_ranges/ALL_PROTEINS_peaks_summary.csv")
    print(master_df.to_string(index=False))