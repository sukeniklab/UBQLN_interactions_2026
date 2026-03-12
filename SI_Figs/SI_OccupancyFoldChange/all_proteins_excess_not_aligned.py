import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from Bio import Phylo
from io import StringIO

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['font.size'] = 28
plt.rcParams['font.sans-serif'] = 'Arial'

def smooth_log_histogram(log_histogram, smooth, fill_value=None):
    """Smooth log histogram with gap filling for missing data"""
    valid_mask = ~np.isnan(log_histogram)
    
    if np.sum(valid_mask) < 4:
        print("Warning: Too few valid points for spline interpolation")
        return log_histogram
    
    x_coords = np.arange(len(log_histogram))
    valid_x = x_coords[valid_mask]
    valid_y = log_histogram[valid_mask]
    
    if fill_value is None:
        fill_value = np.percentile(valid_y, 25)
    
    filled_data = log_histogram.copy()
    filled_data[~valid_mask] = fill_value
    
    try:
        spline = UnivariateSpline(x_coords, filled_data, s=smooth, k=3)
        smoothed = spline(x_coords)
        return smoothed
    except Exception as e:
        print(f"Spline fitting failed: {e}")
        return log_histogram

# Simulation parameters
numReplicas = 20
sim_time = 70
print_interval = 0.01
snaps = numReplicas * int((sim_time/print_interval) - (0.05 * (sim_time/print_interval)))
snaps_large = snaps * 10

trials = list(range(1, 10))
configs = [""]
colors = [['blue', 'green']]
labels = [['STI1-1', 'STI1-2']]
linestyle = [['-', '-']]
alpha = [[1, 1]]

# All protein information
information = {
    "P48510": {"idr": [[75, 145], [223, 325]]},  # Dsk2_full
    "Q9UMX0": {"idr": [[108, 181], [252, 386], [471, 541]]},
    "Q9UHD9": {"idr": [[104, 177], [248, 378], [461, 576]]},
    "Q9NRR5": {"idr": [[84, 191], [262, 392], [477, 553]]},
    "Q9SII8": {"idr": [[94, 167], [237, 380], [450, 504]]},
    "Q9SII9": {"idr": [[94, 162], [232, 363], [445, 490]]},
    "Q9VWD9": {"idr": [[80, 134], [208, 321], [402, 498]]},
    "Q9JJP9": {"idr": [[103, 172], [246, 380], [458, 538]]},
    "G5EFF7": {"idr": [[84, 131], [201, 310], [382, 454]]},
    "D4A3P1": {"idr": [[83, 170], [271, 374], [477, 548]]},
    "D4AA63": {"idr": [[104, 174], [275, 381], [483, 591]]},
}

# Custom x-axis limits for each protein's IDRs
protein_xlims = {
    "Q9VWD9": [(80, 112), (236, 290), (420, 490)],
    "Q9SII8": [(95, 150), (265, 375), (465, 495)],
    "Q9UMX0": [(110, 175), (260, 385), (475, 540)],
    "Q9UHD9": [(105, 172), (255, 365), (470, 575)],
    "Q9NRR5": [(90, 180), (270, 380), (485, 545)],
    "Q9SII9": [(95, 155), (260, 360), (450, 485)],
    "G5EFF7": [(85, 115), (231, 285), (410, 450)],
    "D4A3P1": [(85, 165), (275, 370), (481, 545)],
    "D4AA63": [(110, 168), (292, 380), (490, 580)],
    "Q9JJP9": [(105, 155), (250, 380), (465, 535)],
    "P48510": [(75, 142), (235, 325)],  # Dsk2_full
}

prot_name_map = {
    "P48510": "DSK2_YEAST",
    "Q9UMX0": "UBQL1_HUMAN",
    "Q9UHD9": "UBQL2_HUMAN",
    "Q9NRR5": "UBQL4_HUMAN",
    "Q9SII8": "Dsk2B_PLANT",
    "Q9SII9": "Dsk2A_PLANT",
    "D4A3P1": "UBQL4_RAT",
    "D4AA63": "UBQL2_RAT",
    "Q9JJP9": "UBQL1_RAT",
    "Q9VWD9": "UBQN_Fly",
    "G5EFF7": "UBQL_CElegans",
}

# Phylogenetic tree setup
TREE_NAME_TO_PID = {
    "C. Elegans UBQN": "G5EFF7",
    "Human UBQLN2": "Q9UHD9",
    "Rat UBQLN2": "D4AA63",
    "Human UBQLN1": "Q9UMX0",
    "Rat UBQLN1": "Q9JJP9",
    "Human UBQLN4": "Q9NRR5",
    "Rat UBQLN4": "D4A3P1",
    "Fly UBQN": "Q9VWD9",
    "Plant Dsk2A": "Q9SII9",
    "Plant Dsk2B": "Q9SII8",
    "Yeast Dsk2": "P48510",
}

NEWICK = (
    "((('C. Elegans UBQN':0.21974,((((('Mouse UBQL2':0.01422,'Rat UBQLN2':0.01422)"
    "Inner3:0.03392,'Human UBQLN2':0.04814)Inner7:0.06534,((('Mouse UBQL1':0.01313,"
    "'Rat UBQLN1':0.01313)Inner2:0.02872,'Human UBQLN1':0.04185)Inner6:0.04007,"
    "'Frog UBQLN4':0.08192)Inner8:0.03156)Inner9:0.06903,('Zebra Fish UBQN':0.12486,"
    "(('Mouse UBQL4':0.00821,'Rat UBQLN4':0.00821)Inner1:0.01887,'Human UBQLN4':0.02708)"
    "Inner4:0.09778)Inner10:0.02608)Inner11:0.06816,'Fly UBQN':0.19302)Inner12:0.02672)"
    "Inner13:0.01636,('Plant Dsk2A':0.04048,'Plant Dsk2B':0.04048)Inner5:0.19562)"
    "Inner14:0.14055,'Yeast Dsk2':0.36029)Inner15:0.00000;"
)

tree = Phylo.read(StringIO(NEWICK), 'newick')
tree.root_with_outgroup({'name': 'Yeast Dsk2'})
all_tips = [c.name for c in tree.get_terminals()]

# Create display order based on tree
_pid_to_tree_pos = {}
for rank, tip_name in enumerate(all_tips):
    pid = TREE_NAME_TO_PID.get(tip_name)
    if pid and pid in prot_name_map and pid not in _pid_to_tree_pos:
        _pid_to_tree_pos[pid] = rank

DISPLAY_ORDER = sorted(_pid_to_tree_pos.keys(), key=lambda p: _pid_to_tree_pos[p])
DEEP_PIDS = {'P48510', 'Q9SII9', 'Q9SII8', 'G5EFF7', 'Q9VWD9'}
boundary_idx = max(i for i, p in enumerate(DISPLAY_ORDER) if p in DEEP_PIDS)

IDR_LABELS = {0: 'IDR1', 1: 'IDR2', 2: 'IDR3'}
GAP_SIZE = 10

# Calculate the min and max residue numbers for each IDR across all proteins
idr_residue_ranges = {}
for protein in DISPLAY_ORDER:
    idrs = information[protein]["idr"]
    for i, section in enumerate(idrs):
        if protein in protein_xlims and i < len(protein_xlims[protein]):
            xlim_min, xlim_max = protein_xlims[protein][i]
        else:
            xlim_min, xlim_max = section[0], section[1]
        
        if i not in idr_residue_ranges:
            idr_residue_ranges[i] = {'min': xlim_min, 'max': xlim_max}
        else:
            idr_residue_ranges[i]['min'] = min(idr_residue_ranges[i]['min'], xlim_min)
            idr_residue_ranges[i]['max'] = max(idr_residue_ranges[i]['max'], xlim_max)

# Create aligned positions based on actual residue ranges
aligned_positions = {}
cumulative_pos = 0
for i in sorted(idr_residue_ranges.keys()):
    res_min = idr_residue_ranges[i]['min']
    res_max = idr_residue_ranges[i]['max']
    span = res_max - res_min + 1
    
    aligned_positions[i] = {
        'plot_start': cumulative_pos,
        'plot_end': cumulative_pos + span,
        'res_min': res_min,
        'res_max': res_max
    }
    cumulative_pos += span + GAP_SIZE

total_width = cumulative_pos - GAP_SIZE

# Calculate all tick positions and labels across ALL IDR regions
all_tick_info = []  # List of (plot_x, residue_number, idr_index)

for i in aligned_positions.keys():
    plot_start = aligned_positions[i]['plot_start']
    res_min = aligned_positions[i]['res_min']
    res_max = aligned_positions[i]['res_max']
    
    # Find first residue divisible by 30 in this range
    start_res = ((res_min // 30) + 1) * 30
    
    # Generate ticks every 30 residues
    for res in range(start_res, res_max + 1, 30):
        plot_x = plot_start + (res - res_min)
        all_tick_info.append((plot_x, res, i))

# Sort by plot position
all_tick_info.sort(key=lambda x: x[0])

# Extract positions and labels
all_tick_positions = [x[0] for x in all_tick_info]
all_tick_labels = [str(x[1]) for x in all_tick_info]

# Create combined figure
fig, axes = plt.subplots(len(DISPLAY_ORDER), 1,
                         figsize=(20, 3.3 * len(DISPLAY_ORDER)),
                         sharex=False)
if len(DISPLAY_ORDER) == 1:
    axes = [axes]

# Plot each protein
for prot_idx, protein in enumerate(DISPLAY_ORDER):
    ax = axes[prot_idx]
    is_deep = protein in DEEP_PIDS
    idrs = information[protein]["idr"]
    num_idrs = len(idrs)
    
    all_max_values = []
    
    for con, config in enumerate(configs):
        for sti1 in range(2):
            # For yeast, always use blue color
            if protein == "P48510":
                c = 'blue'
            else:
                c = colors[con][sti1]
            l = linestyle[con][sti1]
            lab = labels[con][sti1]
            al = alpha[con][sti1]
            
            for i, section in enumerate(idrs):
                all_ratios = np.zeros((len(trials), len(np.arange(section[0], section[1]+1, 1))))
                
                for trial in trials:
                    try:
                        # Load histograms - handle Dsk2 special case
                        if protein == "P48510":
                            histogram = np.load(f"old_pipeline_histos/Dsk2_full{config}_trial{trial}_innerVol_idr{section[0]}_{section[1]}.npy")
                            histogram /= snaps
                            histogram_ev = np.load(f"old_pipeline_histos/Dsk2_full_EV_trial{trial}_innerVol_idr{section[0]}_{section[1]}.npy")
                            histogram_ev /= snaps_large
                        else:
                            histogram = np.load(f"old_pipeline_histos/{protein}{config}_trial{trial}_STI1{sti1+1}_idr{section[0]}_{section[1]}.npy")
                            histogram /= snaps_large
                            histogram_ev = np.load(f"old_pipeline_histos/{protein}_EV_trial{trial}_STI1{sti1+1}_idr{section[0]}_{section[1]}.npy")
                            histogram_ev /= snaps_large
                        
                        # Process EV histogram
                        log_ev = np.log10(histogram_ev)
                        log_ev[np.isinf(log_ev)] = np.nan
                        
                        # Smooth based on IDR region
                        if i == 0:
                            smoothed_log_ev = smooth_log_histogram(log_ev[:-4], 1)
                            smoothed_log_ev_full = np.full_like(log_ev, np.nan)
                            smoothed_log_ev_full[:-4] = smoothed_log_ev
                        elif i == 1:
                            smoothed_log_ev = smooth_log_histogram(log_ev[4::], 2.4)
                            smoothed_log_ev_full = np.full_like(log_ev, np.nan)
                            smoothed_log_ev_full[4::] = smoothed_log_ev
                        else:
                            smoothed_log_ev = smooth_log_histogram(log_ev[4::], 2.4)
                            smoothed_log_ev_full = np.full_like(log_ev, np.nan)
                            smoothed_log_ev_full[4::] = smoothed_log_ev
                        
                        smooth_ev = 10**smoothed_log_ev_full
                        ratio = histogram / smooth_ev
                        ratio[np.isinf(ratio)] = np.nan
                        
                        all_ratios[trial-1, :] = ratio
                    
                    except Exception as e:
                        continue
                
                # Remove empty trials
                all_ratios = all_ratios[~np.all(all_ratios == 0, axis=1)]
                
                # Skip if no valid data
                if len(all_ratios) == 0 or np.all(np.isnan(all_ratios)):
                    continue
                
                # Calculate statistics
                mean_ratio = np.nanmean(all_ratios, axis=0)
                std_ratio = np.nanstd(all_ratios, axis=0)
                residues = np.arange(section[0], section[1]+1, 1)
                
                # Apply custom x-limits
                if protein in protein_xlims and i < len(protein_xlims[protein]):
                    xlim_min, xlim_max = protein_xlims[protein][i]
                    mask = (residues >= xlim_min) & (residues <= xlim_max)
                    residues = residues[mask]
                    mean_ratio = mean_ratio[mask]
                    std_ratio = std_ratio[mask]
                
                # Track max for y-limit
                all_max_values.append(np.nanmax(mean_ratio + std_ratio))
                
                # Map residues to plot x-coordinates
                # Residue number maps directly to position within the IDR slot
                plot_start = aligned_positions[i]['plot_start']
                res_min = aligned_positions[i]['res_min']
                
                # For each residue, calculate its plot position
                # residue N maps to position: plot_start + (N - res_min)
                plot_x = plot_start + (residues - res_min)
                
                # Plot the excess probability
                if prot_idx == 0 and i == 0:  # Only add legend once
                    ax.plot(plot_x, mean_ratio, color=c, linestyle=l, 
                           alpha=al, label=lab, zorder=100, linewidth=2)
                else:
                    ax.plot(plot_x, mean_ratio, color=c, linestyle=l, 
                           alpha=al, zorder=100, linewidth=2)
                
                ax.scatter(plot_x, mean_ratio, color=c, s=20, zorder=100, alpha=0.7)
                ax.fill_between(plot_x, mean_ratio - std_ratio, mean_ratio + std_ratio,
                               color=c, alpha=0.15, zorder=50, linewidth=0)
    
    # Set y-limit based on maximum value
    if all_max_values and not all(np.isnan(all_max_values)):
        max_val = np.nanmax(all_max_values)
        y_limit = min(max_val * 1.2, 600)
        # Round up to nearest value divisible by 20
        y_limit = int(np.ceil(y_limit / 20) * 20)
    else:
        y_limit = 600
    
    ax.set_ylim(0, y_limit)
    
    # Draw vertical dashed lines between IDR regions
    for i in range(max(aligned_positions.keys())):
        if i in aligned_positions and (i+1) in aligned_positions:
            boundary_x = aligned_positions[i]['plot_end'] + GAP_SIZE/2
            ax.axvline(x=boundary_x, color='grey', linewidth=2, 
                      linestyle='--', zorder=0)
    
    ax.set_xlim(-10, total_width + 10)
    ax.set_yticks([0, y_limit//2, y_limit])
    ax.tick_params(axis='y', labelsize=28, length=8, width=2.5)
    
    # Add horizontal grid lines at y-tick positions
    ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.5, alpha=0.8, zorder=1)
    
    # Draw vertical grid lines at ALL tick positions
    for grid_x in all_tick_positions:
        ax.axvline(x=grid_x, color='grey', linewidth=0.5, alpha=0.8, zorder=1)
    
    ax.set_axisbelow(True)
    
    # Set tick positions (but show labels only on bottom subplot)
    ax.set_xticks(all_tick_positions)
    if prot_idx == len(DISPLAY_ORDER) - 1:
        # Bottom subplot: show labels with thicker tick marks
        ax.set_xticklabels(all_tick_labels, fontsize=28)
        ax.tick_params(axis='x', length=8, width=2.5)
    else:
        # Other subplots: no labels, but keep thicker tick marks
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=8, width=2.5)
    
    # Add boundary line after deep species
    if prot_idx == boundary_idx:
        ax.axhline(y=-y_limit*0.15, color='#7F8C8D', linewidth=1.5,
                  linestyle='--', zorder=10)
    
    # Remove top and right spines
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

# Only add x-label to bottom plot
axes[-1].set_xlabel("Residue Number", fontsize=28)

# Add common y-label
fig.text(0.02, 0.5, "Occupancy Fold Change ($P_{g}$/$P_{g,EV}$)", 
         va='center', rotation='vertical', fontsize=28)

# Add legend
axes[0].legend(loc='upper right', fontsize=28, framealpha=0.9)

plt.tight_layout(pad=0.5)
plt.subplots_adjust(left=0.08, right=0.98, hspace=0.15)

plt.savefig('ALL_PROTEINS_excess_prob_aligned_tree_ordered_v2.png', dpi=300, bbox_inches="tight")
plt.savefig('ALL_PROTEINS_excess_prob_aligned_tree_ordered_v2.svg', bbox_inches="tight")
print("Saved ALL_PROTEINS_excess_prob_aligned_tree_ordered.png")
plt.close()