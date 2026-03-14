import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from collections import defaultdict
from Bio import Phylo

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['font.size'] = 28
plt.rcParams['font.sans-serif'] = 'Arial'


STI1_DOMAIN = 1  # which STI1 domain to focus on 

MOTIFS_OF_INTEREST = {
    "LIG_LIR_Nem_3": "#ff6600ff",           
    "LIG_EH_1": "#00ffffff",               
    "DOC_USP7_MATH_1": "#aa87deff",        
    "DOC_WW_Pin1_4": "#ffff00ff",        
}

domain_dict = {
    "P48510": [[1,75],[147,223],[327,373]],
    "Q9UMX0": [[37,107],[182,251],[387,470],[542,585]],
    "Q9UHD9": [[33,103],[178,247],[379,462],[577,620]],
    "Q9NRR5": [[13,83],[192,261],[393,476],[554,597]],
    "Q9SII8": [[19,93],[168,236],[381,449],[504,548]],
    "Q9SII9": [[18,93],[163,231],[364,444],[491,535]],
    "D4A3P1": [[10,82],[186,255],[387,456],[549,592]],
    "D4AA63": [[30,103],[190,263],[393,469],[592,610]],
    "Q9JJP9": [[28,102],[173,245],[381,457],[539,579]],
    "Q9VWD9": [[9,79],[135,207],[322,401],[499,541]],
    "G5EFF7": [[8,83],[132,200],[311,381],[455,501]],
}

prot_name_map = {
    "P48510":"DSK2_YEAST",    "Q9UMX0":"UBQL1_HUMAN", "Q9UHD9":"UBQL2_HUMAN",
    "Q9NRR5":"UBQL4_HUMAN",   "Q9SII8":"Dsk2B_PLANT", "Q9SII9":"Dsk2A_PLANT",
    "D4A3P1":"UBQL4_RAT",     "D4AA63":"UBQL2_RAT",   "Q9JJP9":"UBQL1_RAT",
    "Q9VWD9":"UBQN_Fly",      "G5EFF7":"UBQL_CElegans",
}

linker_file_mapping = {
    "P48510": {1:0, 3:1},
    "Q9UMX0": {1:0, 2:1, 3:2}, "Q9UHD9": {1:0, 2:1, 3:2},
    "Q9NRR5": {1:0, 2:1, 3:2}, "Q9SII8": {1:0, 2:1, 3:2},
    "Q9SII9": {1:0, 2:1, 3:2}, "D4A3P1": {1:0, 2:1, 3:2},
    "D4AA63": {1:0, 2:1, 3:2}, "Q9JJP9": {1:0, 2:1, 3:2},
    "Q9VWD9": {1:0, 2:1, 3:2}, "G5EFF7": {1:0, 2:1, 3:2},
}

IDR_LABELS = {0: 'IDR1', 1: 'IDR2', 2: 'IDR3'}
DEEP_PIDS = {'P48510', 'Q9SII9', 'Q9SII8', 'G5EFF7', 'Q9VWD9'}
offset_dict = {p: [d[1] for d in domains] for p, domains in domain_dict.items()}

# Define shapes for different SLiM types based on their ID prefixes
SLIM_TYPE_SHAPES = {
    'MOD_': 's',   'LIG_': 'o',   'DOC_': '^',
    'DEG_': 'v',   'TRG_': 'D',   'CLV_': 'p',
}
DEFAULT_SHAPE = '*'

def get_slim_shape(elm_id):
    for prefix, shape in SLIM_TYPE_SHAPES.items():
        if elm_id.startswith(prefix):
            return shape
    return DEFAULT_SHAPE


# Load and process the phylogenetic tree for conservation scoring and display ordering
DISPLAY_ORDER = []

TREE_NAME_TO_PID = {
    "C. Elegans UBQN": "G5EFF7",  "Human UBQLN2": "Q9UHD9",
    "Rat UBQLN2":      "D4AA63",  "Human UBQLN1": "Q9UMX0",
    "Rat UBQLN1":      "Q9JJP9",  "Human UBQLN4": "Q9NRR5",
    "Rat UBQLN4":      "D4A3P1",  "Fly UBQN":     "Q9VWD9",
    "Plant Dsk2A":     "Q9SII9",  "Plant Dsk2B":  "Q9SII8",
    "Yeast Dsk2":      "P48510",
    "Mouse UBQL2": None, "Mouse UBQL1": None, "Mouse UBQL4": None,
    "Frog UBQLN4": None, "Zebra Fish UBQN": None,
}

tree = Phylo.read('phylo_tree.nwk', 'newick')
tree.root_with_outgroup({'name': 'Yeast Dsk2'})
all_tips   = [c.name for c in tree.get_terminals()]
tree_depth = max(tree.distance(tree.root, t) for t in all_tips)

_pid_to_tree_pos = {}
for rank, tip_name in enumerate(all_tips):
    pid = TREE_NAME_TO_PID.get(tip_name)
    if pid and pid in prot_name_map and pid not in _pid_to_tree_pos:
        _pid_to_tree_pos[pid] = rank
DISPLAY_ORDER[:] = sorted(_pid_to_tree_pos.keys(), key=lambda p: _pid_to_tree_pos[p])


def get_motif_color(elm_id):
    """
    Returns the color for a given motif ID based on the MOTIFS_OF_INTEREST mapping.
    If the motif ID is not in the mapping, returns a default grey color.
    """
    return MOTIFS_OF_INTEREST.get(elm_id, '#95A5A6')

def mrca_depth_score(protein_set):
    leaf_names = [n for n, pid in TREE_NAME_TO_PID.items()
                  if pid in protein_set and n in all_tips]
    if len(leaf_names) < 2: return 0.0
    mrca = tree.common_ancestor(*leaf_names)
    return 1.0 - (tree.distance(tree.root, mrca) / tree_depth)

def get_conservation_tier(protein_set):
    score = mrca_depth_score(protein_set)
    if   score >= 0.85: return 'ancient'
    elif score >= 0.55: return 'deep'
    elif score >= 0.25: return 'pan_vertebrate'
    else:               return 'paralog_specific'

MIN_PROTEINS  = 3
TIERS_TO_SHOW = {'ancient', 'deep', 'pan_vertebrate'}

# Load peak region data for all proteins and filter for the specified STI1 domain
peak_dict = {}
peak_summary_path = "peak_ranges/ALL_PROTEINS_peaks_summary.csv"

if os.path.exists(peak_summary_path):
    peak_df = pd.read_csv(peak_summary_path)
    peak_df = peak_df[peak_df['STI1_domain'] == STI1_DOMAIN]
    print(f"Using {len(peak_df)} peaks for STI1-{STI1_DOMAIN}\n")
    
    for _, row in peak_df.iterrows():
        peak_protein = row['protein']
        uniprot_id = peak_protein if peak_protein != "Dsk2_full" else "P48510"
        
        if uniprot_id in prot_name_map:
            if uniprot_id not in peak_dict:
                peak_dict[uniprot_id] = []
            
            peak_dict[uniprot_id].append({
                'start': int(row['start']),
                'end': int(row['end']),
                'IDR': int(row['IDR']),
                'STI1_domain': int(row['STI1_domain']),
                'condition': row['condition'],
            })

# Map protein IDs to their motif file base names
file_base_map = {}
for f in os.listdir("IDR_motifs"):
    if "_Linker" not in f: continue
    base  = f.split("_Linker")[0]
    parts = base.split('|')
    if len(parts) < 2: continue
    pid = parts[1]
    if pid in prot_name_map and pid not in file_base_map:
        file_base_map[pid] = base

protein_slims_by_idr = defaultdict(lambda: defaultdict(list))

# Load and organize all SLiM motifs by protein and IDR region
for protein in prot_name_map:
    if protein not in file_base_map: continue
    peak_regions = peak_dict.get(protein, [])
    offsets      = offset_dict[protein]

    for linker_num, offset_idx in linker_file_mapping[protein].items():
        if offset_idx >= len(offsets): continue
        fname = f"{file_base_map[protein]}_Linker{linker_num}.csv"
        fpath = f"IDR_motifs/{fname}"
        if not os.path.exists(fpath): continue

        lstart = offsets[offset_idx]
        lend = (domain_dict[protein][offset_idx + 1][0] - 1
                if offset_idx + 1 < len(domain_dict[protein]) else lstart + 100)
        
        linker_peak_regions = [p for p in peak_regions if p['IDR'] == offset_idx + 1]

        with open(fpath) as fh:
            next(fh)
            
            for line in fh:
                parts = line.strip().split(',')
                elm_id = parts[0]
                s_start = int(parts[1]) + offsets[offset_idx]
                s_end = int(parts[2]) + offsets[offset_idx]
                slim_length = s_end - s_start + 1
                in_peak = False
                
                for p in linker_peak_regions:
                    overlap_start = max(s_start, p['start'])
                    overlap_end = min(s_end, p['end'])
                    overlap_length = max(0, overlap_end - overlap_start + 1)
                    if overlap_length > slim_length / 2:
                        in_peak = True
                        break
                
                slim_data = {
                    'id':      elm_id,
                    'start':   s_start,
                    'end':     s_end,
                    'mid':     (s_start + s_end) / 2,
                    'in_peak': in_peak,
                }
                protein_slims_by_idr[offset_idx][protein].append(slim_data)
all_idr_shared_slims = {}
all_idr_slim_proteins = {}

# Identify conserved SLiMs that appear in multiple proteins and meet conservation criteria
for seq_pos in sorted(protein_slims_by_idr.keys()):
    idr_data = protein_slims_by_idr[seq_pos]
    
    slim_to_proteins = defaultdict(set)
    for protein, slims in idr_data.items():
        for slim in slims:
            if slim['in_peak']:
                slim_id = slim['id']
                slim_to_proteins[slim_id].add(protein)
    
    # Filter to SLiMs
    slims_in_multiple_proteins = {
        slim_id for slim_id, proteins in slim_to_proteins.items()
        if len(proteins) >= MIN_PROTEINS
    }
    
    conserved_slims = {
        slim_id for slim_id in slims_in_multiple_proteins
        if get_conservation_tier(slim_to_proteins[slim_id]) in TIERS_TO_SHOW
    }
    
    # Keep only SLiMs that are in our motifs of interest
    final_conserved_slims = {
        slim_id for slim_id in conserved_slims 
        if slim_id in MOTIFS_OF_INTEREST
    }
    
    all_idr_shared_slims[seq_pos] = final_conserved_slims
    all_idr_slim_proteins[seq_pos] = slim_to_proteins

# Calculate the maximum IDR length across all proteins for each sequential position, which will be used for consistent plotting
idr_max_lengths = {}
for seq_pos in sorted(protein_slims_by_idr.keys()):
    max_length = 0

    for protein in DISPLAY_ORDER:
        offs = offset_dict[protein]
        
        if seq_pos < len(offs):
            actual_idx = seq_pos
       
        else:
            continue
        
        lstart = offs[actual_idx]
        lend = (domain_dict[protein][actual_idx + 1][0] - 1
               if actual_idx + 1 < len(domain_dict[protein]) else lstart + 100)
        length = lend - lstart
        max_length = max(max_length, length)
    
    idr_max_lengths[seq_pos] = max_length


# Calculate aligned plotting positions for each IDR with gaps between them
GAP_SIZE = 50
aligned_positions = {}
cumulative_pos = 0
for seq_pos in sorted(protein_slims_by_idr.keys()):
    aligned_positions[seq_pos] = {
        'start': cumulative_pos,
        'end': cumulative_pos + idr_max_lengths[seq_pos]
    }
    cumulative_pos += idr_max_lengths[seq_pos] + GAP_SIZE

total_width = cumulative_pos - GAP_SIZE

fig, axes = plt.subplots(len(DISPLAY_ORDER), 1,
                         figsize=(24, 2.5 * len(DISPLAY_ORDER)),
                         sharex=False)
if len(DISPLAY_ORDER) == 1: axes = [axes]
np.random.seed(42)

# Plot each protein on its own subplot in phylogenetic order
for prot_idx, protein in enumerate(DISPLAY_ORDER):
    ax = axes[prot_idx]
    is_deep = protein in DEEP_PIDS
    peak_regions = peak_dict.get(protein, [])
    
    # Plot each IDR region for this protein
    for seq_pos in sorted(protein_slims_by_idr.keys()):
        offs = offset_dict[protein]
        
        if seq_pos < len(offs):
            actual_idx = seq_pos
            peak_idr_num = seq_pos + 1
        elif protein == "P48510" and seq_pos == 2:
            actual_idx = 1
            peak_idr_num = 3
        else:
            continue
        
        lstart = offs[actual_idx]
        lend = (domain_dict[protein][actual_idx + 1][0] - 1
               if actual_idx + 1 < len(domain_dict[protein]) else lstart + 100)
        length = lend - lstart
        
        slot_start = aligned_positions[seq_pos]['start']
        slot_width = idr_max_lengths[seq_pos]
        offset = (slot_width - length) / 2
        plot_start = slot_start + offset
        plot_end = plot_start + length
        
        idr_data = protein_slims_by_idr[seq_pos]
        shared_slims_idr = all_idr_shared_slims[seq_pos]
        
        ax.plot([plot_start, plot_end], [0, 0], 
               color='lightgrey', linewidth=20, solid_capstyle='butt', zorder=1)
        
        if prot_idx == 0:
            mid_x = (slot_start + aligned_positions[seq_pos]['end']) / 2
            ax.text(mid_x, 0.28, IDR_LABELS[seq_pos], 
                   fontsize=28, fontweight='bold', ha='center', va='bottom',
                   color='#34495E')
        
        lpk = [p for p in peak_regions if p['IDR'] == peak_idr_num]
        
        for peak in lpk:
            offset_in_idr = peak['start'] - lstart
            peak_plot_start = plot_start + offset_in_idr
            plot_width = peak['end'] - peak['start']
            
            ax.add_patch(patches.Rectangle(
                (peak_plot_start, -0.225), plot_width, 0.45,
                linewidth=0, facecolor='#FF8C42', alpha=0.5, zorder=2))
        
        if protein in idr_data:
            sorted_slims = sorted(idr_data[protein], key=lambda s: s['mid'])
            last_plotted = {}
            MIN_DISTANCE = 15
            
            # Plot individual SLiM motifs with appropriate styling
            for slim in sorted_slims:
                eid = slim['id']
                in_peak = slim['in_peak']
                conserved = (eid in shared_slims_idr) and in_peak
                
                if eid in last_plotted:
                    if abs(slim['mid'] - last_plotted[eid]) < MIN_DISTANCE:
                        continue
                
                if conserved:
                    col = get_motif_color(eid)
                    size, ew, alpha, zord, ec = 400, 2.25, 1.0, 1000, 'black'
                else:
                    col = 'k'; size = 160; ew = 0
                    alpha = 0.5; zord = 2; ec = 'none'
                
                # Convert actual position to plot position
                offset_in_idr = slim['mid'] - lstart
                plot_x = plot_start + offset_in_idr
                
                y = np.random.uniform(-0.12, 0.12)
                ax.scatter(plot_x, y,
                          color=col, s=size,
                          marker=get_slim_shape(eid),
                          edgecolors=ec, linewidths=ew,
                          alpha=alpha, zorder=zord)
                
                last_plotted[eid] = slim['mid']

    for seq_pos in sorted(protein_slims_by_idr.keys())[:-1]:
        boundary_x = aligned_positions[seq_pos]['end'] + GAP_SIZE/2
        ax.axvline(x=boundary_x, color='#BDC3C7', linewidth=2, 
                  linestyle='--', alpha=0.6, zorder=0)
    
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlim(-10, total_width + 10)
    ax.set_yticks([])
    
    for sp in ['top', 'right', 'left']: 
        ax.spines[sp].set_visible(False)
    
    tick_positions = []
    tick_labels = []



    boundary_idx = max(i for i, p in enumerate(DISPLAY_ORDER) if p in DEEP_PIDS)
    
    for seq_pos in sorted(protein_slims_by_idr.keys()):
        offs = offset_dict[protein]
        
        # Determine which offset_idx to use for this seq_pos
        if seq_pos < len(offs):
            actual_idx = seq_pos
        elif protein == "P48510" and seq_pos == 2:
            actual_idx = 1
        else:
            continue
        
        lstart = offs[actual_idx]
        lend = (domain_dict[protein][actual_idx + 1][0] - 1
               if actual_idx + 1 < len(domain_dict[protein]) else lstart + 100)
        length = lend - lstart
        
        slot_start = aligned_positions[seq_pos]['start']
        slot_width = idr_max_lengths[seq_pos]
        offset = (slot_width - length) / 2
        plot_start = slot_start + offset
        plot_end = plot_start + length
        
        tick_interval = 30
        start_res = (lstart // tick_interval) * tick_interval
        if start_res < lstart:
            start_res += tick_interval
        
        res = start_res
        idr_ticks = []
        while res <= lend:
            offset_in_idr = res - lstart
            plot_x = plot_start + offset_in_idr
            idr_ticks.append((plot_x, res))
            res += tick_interval
        
        if len(idr_ticks) < 2:
            idr_ticks = [(plot_start, lstart), (plot_end, lend)]
        
        for plot_x, residue in idr_ticks:
            tick_positions.append(plot_x)
            tick_labels.append(str(residue))
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=28)
    ax.tick_params(axis='x', length=4, width=1)
    
    ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
    
    n_total = sum(len(idr_data.get(protein, [])) 
                  for idr_data in protein_slims_by_idr.values())
    n_peak = sum(sum(1 for s in idr_data.get(protein, []) if s['in_peak'])
                 for idr_data in protein_slims_by_idr.values())
    n_shar = sum(sum(1 for s in idr_data.get(protein, []) 
                     if s['id'] in all_idr_shared_slims[seq_pos])
                 for seq_pos, idr_data in protein_slims_by_idr.items())
    
    ax.text(-0.02, 0.5,
           f"{prot_name_map[protein]}\n"
           f"all: {n_total} | peak: {n_peak} | conserved: {n_shar}",
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           va='center', ha='right',
           color='#2C3E50' if is_deep else 'black')
    
    if prot_idx == boundary_idx:
        ax.axhline(y=-0.48, color='#7F8C8D', linewidth=1.5,
                  linestyle='--', zorder=10)
    
    if prot_idx == len(DISPLAY_ORDER) - 1:
        ax.set_xlabel("Residue", fontsize=28)

legend_handles = []
all_conserved_slims = {}
# Aggregate all conserved SLiMs across IDR regions for the legend
for seq_pos in sorted(protein_slims_by_idr.keys()):
    for eid in all_idr_shared_slims[seq_pos]:
        if eid not in all_conserved_slims:
            all_conserved_slims[eid] = all_idr_slim_proteins[seq_pos][eid]
        else:
            all_conserved_slims[eid].update(all_idr_slim_proteins[seq_pos][eid])

# Create legend entries for each conserved SLiM, sorted by prevalence
for eid in sorted(all_conserved_slims.keys(), key=lambda e: -len(all_conserved_slims[e])):
    legend_handles.append(
        Line2D([0],[0], marker=get_slim_shape(eid), color='w',
               markerfacecolor=get_motif_color(eid), markersize=20,
               markeredgecolor='black', markeredgewidth=1.2,
               label='_'.join(eid.split('_')[:3])))

legend_handles += [
    Line2D([0],[0], marker='s', color='w', markerfacecolor='grey',
           markersize=20, markeredgecolor='black', markeredgewidth=0.8,
           label='Modification'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='grey',
           markersize=20, markeredgecolor='black', markeredgewidth=0.8,
           label='Ligand binding'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='grey',
           markersize=20, markeredgecolor='black', markeredgewidth=0.8,
           label='Targeting'),
    Line2D([0],[0], marker='p', color='w', markerfacecolor='grey',
           markersize=20, markeredgecolor='black', markeredgewidth=0.8,
           label='Cleavage)'),
]

fig.legend(handles=legend_handles, fontsize=11,
           title_fontsize=13,
           loc='center left', bbox_to_anchor=(1.01, 0.5),
           frameon=True, ncol=1)

plt.tight_layout(pad=0.5)
plt.subplots_adjust(right=0.78, hspace=0.25)
plt.savefig("Fig6a.svg", bbox_inches='tight')
plt.close()
