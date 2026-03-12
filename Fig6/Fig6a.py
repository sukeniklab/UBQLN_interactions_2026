import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from collections import defaultdict
from Bio import Phylo
from io import StringIO

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['font.size'] = 28
plt.rcParams['font.sans-serif'] = 'Arial'

# ═══════════════════════════════════════════════════════════════════════════
# SELECT WHICH STI1 DOMAIN TO ANALYZE
# ═══════════════════════════════════════════════════════════════════════════
STI1_DOMAIN = 1  # Change to 1 or 2 to analyze different STI1 domains

MOTIFS_OF_INTEREST = [
    "LIG_LIR_Nem_3",
    "DOC_WW_Pin1_4",
    "LIG_EH_1",
    "DOC_USP7_MATH_1",
    "MOD_GSK3_1", 
    "DOC_PP2A_B56_1"
]

# Custom colors for specific motifs (overrides functional category colors)
# Format: "motif_id": "hex_color"
MOTIF_COLORS = {
    "LIG_LIR_Nem_3": "darkorange",           
    "LIG_EH_1": "cyan",               
    "DOC_USP7_MATH_1": "mediumpurple",        
    "MOD_GSK3_1": "hotpink",              
    "DOC_PP2A_B56_1": "yellow",          
}


home_dir = "/home/jkniblo/IDR_folded"

# ── Protein definitions ───────────────────────────────────────────────────────
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

# Mapping between peak detection names and UniProt IDs
peak_to_uniprot = {
    "Dsk2_full": "P48510",
    "Q9UMX0": "Q9UMX0", "Q9UHD9": "Q9UHD9", "Q9NRR5": "Q9NRR5",
    "Q9SII8": "Q9SII8", "Q9SII9": "Q9SII9", "D4A3P1": "D4A3P1",
    "D4AA63": "D4AA63", "Q9JJP9": "Q9JJP9", "Q9VWD9": "Q9VWD9",
    "G5EFF7": "G5EFF7",
}

IDR_LABELS = {0: 'IDR1', 1: 'IDR2', 2: 'IDR3'}
DISPLAY_ORDER = []
DEEP_PIDS = {'P48510', 'Q9SII9', 'Q9SII8', 'G5EFF7', 'Q9VWD9'}
offset_dict = {p: [d[1] for d in domains] for p, domains in domain_dict.items()}

# ── Functional categories ─────────────────────────────────────────────────────
FUNC_CATEGORIES = {
    'MOD_GlcNH':  ('Glycosylation',         '#8E44AD'),
    'MOD_N-GLC':  ('Glycosylation',         '#8E44AD'),
    'MOD_CK':     ('Kinase — CK',           '#E74C3C'),
    'MOD_GSK':    ('Kinase — GSK3',         '#C0392B'),
    'MOD_Plk':    ('Kinase — Plk',          '#E67E22'),
    'MOD_PIKK':   ('Kinase — PIKK',         '#F39C12'),
    'MOD_ProD':   ('Kinase — ProD',         '#D35400'),
    'LIG_LIR':    ('Autophagy — LIR',       '#27AE60'),
    'LIG_EH':     ('Endocytic — EH',        '#16A085'),
    'DOC_USP7':   ('Deubiquitinase — USP7', '#2980B9'),
    'DOC_WW':     ('WW domain',             '#1ABC9C'),
    'DOC_CYCLIN': ('Cell cycle — Cyclin',   '#2C3E50'),
    'LIG_SH2':    ('Signalling — SH2',      "#3D96D1"),
    'LIG_BRCT':   ('DNA damage — BRCT',     "#6E8798"),
    'TRG_':       ('Trafficking',           '#F1C40F'),
    'CLV_':       ('Cleavage',              '#95A5A6'),
    'LIG_WD40':   ('WD40 — scaffold',       '#5D6D7E'),
    'LIG_Arc':    ('Arc complex',           '#A569BD'),
    'LIG_FHA':    ('FHA domain',            '#1F618D'),
    'DEG_SPOP':   ('Degradation — SPOP',    '#6C3483'),
}

def get_func_category(elm_id):
    # Check for custom color first (exact match)
    if elm_id in MOTIF_COLORS:
        # Extract category from functional categories
        for prefix, (cat, _) in FUNC_CATEGORIES.items():
            if elm_id.startswith(prefix):
                return cat, MOTIF_COLORS[elm_id]
        return 'Custom', MOTIF_COLORS[elm_id]
    
    # Fall back to functional category colors
    for prefix, (cat, col) in FUNC_CATEGORIES.items():
        if elm_id.startswith(prefix):
            return cat, col
    return 'Other', '#95A5A6'

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

# ── Phylogenetic tree ─────────────────────────────────────────────────────────
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
all_tips   = [c.name for c in tree.get_terminals()]
tree_depth = max(tree.distance(tree.root, t) for t in all_tips)

_pid_to_tree_pos = {}
for rank, tip_name in enumerate(all_tips):
    pid = TREE_NAME_TO_PID.get(tip_name)
    if pid and pid in prot_name_map and pid not in _pid_to_tree_pos:
        _pid_to_tree_pos[pid] = rank
DISPLAY_ORDER[:] = sorted(_pid_to_tree_pos.keys(), key=lambda p: _pid_to_tree_pos[p])

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

# ═══════════════════════════════════════════════════════════════════════════
# LOAD PEAK DETECTION RESULTS - FILTERED BY STI1 DOMAIN
# ═══════════════════════════════════════════════════════════════════════════
peak_dict = {}  # {uniprot_id: [(start, end, IDR, condition), ...]}
peak_summary_path = "peak_ranges/ALL_PROTEINS_peaks_summary.csv"

if os.path.exists(peak_summary_path):
    peak_df = pd.read_csv(peak_summary_path)
    print(f"Loaded {len(peak_df)} total peaks from {peak_summary_path}")
    
    # *** FILTER FOR SELECTED STI1 DOMAIN ***
    peak_df = peak_df[peak_df['STI1_domain'] == STI1_DOMAIN]
    print(f"Using {len(peak_df)} peaks for STI1-{STI1_DOMAIN}\n")
    
    for _, row in peak_df.iterrows():
        peak_protein = row['protein']
        uniprot_id = peak_to_uniprot.get(peak_protein)
        
        if uniprot_id:
            if uniprot_id not in peak_dict:
                peak_dict[uniprot_id] = []
            
            peak_dict[uniprot_id].append({
                'start': int(row['start']),
                'end': int(row['end']),
                'IDR': int(row['IDR']),
                'STI1_domain': int(row['STI1_domain']),
                'condition': row['condition'],
            })
    print(f"Peak regions loaded for {len(peak_dict)} proteins (STI1-{STI1_DOMAIN})\n")
else:
    print(f"Warning: Peak file not found at {peak_summary_path}\n")

filepath = f"{home_dir}/Analysis/excess_prob_seq_extract/Linker"
file_base_map = {}
for f in os.listdir(filepath):
    if "_Linker" not in f: continue
    base  = f.split("_Linker")[0]
    parts = base.split('|')
    if len(parts) < 2: continue
    pid = parts[1]
    if pid in prot_name_map and pid not in file_base_map:
        file_base_map[pid] = base

# ── Collect ALL SLIMs, flagging which fall in detected peak regions ───────────
protein_slims_by_idr = defaultdict(lambda: defaultdict(list))

for protein in prot_name_map:
    if protein not in file_base_map: continue
    peak_regions = peak_dict.get(protein, [])
    offsets      = offset_dict[protein]

    for linker_num, offset_idx in linker_file_mapping[protein].items():
        if offset_idx >= len(offsets): continue
        fname = f"{file_base_map[protein]}_Linker{linker_num}.csv"
        fpath = f"{filepath}/{fname}"
        if not os.path.exists(fpath): continue

        lstart = offsets[offset_idx]
        lend   = (domain_dict[protein][offset_idx + 1][0] - 1
                  if offset_idx + 1 < len(domain_dict[protein]) else lstart + 100)
        
        # Get detected peaks for this IDR (already filtered by STI1 domain)
        linker_peak_regions = [p for p in peak_regions 
                               if p['IDR'] == offset_idx + 1]

        with open(fpath) as fh:
            next(fh)
            for line in fh:
                parts   = line.strip().split(',')
                elm_id  = parts[0]
                s_start = int(parts[1]) + offsets[offset_idx]
                s_end   = int(parts[2]) + offsets[offset_idx]
                
                # Check if majority (>50%) of SLIM is within any detected peak
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
                
                # Special case: Yeast Dsk2 IDR2 appears in both IDR2 and IDR3 figures
                # if protein == "P48510" and offset_idx == 1:
                #     protein_slims_by_idr[2][protein].append(slim_data.copy())

def get_short_label(elm_id):
    parts = elm_id.split('_')
    return '_'.join(parts[:3]) if len(parts) >= 3 else elm_id

# ── Prepare conservation data for all IDRs ───────────────────────────────────
all_idr_shared_slims = {}
all_idr_slim_tier = {}
all_idr_slim_score = {}
all_idr_slim_proteins = {}

for seq_pos in sorted(protein_slims_by_idr.keys()):
    idr_data = protein_slims_by_idr[seq_pos]
    
    # Conservation analysis uses peak-SLIMs only
    slim_proteins_idr = defaultdict(set)
    for protein, slims in idr_data.items():
        for s in slims:
            if s['in_peak']:
                slim_proteins_idr[s['id']].add(protein)
    
    filtered_slims = {eid for eid, prots in slim_proteins_idr.items()
                      if len(prots) >= MIN_PROTEINS}
    slim_tier  = {}
    slim_score = {}
    for eid in filtered_slims:
        prots           = slim_proteins_idr[eid]
        slim_score[eid] = mrca_depth_score(prots)
        slim_tier[eid]  = get_conservation_tier(prots)
    
    shared_slims_idr = {eid for eid in filtered_slims
                        if slim_tier[eid] in TIERS_TO_SHOW}
    
    # Filter to specific motifs of interest if specified (EXACT MATCH ONLY)
    if MOTIFS_OF_INTEREST:
        shared_slims_idr = {eid for eid in shared_slims_idr if eid in MOTIFS_OF_INTEREST}
    
    all_idr_shared_slims[seq_pos] = shared_slims_idr
    all_idr_slim_tier[seq_pos] = slim_tier
    all_idr_slim_score[seq_pos] = slim_score
    all_idr_slim_proteins[seq_pos] = slim_proteins_idr
    
    print(f"\n{IDR_LABELS[seq_pos]} (STI1-{STI1_DOMAIN}): {len(shared_slims_idr)} conserved peak-SLiMs"
          f"{' (filtered to motifs of interest)' if MOTIFS_OF_INTEREST else ''}")

# ── Create single continuous plot per protein with ALIGNED IDRs ──────────────
boundary_idx = max(i for i, p in enumerate(DISPLAY_ORDER) if p in DEEP_PIDS)

# Calculate maximum length for each IDR position across all proteins
idr_max_lengths = {}
for seq_pos in sorted(protein_slims_by_idr.keys()):
    max_length = 0
    for protein in DISPLAY_ORDER:
        offs = offset_dict[protein]
        
        if seq_pos < len(offs):
            actual_idx = seq_pos
        # elif protein == "P48510" and seq_pos == 2:
        #     actual_idx = 1
        else:
            continue
        
        lstart = offs[actual_idx]
        lend = (domain_dict[protein][actual_idx + 1][0] - 1
               if actual_idx + 1 < len(domain_dict[protein]) else lstart + 100)
        length = lend - lstart
        max_length = max(max_length, length)
    
    idr_max_lengths[seq_pos] = max_length

# Set up aligned positions for each IDR
GAP_SIZE = 50  # spacing between IDRs
aligned_positions = {}
cumulative_pos = 0
for seq_pos in sorted(protein_slims_by_idr.keys()):
    aligned_positions[seq_pos] = {
        'start': cumulative_pos,
        'end': cumulative_pos + idr_max_lengths[seq_pos]
    }
    cumulative_pos += idr_max_lengths[seq_pos] + GAP_SIZE

# Total width for x-axis
total_width = cumulative_pos - GAP_SIZE

fig, axes = plt.subplots(len(DISPLAY_ORDER), 1,
                         figsize=(24, 2.5 * len(DISPLAY_ORDER)),
                         sharex=False)
if len(DISPLAY_ORDER) == 1: axes = [axes]
np.random.seed(42)

# Plot each protein with all IDRs aligned
for prot_idx, protein in enumerate(DISPLAY_ORDER):
    ax = axes[prot_idx]
    is_deep = protein in DEEP_PIDS
    peak_regions = peak_dict.get(protein, [])
    
    # Plot each IDR in its aligned position
    for seq_pos in sorted(protein_slims_by_idr.keys()):
        offs = offset_dict[protein]
        
        # Determine which offset_idx to use for this seq_pos
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
        
        # Center this IDR within its aligned slot
        slot_start = aligned_positions[seq_pos]['start']
        slot_width = idr_max_lengths[seq_pos]
        offset = (slot_width - length) / 2  # center within slot
        plot_start = slot_start + offset
        plot_end = plot_start + length
        
        idr_data = protein_slims_by_idr[seq_pos]
        shared_slims_idr = all_idr_shared_slims[seq_pos]
        slim_tier = all_idr_slim_tier[seq_pos]
        
        # Draw IDR baseline
        ax.plot([plot_start, plot_end], [0, 0], 
               color='lightgrey', linewidth=20, solid_capstyle='butt', zorder=1)
        
        # Add IDR label above the region (only on first row)
        if prot_idx == 0:
            mid_x = (slot_start + aligned_positions[seq_pos]['end']) / 2
            ax.text(mid_x, 0.28, IDR_LABELS[seq_pos], 
                   fontsize=28, fontweight='bold', ha='center', va='bottom',
                   color='#34495E')
        
        # Draw detected peak regions
        lpk = [p for p in peak_regions if p['IDR'] == peak_idr_num]
        for peak in lpk:
            # Convert actual residue position to plot position
            offset_in_idr = peak['start'] - lstart
            peak_plot_start = plot_start + offset_in_idr
            plot_width = peak['end'] - peak['start']
            
            ax.add_patch(patches.Rectangle(
                (peak_plot_start, -0.225), plot_width, 0.45,
                linewidth=0, facecolor='#FF8C42', alpha=0.5, zorder=2))
        
        # Plot SLIMs with deduplication
        if protein in idr_data:
            # Sort SLIMs by position for deduplication
            sorted_slims = sorted(idr_data[protein], key=lambda s: s['mid'])
            
            # Track last plotted motif to avoid duplicates
            last_plotted = {}  # {motif_id: position}
            MIN_DISTANCE = 15  # Minimum residue distance to plot same motif again
            
            for slim in sorted_slims:
                eid = slim['id']
                in_peak = slim['in_peak']
                conserved = (eid in shared_slims_idr) and in_peak
                
                # Skip if same motif was just plotted nearby
                if eid in last_plotted:
                    distance = abs(slim['mid'] - last_plotted[eid])
                    if distance < MIN_DISTANCE:
                        continue  # Skip this duplicate
                
                if conserved:
                    tier = slim_tier[eid]
                    cat, base_col = get_func_category(eid)
                    col = base_col
                    size = 400; ew = 2.25; alpha = 1.0; zord = 1000
                    ec = 'black'
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
                
                # Update last plotted position for this motif
                last_plotted[eid] = slim['mid']
    
    # Draw vertical dashed lines to indicate boundaries between IDR slots
    for seq_pos in sorted(protein_slims_by_idr.keys())[:-1]:
        boundary_x = aligned_positions[seq_pos]['end'] + GAP_SIZE/2
        ax.axvline(x=boundary_x, color='#BDC3C7', linewidth=2, 
                  linestyle='--', alpha=0.6, zorder=0)
    
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlim(-10, total_width + 10)
    ax.set_yticks([])
    
    for sp in ['top', 'right', 'left']: 
        ax.spines[sp].set_visible(False)
    
    # Add x-ticks showing actual residue positions for this specific protein
    tick_positions = []
    tick_labels = []
    
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
        
        # Get plot positions for this IDR
        slot_start = aligned_positions[seq_pos]['start']
        slot_width = idr_max_lengths[seq_pos]
        offset = (slot_width - length) / 2
        plot_start = slot_start + offset
        plot_end = plot_start + length
        
        # Add ticks at regular residue intervals (every 30 residues)
        tick_interval = 30
        start_res = (lstart // tick_interval) * tick_interval
        if start_res < lstart:
            start_res += tick_interval
        
        # Calculate how many ticks we would get with this interval
        res = start_res
        idr_ticks = []
        while res <= lend:
            offset_in_idr = res - lstart
            plot_x = plot_start + offset_in_idr
            idr_ticks.append((plot_x, res))
            res += tick_interval
        
        # Ensure at least 2 ticks per IDR region
        if len(idr_ticks) < 2:
            # If less than 2 ticks, place one at start and one at end
            idr_ticks = [
                (plot_start, lstart),
                (plot_end, lend)
            ]
        
        # Add all ticks for this IDR to the main lists
        for plot_x, residue in idr_ticks:
            tick_positions.append(plot_x)
            tick_labels.append(str(residue))
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=28)
    ax.tick_params(axis='x', length=4, width=1)
    
    ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Calculate total stats across all IDRs
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

# ── Legend ────────────────────────────────────────────────────────────────
legend_handles = []

# Collect all unique conserved SLIMs across all IDRs
all_conserved_slims = {}
for seq_pos in sorted(protein_slims_by_idr.keys()):
    for eid in all_idr_shared_slims[seq_pos]:
        if eid not in all_conserved_slims:
            all_conserved_slims[eid] = {
                'proteins': all_idr_slim_proteins[seq_pos][eid],
                'tier': all_idr_slim_tier[seq_pos][eid],
                'idrs': [seq_pos]
            }
        else:
            all_conserved_slims[eid]['idrs'].append(seq_pos)

# Add conserved SLIMs to legend
for eid in sorted(all_conserved_slims.keys(), key=lambda e: -len(all_conserved_slims[e]['proteins'])):
    cat, base_col = get_func_category(eid)
    col = base_col
    idr_list = ','.join([IDR_LABELS[i] for i in all_conserved_slims[eid]['idrs']])
    legend_handles.append(
        Line2D([0],[0], marker=get_slim_shape(eid), color='w',
               markerfacecolor=col, markersize=20,
               markeredgecolor='black', markeredgewidth=1.2,
               label=f"{get_short_label(eid)}"))

# Add separator and guide
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

plt.savefig(f"slim_peak_conservation_STI{STI1_DOMAIN}_ALL_IDRs_aligned_dedup.png",
            dpi=300, bbox_inches='tight')
plt.savefig(f"slim_peak_conservation_STI{STI1_DOMAIN}_ALL_IDRs_aligned_dedup.svg",
            bbox_inches='tight')
print(f"\nSaved slim_peak_conservation_STI{STI1_DOMAIN}_ALL_IDRs_aligned.png")
plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────
for seq_pos in sorted(protein_slims_by_idr.keys()):
    idr_data = protein_slims_by_idr[seq_pos]
    slim_proteins_idr = defaultdict(set)
    for protein, slims in idr_data.items():
        for s in slims:
            if s['in_peak']:
                slim_proteins_idr[s['id']].add(protein)
    filtered     = {eid: prots for eid, prots in slim_proteins_idr.items()
                    if len(prots) >= MIN_PROTEINS}
    slim_tier_s  = {eid: get_conservation_tier(prots) for eid, prots in filtered.items()}
    slim_score_s = {eid: mrca_depth_score(prots)      for eid, prots in filtered.items()}
    shared       = {eid: prots for eid, prots in filtered.items()
                    if slim_tier_s[eid] in TIERS_TO_SHOW}
    
    # Filter to specific motifs of interest if specified (EXACT MATCH ONLY)
    if MOTIFS_OF_INTEREST:
        shared = {eid: prots for eid, prots in shared.items() if eid in MOTIFS_OF_INTEREST}
    
    tier_order   = ['ancient', 'deep', 'pan_vertebrate', 'paralog_specific']

    print(f"\n{'='*85}")
    print(f"{IDR_LABELS[seq_pos]} — STI1-{STI1_DOMAIN} conserved peak-SLiMs (≥{MIN_PROTEINS} proteins)"
          f"{' [FILTERED]' if MOTIFS_OF_INTEREST else ''}")
    print(f"{'='*85}")
    print(f"{'SLIM':<35} {'function':<25} {'tier':<18} {'score':>6} {'n':>4}  proteins")
    print("-" * 85)
    for eid, prots in sorted(shared.items(),
            key=lambda x: (tier_order.index(slim_tier_s[x[0]])
                           if slim_tier_s[x[0]] in tier_order else 99,
                           -slim_score_s[x[0]], -len(x[1]))):
        cat, _ = get_func_category(eid)
        pnames = ', '.join(prot_name_map[p] for p in prots)
        print(f"{eid:<35} {cat:<25} {slim_tier_s[eid]:<18} "
              f"{slim_score_s[eid]:>6.3f} {len(prots):>4}  {pnames}")