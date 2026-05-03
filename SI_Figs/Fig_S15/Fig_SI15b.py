import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from Bio import Phylo, SeqIO

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['font.size']      = 28
plt.rcParams['font.sans-serif'] = 'Arial'

def make_boxplot(ax, hs_vals, nhs_vals, title):
    bp = ax.boxplot(
        [hs_vals, nhs_vals],
        positions=[0, 1],
        widths=BOX_WIDTH,
        patch_artist=True,
        notch=False,
        showfliers=False,
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(linewidth=2.0),
        capprops=dict(linewidth=2.0),
        boxprops=dict(linewidth=1.5),
    )
    for patch, color in zip(bp['boxes'], [HOTSPOT_COLOR, NONHOTSPOT_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Overlay scatter points
    np.random.seed(42)
    for pos, vals, color in [(0, hs_vals, HOTSPOT_COLOR), (1, nhs_vals, NONHOTSPOT_COLOR)]:
        if len(vals) > 0:
            x_jitter = np.random.normal(0, 0.07, size=len(vals))
            ax.scatter(
                [pos] * len(vals) + x_jitter,
                vals,
                alpha=0.4,
                s=50,
                color='k',
                edgecolors='k',
                zorder=3
            )

    ax.set_ylim(YLIM)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels([str(t) for t in YTICKS], fontsize=20)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Hotspot\n(n={len(hs_vals)})', f'Non-hotspot\n(n={len(nhs_vals)})'], fontsize=20)
    ax.set_ylabel('Hydrophobicity', fontsize=22)
    ax.set_title(title, fontsize=24, pad=14)
    ax.set_xlim(-0.6, 1.6)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    if len(hs_vals) >= 2 and len(nhs_vals) >= 2:
        t_stat, p_val = stats.ttest_ind(hs_vals, nhs_vals, equal_var=False)
        p_str = f'p = {p_val:.4f}'
        ax.text(0.25, 0.9, f"Welch's {p_str}",
                transform=ax.transAxes, fontsize=20, color='black')

def calculate_hydrophobicity(sequence):
    return [EISENBERG_SCALE.get(aa.upper(), 0.0) for aa in sequence]



EISENBERG_SCALE = {
    'A': 0.620, 'R': -2.530, 'N': -0.780, 'D': -0.900, 'C': 0.290,
    'Q': -0.850, 'E': -0.740, 'G': 0.480,  'H': -0.400, 'I': 1.380,
    'L': 1.060, 'K': -1.500, 'M': 0.640,  'F': 1.190,  'P': 0.120,
    'S': -0.180, 'T': -0.050, 'W': 0.810,  'Y': 0.260,  'V': 1.080
}
YLIM = (-1, 1)
YTICKS = [-1, 0, 1]


STI1_DOMAIN = 1
MIN_SEG_LEN = 10

HOTSPOT_COLOR    = '#FF8C42'
NONHOTSPOT_COLOR = 'lightgrey'
IDR_LABELS       = {0: 'IDR1', 1: 'IDR2', 2: 'IDR3'}

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
    "P48510": {1:0, 2:1},
    "Q9UMX0": {1:0, 2:1, 3:2}, "Q9UHD9": {1:0, 2:1, 3:2},
    "Q9NRR5": {1:0, 2:1, 3:2}, "Q9SII8": {1:0, 2:1, 3:2},
    "Q9SII9": {1:0, 2:1, 3:2}, "D4A3P1": {1:0, 2:1, 3:2},
    "D4AA63": {1:0, 2:1, 3:2}, "Q9JJP9": {1:0, 2:1, 3:2},
    "Q9VWD9": {1:0, 2:1, 3:2}, "G5EFF7": {1:0, 2:1, 3:2},
}
peak_to_uniprot = {
    "Dsk2_full":"P48510", "Q9UMX0":"Q9UMX0", "Q9UHD9":"Q9UHD9",
    "Q9NRR5":"Q9NRR5",   "Q9SII8":"Q9SII8", "Q9SII9":"Q9SII9",
    "D4A3P1":"D4A3P1",   "D4AA63":"D4AA63", "Q9JJP9":"Q9JJP9",
    "Q9VWD9":"Q9VWD9",   "G5EFF7":"G5EFF7",
}
offset_dict = {p: [d[1] for d in domains] for p, domains in domain_dict.items()}

TREE_NAME_TO_PID = {
    "C. Elegans UBQN":"G5EFF7", "Human UBQLN2":"Q9UHD9",
    "Rat UBQLN2":"D4AA63",      "Human UBQLN1":"Q9UMX0",
    "Rat UBQLN1":"Q9JJP9",      "Human UBQLN4":"Q9NRR5",
    "Rat UBQLN4":"D4A3P1",      "Fly UBQN":"Q9VWD9",
    "Plant Dsk2A":"Q9SII9",     "Plant Dsk2B":"Q9SII8",
    "Yeast Dsk2":"P48510",
    "Mouse UBQL2":None, "Mouse UBQL1":None, "Mouse UBQL4":None,
    "Frog UBQLN4":None, "Zebra Fish UBQN":None,
}

tree = Phylo.read('phylo_tree.nwk', 'newick')
tree.root_with_outgroup({'name': 'Yeast Dsk2'})
DISPLAY_ORDER = []
_pid_to_tree_pos = {}
for rank, tip in enumerate(tree.get_terminals()):
    pid = TREE_NAME_TO_PID.get(tip.name)
    if pid and pid in prot_name_map and pid not in _pid_to_tree_pos:
        _pid_to_tree_pos[pid] = rank
DISPLAY_ORDER[:] = sorted(_pid_to_tree_pos, key=lambda p: _pid_to_tree_pos[p])
#load sequences & compute hydrophobicity 
linker_sequences = {}
linkers_all_path = "Linkers_all.fasta"

if os.path.exists(linkers_all_path):
    for record in SeqIO.parse(linkers_all_path, "fasta"):
        parts = record.id.split('|')
        if len(parts) >= 3:
            uid, name_part = parts[1], parts[2]
            if '_Linker' in name_part and uid in prot_name_map:
                try:
                    lnum = int(name_part.split('_Linker')[-1])
                    linker_sequences[(uid, lnum)] = str(record.seq)
                except ValueError:
                    pass

# Compute hydrophobicity for all sequences
linker_hydrophobicity = {
    k: calculate_hydrophobicity(seq)
    for k, seq in linker_sequences.items()
}

#Get hotspots 
peak_dict = {}
peak_summary_path = "ALL_PROTEINS_peaks_summary.csv"

if os.path.exists(peak_summary_path):
    peak_df = pd.read_csv(peak_summary_path)
    peak_df = peak_df[peak_df['STI1_domain'] == STI1_DOMAIN]
    for _, row in peak_df.iterrows():
        uid = peak_to_uniprot.get(row['protein'])
        if uid:
            peak_dict.setdefault(uid, []).append({
                'start': int(row['start']), 'end': int(row['end']),
                'IDR':   int(row['IDR']),
            })


protein_idr_info = {}
for protein in prot_name_map:
    protein_idr_info[protein] = {}
    offs = offset_dict[protein]
    for linker_num, offset_idx in linker_file_mapping.get(protein, {}).items():
        if offset_idx >= len(offs):
            continue
        lstart = offs[offset_idx]
        lend   = (domain_dict[protein][offset_idx+1][0] - 1
                  if offset_idx+1 < len(domain_dict[protein]) else lstart+100)
        protein_idr_info[protein][offset_idx] = {
            'lstart': lstart, 'lend': lend, 'linker_num': linker_num
        }
        if protein == "P48510" and offset_idx == 1:
            protein_idr_info[protein][2] = {
                'lstart': lstart, 'lend': lend, 'linker_num': linker_num
            }

BOX_WIDTH = 0.5

#mean of non-hotspot hydrophobicity 
idr_data = {sp: {'hotspot': [], 'nonhotspot': []} for sp in range(3)}

for protein in DISPLAY_ORDER:
    peak_regions = peak_dict.get(protein, [])
    for seq_pos in range(3):
        if seq_pos not in protein_idr_info[protein]:
            continue
        info       = protein_idr_info[protein][seq_pos]
        lstart     = info['lstart']
        lend       = info['lend']
        linker_num = info['linker_num']
        if (protein, linker_num) not in linker_hydrophobicity:
            continue

        hydro   = np.array(linker_hydrophobicity[(protein, linker_num)])
        idr_len = lend - lstart
        peak_idr_num = seq_pos + 1
        if protein == "P48510" and seq_pos == 2:
            peak_idr_num = 3

        # hotspot mask
        mask = np.zeros(idr_len, dtype=bool)
        for pk in peak_regions:
            if pk['IDR'] == peak_idr_num:
                s = max(0, pk['start'] - lstart)
                e = min(idr_len, pk['end'] - lstart)
                if s < e:
                    mask[s:e] = True

        # one mean per hotspot peak
        for pk in peak_regions:
            if pk['IDR'] == peak_idr_num:
                s = max(0, pk['start'] - lstart)
                e = min(idr_len, pk['end'] - lstart)
                if e > s:
                    idr_data[seq_pos]['hotspot'].append(float(np.mean(hydro[s:e])))

        # one mean per non-hotspot segment ≥ MIN_SEG_LEN
        in_seg, seg_start = False, 0
        for i in range(idr_len + 1):
            is_non = (i < idr_len) and (not mask[i])
            if is_non and not in_seg:
                seg_start, in_seg = i, True
            elif not is_non and in_seg:
                if (i - seg_start) >= MIN_SEG_LEN:
                    idr_data[seq_pos]['nonhotspot'].append(float(np.mean(hydro[seg_start:i])))
                in_seg = False

all_hs  = [v for sp in range(3) for v in idr_data[sp]['hotspot']]
all_nhs = [v for sp in range(3) for v in idr_data[sp]['nonhotspot']]


##plotting 
fig, ax = plt.subplots(figsize=(7, 7))
make_boxplot(ax, all_hs, all_nhs, 'All IDRs')
plt.tight_layout()
fig.savefig('Fig_SI15b.svg', bbox_inches='tight')
plt.close(fig)
