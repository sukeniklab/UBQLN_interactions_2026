import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from Bio import Phylo, SeqIO

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['font.size'] = 34
plt.rcParams['font.sans-serif'] = 'Arial'

def smooth_profile(arr, window=5):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')

def load_hel_file(path):
    df = pd.read_csv(path, header=None, names=HEL_COLS, skipinitialspace=True)
    df = df[df['res'].astype(str).str.strip() != 'res'].reset_index(drop=True)
    df = df[~df['aa'].astype(str).str.strip().isin(['B', 'U'])].reset_index(drop=True)
    hel = df['Hel'].astype(float).values
    return hel

def parse_sequence_name(seq_name):
    uid = None
    for known_uid in prot_name_map:
        if known_uid in seq_name:
            uid = known_uid
            break
    linker_num = None
    m = re.search(r'_Linker(\d+)', seq_name)
    if m:
        linker_num = int(m.group(1))
    return uid, linker_num

STI1_DOMAIN = 1
HYDRO_SMOOTH_WINDOW = 5

HEL_Y_MIN, HEL_Y_MAX = -5, 30
HYDRO_Y_MIN, HYDRO_Y_MAX = -2, 1.6

COL_HYDRO = '#2874A6'
COL_HEL = '#C0392B'
COL_PEAK = '#FF8C42'

HEL_COLS = ['res', 'aa', 'Hel', 'Ncap', 'Ccap', 'Hstaple', 'Schellm', 'CaH', '13Ca', 'JaN']

HYDROPHOBICITY = {
    'A': 0.620, 'R': -2.530, 'N': -0.780, 'D': -0.900, 'C': 0.290,
    'Q': -0.850, 'E': -0.740, 'G': 0.480, 'H': -0.400, 'I': 1.380,
    'L': 1.060, 'K': -1.500, 'M': 0.640, 'F': 1.190, 'P': 0.120,
    'S': -0.180, 'T': -0.050, 'W': 0.810, 'Y': 0.260, 'V': 1.080,
    'X': 0.0, 'B': 0.0, 'Z': 0.0, 'U': 0.0, 'O': 0.0,
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
    "P48510": "DSK2_YEAST",    "Q9UMX0": "UBQL1_HUMAN", "Q9UHD9": "UBQL2_HUMAN",
    "Q9NRR5": "UBQL4_HUMAN",   "Q9SII8": "Dsk2B_PLANT", "Q9SII9": "Dsk2A_PLANT",
    "D4A3P1": "UBQL4_RAT",     "D4AA63": "UBQL2_RAT",   "Q9JJP9": "UBQL1_RAT",
    "Q9VWD9": "UBQN_Fly",      "G5EFF7": "UBQL_CElegans",
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
all_tips = [c.name for c in tree.get_terminals()]

_pid_to_tree_pos = {}
for rank, tip_name in enumerate(all_tips):
    pid = TREE_NAME_TO_PID.get(tip_name)
    if pid and pid in prot_name_map and pid not in _pid_to_tree_pos:
        _pid_to_tree_pos[pid] = rank
DISPLAY_ORDER[:] = sorted(_pid_to_tree_pos.keys(), key=lambda p: _pid_to_tree_pos[p])

protein_idr_info = {}
for protein in prot_name_map:
    protein_idr_info[protein] = {}
    offs = offset_dict[protein]
    for linker_num, offset_idx in linker_file_mapping.get(protein, {}).items():
        if offset_idx >= len(offs):
            continue
        lstart = offs[offset_idx]
        lend   = (domain_dict[protein][offset_idx + 1][0] - 1
                  if offset_idx + 1 < len(domain_dict[protein]) else lstart + 100)
        protein_idr_info[protein][offset_idx] = {
            'lstart': lstart, 'lend': lend, 'linker_num': linker_num
        }
        if protein == "P48510" and offset_idx == 1:
            protein_idr_info[protein][2] = {
                'lstart': lstart, 'lend': lend, 'linker_num': linker_num
            }

peak_dict = {}
peak_dict_all = {}

peak_summary_path = "ALL_PROTEINS_peaks_summary.csv"
peak_df = pd.read_csv(peak_summary_path)
for _, row in peak_df.iterrows():
    uni_id = peak_to_uniprot.get(row['protein'])
    if uni_id:
        peak_dict_all.setdefault(uni_id, []).append({
            'start': int(row['start']), 'end': int(row['end']),
            'IDR': int(row['IDR']), 'STI1_domain': int(row['STI1_domain']),
            'condition': row['condition'],
        })
for _, row in peak_df[peak_df['STI1_domain'] == STI1_DOMAIN].iterrows():
    uni_id = peak_to_uniprot.get(row['protein'])
    if uni_id:
        peak_dict.setdefault(uni_id, []).append({
            'start': int(row['start']), 'end': int(row['end']),
            'IDR': int(row['IDR']), 'STI1_domain': int(row['STI1_domain']),
            'condition': row['condition'],
        })

# Load window map
window_map_path = "agadir_window_map.tsv"
if not os.path.exists(window_map_path):
    raise FileNotFoundError(f"Window map not found: {window_map_path}")

window_map_df = pd.read_csv(window_map_path, sep='\t')

# Initialise accumulators
hel_sum   = {}
hel_count = {}

for uid in prot_name_map:
    for linker_num, offset_idx in linker_file_mapping.get(uid, {}).items():
        if offset_idx in protein_idr_info[uid]:
            n = (protein_idr_info[uid][offset_idx]['lend'] -
                 protein_idr_info[uid][offset_idx]['lstart'])
            hel_sum[(uid, linker_num)]   = np.zeros(n)
            hel_count[(uid, linker_num)] = np.zeros(n, dtype=int)

skipped, loaded = [], 0

for _, row in window_map_df.iterrows():
    window_id = row['window_id']
    seq_name  = row['sequence_name']
    seq_start = int(row['seq_start'])
    t_start   = int(row['trusted_seq_start'])
    t_end     = int(row['trusted_seq_end'])
    win_len   = int(row['win_len'])

    uid, linker_num = parse_sequence_name(seq_name)
    if uid is None or linker_num is None:
        skipped.append((window_id, f"unrecognised protein in '{seq_name}'"))
        continue

    key = (uid, linker_num)
    if key not in hel_sum:
        skipped.append((window_id, f"no accumulator for {uid} linker {linker_num}"))
        continue

    hel_path = os.path.join("helicity_scores", f"{window_id}.HEL")
    if not os.path.exists(hel_path):
        skipped.append((window_id, "no .HEL file"))
        continue

    hel_vals = load_hel_file(hel_path)

    if win_len < 30:
        win_trust_start, win_trust_end = 0, win_len
    else:
        win_trust_start = t_start - seq_start
        win_trust_end   = t_end   - seq_start

    trusted_hel = hel_vals[win_trust_start:win_trust_end]

    n_arr     = len(hel_sum[key])
    arr_start = t_start
    arr_end   = min(t_end, n_arr)
    clip_len  = arr_end - arr_start

    if arr_start >= n_arr or clip_len <= 0:
        skipped.append((window_id, "trusted zone out of linker bounds"))
        continue

    hel_sum[key][arr_start:arr_end]   += trusted_hel[:clip_len]
    hel_count[key][arr_start:arr_end] += 1
    loaded += 1



linker_helicity = {}
linker_coverage = {}

for key in list(hel_sum.keys()):
    c = hel_count[key]
    if c.max() == 0:
        continue
    profile = np.where(c > 0, hel_sum[key] / c, np.nan)
    linker_helicity[key] = profile
    linker_coverage[key] = c > 0


for protein in DISPLAY_ORDER:
    peak_regions = peak_dict.get(protein, [])
    for seq_pos in range(3):
        if seq_pos not in protein_idr_info[protein]:
            continue
        info       = protein_idr_info[protein][seq_pos]
        lstart     = info['lstart']
        lend       = info['lend']
        linker_num = info['linker_num']

        if (protein, linker_num) not in linker_helicity:
            continue

        hel          = linker_helicity[(protein, linker_num)]
        peak_idr_num = seq_pos + 1
        if protein == "P48510" and seq_pos == 2:
            peak_idr_num = 3

        for pk in peak_regions:
            if pk['IDR'] != peak_idr_num:
                continue
            s = max(0, pk['start'] - lstart)
            e = min(len(hel), pk['end'] - lstart)
            if e <= s:
                continue

            peak_vals = hel[s:e]
            peak_mean = float(np.nanmean(peak_vals))
            peak_max  = float(np.nanmax(peak_vals))


linker_hydrophobicity = {}

linkers_all_path = "Linkers_all.fasta"
if not os.path.exists(linkers_all_path):
    raise FileNotFoundError(f"Linkers_all.fasta not found: {linkers_all_path}")

for record in SeqIO.parse(linkers_all_path, "fasta"):
    parts = record.id.split('|')
    if len(parts) < 3:
        continue
    uniprot_id = parts[1]
    name_part  = parts[2]
    if '_Linker' not in name_part or uniprot_id not in prot_name_map:
        continue
    try:
        linker_num = int(name_part.split('_Linker')[-1])
    except ValueError:
        continue
    seq = str(record.seq)
    linker_hydrophobicity[(uniprot_id, linker_num)] = np.array(
        [HYDROPHOBICITY.get(aa.upper(), 0.0) for aa in seq]
    )

idr_max_lengths = {}
for seq_pos in range(3):
    idr_max_lengths[seq_pos] = max(
        (protein_idr_info[p][seq_pos]['lend'] - protein_idr_info[p][seq_pos]['lstart']
         for p in DISPLAY_ORDER if seq_pos in protein_idr_info[p]),
        default=0
    )

GAP_SIZE = 50
aligned_positions = {}
cumulative_pos = 0
for seq_pos in range(3):
    aligned_positions[seq_pos] = {
        'start': cumulative_pos,
        'end':   cumulative_pos + idr_max_lengths[seq_pos],
    }
    cumulative_pos += idr_max_lengths[seq_pos] + GAP_SIZE

total_width = cumulative_pos - GAP_SIZE
boundary_idx = max(i for i, p in enumerate(DISPLAY_ORDER) if p in DEEP_PIDS)

fig, axes = plt.subplots(
    len(DISPLAY_ORDER), 1,
    figsize=(24, 2.8 * len(DISPLAY_ORDER)),
    sharex=False,
)
if len(DISPLAY_ORDER) == 1:
    axes = [axes]

twin_axes = []

for prot_idx, protein in enumerate(DISPLAY_ORDER):
    ax_h  = axes[prot_idx]   # left  axis — hydrophobicity (blue)
    ax_he = ax_h.twinx()     # right axis — helicity       (red)
    twin_axes.append(ax_he)

    is_deep = protein in DEEP_PIDS
    peak_regions = peak_dict.get(protein, [])

    for seq_pos in range(3):
        if seq_pos not in protein_idr_info[protein]:
            continue

        info = protein_idr_info[protein][seq_pos]
        lstart = info['lstart']
        lend = info['lend']
        linker_num = info['linker_num']
        length = lend - lstart

        peak_idr_num = seq_pos + 1
        if protein == "P48510" and seq_pos == 2:
            peak_idr_num = 3

        slot_start = aligned_positions[seq_pos]['start']
        slot_width = idr_max_lengths[seq_pos]
        offset = (slot_width - length) / 2
        plot_start = slot_start + offset
        plot_end = plot_start + length
        ax_h.plot([plot_start, plot_end], [0, 0],
                  color='lightgrey', linewidth=12,
                  solid_capstyle='butt', zorder=1)

        if prot_idx == 0:
            mid_x = (slot_start + aligned_positions[seq_pos]['end']) / 2
            ax_h.text(mid_x, HYDRO_Y_MAX * 1.12,
                      IDR_LABELS[seq_pos],
                      fontsize=34, fontweight='bold',
                      ha='center', va='bottom', color='#34495E')

        key = (protein, linker_num)

        if key in linker_hydrophobicity:
            hydro = linker_hydrophobicity[key]
            hydro_smooth = smooth_profile(hydro, window=HYDRO_SMOOTH_WINDOW)
            xc = np.linspace(plot_start, plot_end, len(hydro_smooth))
            ax_h.plot(xc, hydro_smooth,
                      color=COL_HYDRO, linewidth=3.5, zorder=5,
                      label='Hydrophobicity (Eisenberg, smoothed)'
                      if (seq_pos == 0 and prot_idx == 0) else '')

        if key in linker_helicity:
            hel = linker_helicity[key]
            xc = np.linspace(plot_start, plot_end, len(hel))
            ax_he.plot(xc, hel,
                       color=COL_HEL, linewidth=3.5,  zorder=6,
                       label='Helicity (AGADIR avg)'
                       if (seq_pos == 0 and prot_idx == 0) else '')

        for peak in [p for p in peak_regions if p['IDR'] == peak_idr_num]:
            offset_in_idr = peak['start'] - lstart
            peak_plot_start = plot_start + offset_in_idr
            plot_width = peak['end'] - peak['start']

            ax_h.add_patch(patches.Rectangle(
                (peak_plot_start, HYDRO_Y_MIN), plot_width,
                HYDRO_Y_MAX - HYDRO_Y_MIN,
                linewidth=0, facecolor=COL_PEAK, alpha=0.25, zorder=2,
            ))

    for seq_pos in range(2):
        bx = aligned_positions[seq_pos]['end'] + GAP_SIZE / 2
        ax_h.axvline(x=bx, color='#BDC3C7', linewidth=2,
                     linestyle='--', alpha=0.6, zorder=0)

    ax_h.set_ylim(HYDRO_Y_MIN, HYDRO_Y_MAX)
    ax_h.set_xlim(-10, total_width + 10)
    ax_h.set_yticks([-2, -1, 0, 1])
    ax_h.set_yticklabels(['-2', '-1', '0', '1'],
                          fontsize=34, color=COL_HYDRO)
    ax_h.tick_params(axis='y', colors=COL_HYDRO,
                     length=8, width=2)
    ax_h.spines['left'].set_color(COL_HYDRO)
    ax_h.spines['top'].set_visible(False)
    ax_h.axhline(0, color=COL_HYDRO, linewidth=0.6, linestyle=':', alpha=0.4)

    ax_he.set_ylim(HEL_Y_MIN, HEL_Y_MAX)
    ax_he.set_yticks([0, 10, 20, 30])
    ax_he.set_yticklabels([0, 10, 20, 30],
                           fontsize=34, color=COL_HEL)
    ax_he.tick_params(axis='y', colors=COL_HEL,
                      length=8, width=2)
    ax_he.spines['right'].set_color(COL_HEL)
    ax_he.spines['top'].set_visible(False)
    ax_he.spines['left'].set_visible(False)

    TICK_INTERVAL = 30
    tick_positions, tick_labels = [], []

    for seq_pos in range(3):
        if seq_pos not in protein_idr_info[protein]:
            continue
        info_t = protein_idr_info[protein][seq_pos]
        lstart_t = info_t['lstart']
        lend_t = info_t['lend']
        ps_t = (aligned_positions[seq_pos]['start'] +
                (idr_max_lengths[seq_pos] - (lend_t - lstart_t)) / 2)

        first_res = ((lstart_t // TICK_INTERVAL) + 1) * TICK_INTERVAL
        idr_ticks = list(range(first_res, lend_t + 1, TICK_INTERVAL))

        if len(idr_ticks) < 2:
            mid = (lstart_t + lend_t) // 2
            t1 = mid - 15
            t2 = t1 + 30
            idr_ticks = [t1, t2]

        for res in idr_ticks:
            tick_positions.append(ps_t + (res - lstart_t))
            tick_labels.append(str(res))

    ax_h.set_xticks(tick_positions)
    ax_h.set_xticklabels(tick_labels, fontsize=34)
    ax_h.tick_params(axis='x', length=8, width=2)

    if prot_idx == boundary_idx:
        ax_h.axhline(y=HYDRO_Y_MIN + 0.05, color='#7F8C8D',
                     linewidth=1.5, linestyle='--', zorder=10)

    if prot_idx == len(DISPLAY_ORDER) - 1:
        ax_h.set_xlabel("Residue", fontsize=34)


plt.tight_layout(pad=0.5)
plt.subplots_adjust(hspace=0.35)

plt.savefig(f"Fig_SI15a.svg", bbox_inches='tight')
plt.close()