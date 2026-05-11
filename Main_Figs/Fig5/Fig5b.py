import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import AlignIO, Phylo
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 2.
plt.rcParams['xtick.major.size'] = 14
plt.rcParams['xtick.major.width'] = 2.
plt.rcParams['ytick.major.size'] = 14
plt.rcParams['ytick.major.width'] = 2.
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['font.size'] = 24
plt.rcParams['font.sans-serif'] = 'Arial'

label_dict = {
    "sp|P48510|DSK2_YEAST_UBA": "Yeast Dsk2",
    "sp|Q9UMX0|UBQL1_HUMAN_UBA": "Human UBQLN1",
    "sp|Q9UHD9|UBQL2_HUMAN_UBA": "Human UBQLN2",
    "sp|Q9NRR5|UBQL4_HUMAN_UBA": "Human UBQLN4",
    "sp|Q9SII9|DSK2A_ARATH_UBA": "Plant Dsk2A", 
    "sp|Q9SII8|DSK2B_ARATH_UBA": "Plant Dsk2B",
    "sp|G5EFF7|UBQL_CAEEL_UBA": "C. Elegans UBQN", 
    "tr|Q9VWD9|Q9VWD9_DROME_UBA": "Fly UBQN",
    "sp|Q9JJP9|UBQL1_RAT_UBA": "Rat UBQLN1",
    "tr|D4AA63|D4AA63_RAT_UBA": "Rat UBQLN2",
    "tr|D4A3P1|D4A3P1_RAT_UBA": "Rat UBQLN4",
    "tr|F6RXL5|F6RXL5_XENTR_UBA": "Frog UBQLN4",
    "sp|Q8R317|UBQL1_MOUSE_UBA": "Mouse UBQL1",
    "sp|Q9QZM0|UBQL2_MOUSE_UBA": "Mouse UBQL2",
    "sp|Q99NB8|UBQL4_MOUSE_UBA": "Mouse UBQL4",
    "tr|Q4G000|Q4G000_DANRE_UBA": "Zebra Fish UBQN"
}


alignment = AlignIO.read("domain_fastas/UBA_aligned.fasta", "fasta")
n_seqs = len(alignment)
seq_ids = [record.id for record in alignment]


# Calculate pairwise percent identity matrix
identity_matrix = np.zeros((n_seqs, n_seqs))

for i in range(n_seqs):
    for j in range(n_seqs):
        if i == j:
            identity_matrix[i][j] = 100.0
        else:
            seq_i = alignment[i].seq
            seq_j = alignment[j].seq
            
            matches = 0
            valid_positions = 0
            
            for a, b in zip(seq_i, seq_j):
                if a != '-' and b != '-':
                    valid_positions += 1
                    if a == b:
                        matches += 1
            
            if valid_positions > 0:
                identity_matrix[i][j] = (matches / valid_positions) * 100
            else:
                identity_matrix[i][j] = 0.0

# Create DataFrame
matrix = pd.DataFrame(identity_matrix, index=seq_ids, columns=seq_ids)
matrix = matrix.rename(index=label_dict, columns=label_dict)


tree = Phylo.read("phylo_tree.nwk", "newick")

tree_order = [leaf.name for leaf in tree.get_terminals()]
available_tree_order = [name for name in tree_order if name in matrix.index]
matrix = matrix.reindex(index=available_tree_order, columns=available_tree_order)

# Create figure with just heatmap
fig, ax_heatmap = plt.subplots(figsize=(12, 10))

# Create mask for upper triangle
mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

sns.heatmap(matrix, cmap="Purples", square=True, 
            linewidths=0.5, linecolor='lightgray',
            vmin=0, vmax=100,
            mask=mask,
            annot=False, fmt='.1f',
            cbar_kws={
                'label': 'Percent Identity (%)',
                'orientation': 'horizontal',
                'shrink': 0.8,
                'aspect': 20,
                'pad': 0.08
            },
            yticklabels=True,
            xticklabels=True,
            ax=ax_heatmap)

for i in range(len(matrix)):
    for j in range(i+1, len(matrix)):
        ax_heatmap.add_patch(Rectangle((j, i), 1, 1, fill=True, 
                                       facecolor='white', edgecolor='white', 
                                       linewidth=0, zorder=10))

# Move x-axis labels to bottom
ax_heatmap.xaxis.tick_bottom()
ax_heatmap.xaxis.set_label_position('bottom')
ax_heatmap.set_ylabel('')

plt.setp(ax_heatmap.get_xticklabels(), rotation=90, ha='right')
plt.setp(ax_heatmap.get_yticklabels(), rotation=0)


plt.savefig("Fig5b.svg", dpi=300, bbox_inches='tight')
plt.close()

