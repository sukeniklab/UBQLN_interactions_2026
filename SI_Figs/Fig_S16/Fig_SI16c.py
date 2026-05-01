import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import AlignIO, Phylo

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

sti1_1_label_dict = {
    "sp|P48510|DSK2_YEAST_STI1": "Yeast Dsk2",
    "sp|Q9UMX0|UBQL1_HUMAN_STI1-1": "Human UBQLN1",
    "sp|Q9UHD9|UBQL2_HUMAN_STI1-1": "Human UBQLN2",
    "sp|Q9NRR5|UBQL4_HUMAN_STI1-1": "Human UBQLN4",
    "sp|Q9SII9|DSK2A_ARATH_STI1-1": "Plant Dsk2A", 
    "sp|Q9SII8|DSK2B_ARATH_STI1-1": "Plant Dsk2B",
    "sp|G5EFF7|UBQL_CAEEL_STI1-1": "C. Elegans UBQN", 
    "tr|Q9VWD9|Q9VWD9_DROME_STI1-1": "Fly UBQN",
    "sp|Q9JJP9|UBQL1_RAT_STI1-1": "Rat UBQLN1",
    "tr|D4AA63|D4AA63_RAT_STI1-1": "Rat UBQLN2",
    "tr|D4A3P1|D4A3P1_RAT_STI1-1": "Rat UBQLN4",
    "tr|F6RXL5|F6RXL5_XENTR_STI1-1": "Frog UBQLN4",
    "sp|Q8R317|UBQL1_MOUSE_STI1-1": "Mouse UBQL1",
    "sp|Q9QZM0|UBQL2_MOUSE_STI1-1": "Mouse UBQL2",
    "sp|Q99NB8|UBQL4_MOUSE_STI1-1": "Mouse UBQL4",
    "tr|Q4G000|Q4G000_DANRE_STI1-1": "Zebra Fish UBQN"
}

sti1_2_label_dict = {
    "sp|P48510|DSK2_YEAST_STI1": "Yeast Dsk2",
    "sp|Q9UMX0|UBQL1_HUMAN_STI1-2": "Human UBQLN1",
    "sp|Q9UHD9|UBQL2_HUMAN_STI1-2": "Human UBQLN2",
    "sp|Q9NRR5|UBQL4_HUMAN_STI1-2": "Human UBQLN4",
    "sp|Q9SII9|DSK2A_ARATH_STI1-2": "Plant Dsk2A", 
    "sp|Q9SII8|DSK2B_ARATH_STI1-2": "Plant Dsk2B",
    "sp|G5EFF7|UBQL_CAEEL_STI1-2": "C. Elegans UBQN", 
    "tr|Q9VWD9|Q9VWD9_DROME_STI1-2": "Fly UBQN",
    "sp|Q9JJP9|UBQL1_RAT_STI1-2": "Rat UBQLN1",
    "tr|D4AA63|D4AA63_RAT_STI1-2": "Rat UBQLN2",
    "tr|D4A3P1|D4A3P1_RAT_STI1-2": "Rat UBQLN4",
    "tr|F6RXL5|F6RXL5_XENTR_STI1-2": "Frog UBQLN4",
    "sp|Q8R317|UBQL1_MOUSE_STI1-2": "Mouse UBQL1",
    "sp|Q9QZM0|UBQL2_MOUSE_STI1-2": "Mouse UBQL2",
    "sp|Q99NB8|UBQL4_MOUSE_STI1-2": "Mouse UBQL4",
    "tr|Q4G000|Q4G000_DANRE_STI1-2": "Zebra Fish UBQN"
}

alignment = AlignIO.read("STI1_aligned.fasta", "fasta")


sti1_1_records = [rec for rec in alignment if rec.id.endswith("STI1-1") or rec.id == "sp|P48510|DSK2_YEAST_STI1"]
sti1_2_records = [rec for rec in alignment if rec.id.endswith("STI1-2") or rec.id == "sp|P48510|DSK2_YEAST_STI1"]


# Calculate cross-domain percent identity matrix
identity_matrix = np.zeros((len(sti1_1_records), len(sti1_2_records)))

for i, rec_1 in enumerate(sti1_1_records):
    for j, rec_2 in enumerate(sti1_2_records):
        seq_1 = rec_1.seq
        seq_2 = rec_2.seq
        
        matches = 0
        valid_positions = 0
        
        for a, b in zip(seq_1, seq_2):
            if a != '-' and b != '-':
                valid_positions += 1
                if a == b:
                    matches += 1
        
        if valid_positions > 0:
            identity_matrix[i][j] = (matches / valid_positions) * 100
        else:
            identity_matrix[i][j] = 0.0

sti1_1_ids = [rec.id for rec in sti1_1_records]
sti1_2_ids = [rec.id for rec in sti1_2_records]

matrix = pd.DataFrame(identity_matrix, index=sti1_1_ids, columns=sti1_2_ids)
matrix = matrix.rename(index=sti1_1_label_dict, columns=sti1_2_label_dict)


tree = Phylo.read("phylo_tree.nwk", "newick")

tree_order = [leaf.name for leaf in tree.get_terminals()]
available_tree_order_rows = [name for name in tree_order if name in matrix.index]
available_tree_order_cols = [name for name in tree_order if name in matrix.columns]

# Reorder both axes by tree
matrix = matrix.reindex(index=available_tree_order_rows, columns=available_tree_order_cols)

fig, ax_heatmap = plt.subplots(figsize=(14, 12))

sns.heatmap(matrix, cmap="Greys", square=False, 
            linewidths=0.5, linecolor='lightgray',
            vmin=0, vmax=100,
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

ax_heatmap.xaxis.tick_bottom()
ax_heatmap.xaxis.set_label_position('bottom')
ax_heatmap.set_ylabel('STI1-1', fontsize=28)
ax_heatmap.set_xlabel('STI1-2', fontsize=28)

plt.setp(ax_heatmap.get_xticklabels(), rotation=90, ha='left')
plt.setp(ax_heatmap.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig("Fig_SI16c.svg", dpi=300, bbox_inches='tight')
plt.close()
