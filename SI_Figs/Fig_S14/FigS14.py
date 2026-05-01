import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import AlignIO, Phylo
from matplotlib.patches import Rectangle

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 2.
plt.rcParams['xtick.major.size'] = 14
plt.rcParams['xtick.major.width'] = 2.
plt.rcParams['ytick.major.size'] = 14
plt.rcParams['ytick.major.width'] = 2.
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['font.size'] = 14
plt.rcParams['font.sans-serif'] = 'Arial'

tree = Phylo.read("phylo_tree.nwk", "newick")
tree_order_labels = [leaf.name for leaf in tree.get_terminals()]

species_order = []
for label in tree_order_labels:
    for key in ["Yeast Dsk2", "Human UBQLN1", "Human UBQLN2", "Human UBQLN4", 
                "Plant Dsk2A", "Plant Dsk2B", "C. Elegans UBQLN", "Fly UBQLN",
                "Rat UBQLN1", "Rat UBQLN2", "Rat UBQLN4", "Frog UBQLN4",
                "Mouse UBQL1", "Mouse UBQL2", "Mouse UBQL4", "Zebra Fish UBQLN"]:
        if label == key:
            species_order.append(key)
            break

label_to_species = {
    "Yeast Dsk2 IDR 1": "Yeast Dsk2", "Yeast Dsk2 IDR 2": "Yeast Dsk2",
    "Human UBQLN1 IDR 1": "Human UBQLN1", "Human UBQLN1 IDR 2": "Human UBQLN1", "Human UBQLN1 IDR 3": "Human UBQLN1",
    "Human UBQLN2 IDR 1": "Human UBQLN2", "Human UBQLN2 IDR 2": "Human UBQLN2", "Human UBQLN2 IDR 3": "Human UBQLN2",
    "Human UBQLN4 IDR 1": "Human UBQLN4", "Human UBQLN4 IDR 2": "Human UBQLN4", "Human UBQLN4 IDR 3": "Human UBQLN4",
    "Plant Dsk2A IDR 1": "Plant Dsk2A", "Plant Dsk2A IDR 2": "Plant Dsk2A", "Plant Dsk2A IDR 3": "Plant Dsk2A",
    "Plant Dsk2B IDR 1": "Plant Dsk2B", "Plant Dsk2B IDR 2": "Plant Dsk2B", "Plant Dsk2B IDR 3": "Plant Dsk2B",
    "C. elegans IDR 1": "C. Elegans UBQLN", "C. elegans IDR 2": "C. Elegans UBQLN", "C. elegans IDR 3": "C. Elegans UBQLN",
    "Fly IDR 1": "Fly UBQLN", "Fly IDR 2": "Fly UBQLN", "Fly IDR 3": "Fly UBQLN",
    "Rat UBQLN1 IDR 1": "Rat UBQLN1", "Rat UBQLN1 IDR 2": "Rat UBQLN1", "Rat UBQLN1 IDR 3": "Rat UBQLN1",
    "Rat UBQLN2 IDR 1": "Rat UBQLN2", "Rat UBQLN2 IDR 2": "Rat UBQLN2", "Rat UBQLN2 IDR 3": "Rat UBQLN2",
    "Rat UBQLN4 IDR 1": "Rat UBQLN4", "Rat UBQLN4 IDR 2": "Rat UBQLN4", "Rat UBQLN4 IDR 3": "Rat UBQLN4",
    "Frog IDR 1": "Frog UBQLN4", "Frog IDR 2": "Frog UBQLN4", "Frog IDR 3": "Frog UBQLN4",
    "Mouse UBQL1 IDR 1": "Mouse UBQL1", "Mouse UBQL1 IDR 2": "Mouse UBQL1", "Mouse UBQL1 IDR 3": "Mouse UBQL1",
    "Mouse UBQL2 IDR 1": "Mouse UBQL2", "Mouse UBQL2 IDR 2": "Mouse UBQL2", "Mouse UBQL2 IDR 3": "Mouse UBQL2",
    "Mouse UBQL4 IDR 1": "Mouse UBQL4", "Mouse UBQL4 IDR 2": "Mouse UBQL4", "Mouse UBQL4 IDR 3": "Mouse UBQL4",
    "Zebrafish IDR 1": "Zebra Fish UBQLN", "Zebrafish IDR 2": "Zebra Fish UBQLN", "Zebrafish IDR 3": "Zebra Fish UBQLN"
}

label_dict = {
    # Linker1
    "sp|P48510|DSK2_YEAST_Linker1": "Yeast Dsk2 IDR 1",
    "sp|Q9UMX0|UBQL1_HUMAN_Linker1": "Human UBQLN1 IDR 1",
    "sp|Q9UHD9|UBQL2_HUMAN_Linker1": "Human UBQLN2 IDR 1",
    "sp|Q9NRR5|UBQL4_HUMAN_Linker1": "Human UBQLN4 IDR 1",
    "sp|Q9SII8|DSK2B_ARATH_Linker1": "Plant Dsk2B IDR 1",
    "sp|Q9SII9|DSK2A_ARATH_Linker1": "Plant Dsk2A IDR 1",
    "sp|G5EFF7|UBQL_CAEEL_Linker1": "C. elegans UBQLN IDR 1",
    "tr|Q9VWD9|Q9VWD9_DROME_Linker1": "Fly UBQLN IDR 1",
    "sp|Q9JJP9|UBQL1_RAT_Linker1": "Rat UBQLN1 IDR 1",
    "tr|D4AA63|D4AA63_RAT_Linker1": "Rat UBQLN2 IDR 1",
    "tr|D4A3P1|D4A3P1_RAT_Linker1": "Rat UBQLN4 IDR 1",
    "tr|F6RXL5|F6RXL5_XENTR_Linker1": "Frog UBQLN IDR 1",
    "sp|Q9QZM0|UBQL2_MOUSE_Linker1": "Mouse UBQL2 IDR 1",
    "sp|Q99NB8|UBQL4_MOUSE_Linker1": "Mouse UBQL4 IDR 1",
    "sp|Q8R317|UBQL1_MOUSE_Linker1": "Mouse UBQL1 IDR 1",
    "tr|Q4G000|Q4G000_DANRE_Linker1": "Zebrafish IDR 1",
    
    # Linker2
    "sp|Q9UMX0|UBQL1_HUMAN_Linker2": "Human UBQLN1 IDR 2",
    "sp|Q9UHD9|UBQL2_HUMAN_Linker2": "Human UBQLN2 IDR 2",
    "sp|Q9NRR5|UBQL4_HUMAN_Linker2": "Human UBQLN4 IDR 2",
    "sp|Q9SII8|DSK2B_ARATH_Linker2": "Plant Dsk2B IDR 2",
    "sp|Q9SII9|DSK2A_ARATH_Linker2": "Plant Dsk2A IDR 2",
    "sp|G5EFF7|UBQL_CAEEL_Linker2": "C. elegans UBQLN IDR 2",
    "tr|Q9VWD9|Q9VWD9_DROME_Linker2": "Fly UBQLN IDR 2",
    "sp|Q9JJP9|UBQL1_RAT_Linker2": "Rat UBQLN1 IDR 2",
    "tr|D4AA63|D4AA63_RAT_Linker2": "Rat UBQLN2 IDR 2",
    "tr|D4A3P1|D4A3P1_RAT_Linker2": "Rat UBQLN4 IDR 2",
    "tr|F6RXL5|F6RXL5_XENTR_Linker2": "Frog UBQLN IDR 2",
    "sp|Q9QZM0|UBQL2_MOUSE_Linker2": "Mouse UBQL2 IDR 2",
    "sp|Q99NB8|UBQL4_MOUSE_Linker2": "Mouse UBQL4 IDR 2",
    "sp|Q8R317|UBQL1_MOUSE_Linker2": "Mouse UBQL1 IDR 2",
    "tr|Q4G000|Q4G000_DANRE_Linker2": "Zebrafish IDR 2",
    
    # Linker3
    "sp|P48510|DSK2_YEAST_Linker3": "Yeast Dsk2 IDR 2",
    "sp|Q9UMX0|UBQL1_HUMAN_Linker3": "Human UBQLN1 IDR 3",
    "sp|Q9UHD9|UBQL2_HUMAN_Linker3": "Human UBQLN2 IDR 3",
    "sp|Q9NRR5|UBQL4_HUMAN_Linker3": "Human UBQLN4 IDR 3",
    "sp|Q9SII8|DSK2B_ARATH_Linker3": "Plant Dsk2B IDR 3",
    "sp|Q9SII9|DSK2A_ARATH_Linker3": "Plant Dsk2A IDR 3",
    "sp|G5EFF7|UBQL_CAEEL_Linker3": "C. elegans UBQLN IDR 3",
    "tr|Q9VWD9|Q9VWD9_DROME_Linker3": "Fly UBQLN IDR 3",
    "sp|Q9JJP9|UBQL1_RAT_Linker3": "Rat UBQLN1 IDR 3",
    "tr|D4AA63|D4AA63_RAT_Linker3": "Rat UBQLN2 IDR 3",
    "tr|D4A3P1|D4A3P1_RAT_Linker3": "Rat UBQLN4 IDR 3",
    "tr|F6RXL5|F6RXL5_XENTR_Linker3": "Frog IDR 3",
    "sp|Q9QZM0|UBQL2_MOUSE_Linker3": "Mouse UBQL2 IDR 3",
    "sp|Q99NB8|UBQL4_MOUSE_Linker3": "Mouse UBQL4 IDR 3",
    "sp|Q8R317|UBQL1_MOUSE_Linker3": "Mouse UBQL1 IDR 3",
    "tr|Q4G000|Q4G000_DANRE_Linker3": "Zebrafish IDR 3"
}


alignment = AlignIO.read("Linkers_aligned.fasta", "fasta")
n_seqs = len(alignment)
seq_ids = [record.id for record in alignment]

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

matrix = pd.DataFrame(identity_matrix, index=seq_ids, columns=seq_ids)

matrix = matrix.rename(index=label_dict, columns=label_dict)

ordered_labels = []
for species in species_order:
    # Add IDR 1
    for label, sp in label_to_species.items():
        if sp == species and " IDR 1" in label and label in matrix.index:
            ordered_labels.append(label)
            break

for species in species_order:
    # Add IDR 2 
    for label, sp in label_to_species.items():
        if sp == species and " IDR 2" in label and label in matrix.index and label != "Yeast Dsk2 IDR 2":
            ordered_labels.append(label)
            break

for species in species_order:
    # Add IDR 3 
    for label, sp in label_to_species.items():
        if sp == species and " IDR 3" in label and label in matrix.index:
            ordered_labels.append(label)
            break

if "Yeast Dsk2 IDR 2" in matrix.index:
    ordered_labels.append("Yeast Dsk2 IDR 2")

matrix = matrix.reindex(index=ordered_labels, columns=ordered_labels)

n_idr1 = sum(1 for l in ordered_labels if " IDR 1" in l)
n_idr2 = sum(1 for l in ordered_labels if " IDR 2" in l)
n_idr3 = sum(1 for l in ordered_labels if " IDR 3" in l)

print(f"IDR 1: {n_idr1}, IDR 2: {n_idr2}, IDR 3: {n_idr3}")

fig, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(matrix, cmap="Reds", square=True, 
            linewidths=0.5, linecolor='lightgray',
            vmin=0, vmax=100,
            cbar_kws={
                'label': 'Percent Identity (%)',
                'orientation': 'horizontal',
                'shrink': 0.4,
                'aspect': 40,
                'pad': 0.05
            },
            ax=ax)

# Add separator lines between IDR 1/IDR 2/IDR 3 blocks
ax.axhline(y=n_idr1, color='black', linewidth=3)
ax.axhline(y=n_idr1 + n_idr2, color='black', linewidth=3)
ax.axvline(x=n_idr1, color='black', linewidth=3)
ax.axvline(x=n_idr1 + n_idr2, color='black', linewidth=3)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=90, ha='left', fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.savefig("FigS14.svg", dpi=300, bbox_inches='tight')
plt.close()
