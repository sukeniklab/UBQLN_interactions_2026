import pandas as pd
import numpy as np
from Bio import AlignIO, Phylo, SeqIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import subprocess
import os

label_dict = {
    "P48510": "Yeast Dsk2",
    "sp|Q9UMX0|UBQL1_HUMAN": "Human UBQLN1",
    "sp|Q9UHD9|UBQL2_HUMAN": "Human UBQLN2",
    "sp|Q9NRR5|UBQL4_HUMAN": "Human UBQLN4",
    "sp|Q9SII9|DSK2A_ARATH": "Plant Dsk2A", 
    "sp|Q9SII8|DSK2B_ARATH": "Plant Dsk2B",
    "sp|G5EFF7|UBQL_CAEEL": "C. Elegans UBQN", 
    "tr|Q9VWD9|Q9VWD9_DROME": "Fly UBQN",
    "sp|Q9JJP9|UBQL1_RAT": "Rat UBQLN1",
    "tr|D4AA63|D4AA63_RAT": "Rat UBQLN2",
    "tr|D4A3P1|D4A3P1_RAT": "Rat UBQLN4",
    "tr|F6RXL5|F6RXL5_XENTR": "Frog UBQLN4",
    "sp|Q8R317|UBQL1_MOUSE": "Mouse UBQL1",
    "sp|Q9QZM0|UBQL2_MOUSE": "Mouse UBQL2",
    "sp|Q99NB8|UBQL4_MOUSE": "Mouse UBQL4",
    "tr|Q4G000|Q4G000_DANRE": "Zebra Fish UBQN"
}

# Run MUSCLE alignment
input_fasta = "Dsk2_topology.fasta"
output_aligned = "Dsk2_aligned.fasta"

# MUSCLE v5 syntax
subprocess.run([
    "muscle",
    "-align", input_fasta,
    "-output", output_aligned
], check=True, capture_output=True, text=True)

# Load  alignment
alignment = AlignIO.read(output_aligned, "fasta")

# Distance matrix
calculator = DistanceCalculator('identity')
distance_matrix = calculator.get_distance(alignment)

# Build tree using UPGMA
constructor = DistanceTreeConstructor()
tree = constructor.upgma(distance_matrix)

for leaf in tree.get_terminals():
    original_name = leaf.name
    if original_name in label_dict:
        leaf.name = label_dict[original_name]

Phylo.write(tree, "phylo_tree.nwk", "newick")
