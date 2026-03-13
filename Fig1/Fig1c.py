import matplotlib.pyplot as plt
import csv
import ast
import numpy as np


plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.size'] = 8 
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.sans-serif'] = 'Arial'

# Read the metapredict output
input_file = 'identify_disordered_regions/disorder_complete_metapredictv3_Dec17.csv'

LINKER_THRESHOLD = 30  


def merge_close_folded_domains(folded_boundaries, threshold=30):
    if len(folded_boundaries) <= 1:
        return folded_boundaries
    
    # Sort by start position
    sorted_bounds = sorted(folded_boundaries, key=lambda x: x[0])
    
    merged = [sorted_bounds[0]]
    
    for current_start, current_end in sorted_bounds[1:]:
        prev_start, prev_end = merged[-1]
        
        linker_length = current_start - prev_end - 1
        
        if linker_length < threshold: #merge
            merged[-1] = (prev_start, current_end)
        else: # Keep separate
            merged.append((current_start, current_end))
    
    return merged


def remove_contained_disordered_regions(folded_boundaries, disordered_boundaries):
#   Remove linkers <30AA 
    if not folded_boundaries or not disordered_boundaries:
        return disordered_boundaries
    
    filtered_disordered = []
    
    for dis_start, dis_end in disordered_boundaries:
        is_contained = False
        for fold_start, fold_end in folded_boundaries:
            if dis_start >= fold_start and dis_end <= fold_end:
                is_contained = True
                break
        
        # Only keep disordered regions that are NOT contained
        if not is_contained:
            filtered_disordered.append([dis_start, dis_end])
    
    return filtered_disordered

def remove_short_disordered_regions(disordered_boundaries, threshold):

    if not disordered_boundaries:
        return disordered_boundaries
    
    filtered_disordered = []
    
    for dis_start, dis_end in disordered_boundaries:
        dis_length = dis_end - dis_start + 1
        
        # Keep only disordered regions >= threshold length
        if dis_length >= threshold:
            filtered_disordered.append([dis_start, dis_end])
    
    return filtered_disordered

def get_linker_lengths(folded_bounds, disordered_bounds):
    linkers = []
    
    folded_sorted = sorted(folded_bounds, key=lambda x: x[0])
    disordered_sorted = sorted(disordered_bounds, key=lambda x: x[0])
    
    for d_start, d_end in disordered_sorted:
        # Check if there's a FD before & after 
        has_folded_before = any(f_end < d_start for f_start, f_end in folded_sorted)
        has_folded_after = any(f_start > d_end for f_start, f_end in folded_sorted)
        
        if has_folded_before and has_folded_after:
            # idr between folded domains
            linker_length = d_end - d_start + 1
            linkers.append((d_start, d_end, linker_length))
    
    return linkers

proteins_with_linkers = []

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        idx = int(row['idx'])
        name = row['name']
        seq_length = int(row['sequence_length'])
        
        folded_boundaries = ast.literal_eval(row['folded_boundaries'])
        disordered_boundaries = ast.literal_eval(row['disordered_boundaries'])
        
        # Merge folded domains with short linkers
        merged_folded_boundaries = merge_close_folded_domains(folded_boundaries, LINKER_THRESHOLD)
        
        # Remove idrs that are now within folded domains
        filtered_disordered_boundaries = remove_contained_disordered_regions(merged_folded_boundaries, disordered_boundaries)
        
        # Remove idrs < 30aa
        filtered_disordered_boundaries = remove_short_disordered_regions(filtered_disordered_boundaries, LINKER_THRESHOLD)
        
        # recount 
        num_folded_merged = len(merged_folded_boundaries)
        num_disordered_filtered = len(filtered_disordered_boundaries)
        
        
        # Process any protein with 2+ folded domains and disordered regions AFTER filtering
        if num_folded_merged >= 2 and num_disordered_filtered > 0:
            protein_data = {
                'idx': idx,
                'name': name,
                'length': seq_length,
                'num_folded': num_folded_merged,
                'num_disordered': num_disordered_filtered,
                'folded_boundaries': merged_folded_boundaries,
                'disordered_boundaries': filtered_disordered_boundaries
            }
            proteins_with_linkers.append(protein_data)




all_linker_lengths = []
linker_details = []

for protein in proteins_with_linkers:
    linkers = get_linker_lengths(
        protein['folded_boundaries'],
        protein['disordered_boundaries']
    )
    
    for start, end, length in linkers:
        all_linker_lengths.append(length)
        linker_details.append({
            'protein_idx': protein['idx'],
            'protein_name': protein['name'],
            'num_folded': protein['num_folded'],
            'linker_start': start,
            'linker_end': end,
            'linker_length': length
        })


if all_linker_lengths:
    fig, ax = plt.subplots(figsize=(4, 4))

    n, bins, patches = ax.hist(all_linker_lengths, bins=50, range=(0, 500), color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Linker length (residues)', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, max(n) * 1.1)  
    ax.tick_params(axis='both', labelsize=18)

    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    
    plt.tight_layout()
    plt.savefig("Fig3c.svg", dpi=300, bbox_inches='tight')
    plt.close()
