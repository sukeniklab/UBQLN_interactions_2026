import csv
import matplotlib.pyplot as plt
import ast
import numpy as np

# Apply styling
plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.size'] = 8 
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.sans-serif'] = 'Arial'

# Read the metapredict output
input_file = 'disorder_complete_metapredictv3_Dec17.csv'

LINKER_THRESHOLD = 30  # Merge folded domains separated by < 30 AA

def merge_close_folded_domains(folded_boundaries, threshold=30):
    """Merge folded domains separated by linkers shorter than threshold"""
    if len(folded_boundaries) <= 1:
        return folded_boundaries
    
    # Sort by start position
    sorted_bounds = sorted(folded_boundaries, key=lambda x: x[0])
    
    merged = [sorted_bounds[0]]
    
    for current_start, current_end in sorted_bounds[1:]:
        prev_start, prev_end = merged[-1]
        
        # Calculate linker length between domains
        linker_length = current_start - prev_end - 1
        
        if linker_length < threshold:
            # Merge: extend the previous domain to include current
            merged[-1] = (prev_start, current_end)
        else:
            # Keep separate
            merged.append((current_start, current_end))
    
    return merged

def remove_contained_disordered_regions(folded_boundaries, disordered_boundaries):
    """Remove disordered regions that are completely contained within folded domains"""
    if not folded_boundaries or not disordered_boundaries:
        return disordered_boundaries
    
    filtered_disordered = []
    
    for dis_start, dis_end in disordered_boundaries:
        # Check if this disordered region is contained in any folded domain
        is_contained = False
        for fold_start, fold_end in folded_boundaries:
            if dis_start >= fold_start and dis_end <= fold_end:
                is_contained = True
                break
        
        # Only keep disordered regions that are NOT contained
        if not is_contained:
            filtered_disordered.append([dis_start, dis_end])
    
    return filtered_disordered

def remove_short_disordered_regions(disordered_boundaries, threshold=30):
    """Remove ALL disordered regions that are shorter than threshold."""
    if not disordered_boundaries:
        return disordered_boundaries
    
    filtered_disordered = []
    
    for dis_start, dis_end in disordered_boundaries:
        dis_length = dis_end - dis_start + 1
        
        # Keep only disordered regions >= threshold length
        if dis_length >= threshold:
            filtered_disordered.append([dis_start, dis_end])
    
    return filtered_disordered

print("Reading data and applying filters...")
proteins_with_linkers = []

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        idx = int(row['idx'])
        name = row['name']
        seq_length = int(row['sequence_length'])
        
        # Parse the boundaries
        folded_boundaries = ast.literal_eval(row['folded_boundaries'])
        disordered_boundaries = ast.literal_eval(row['disordered_boundaries'])
        
        # Step 1: Merge folded domains with short linkers
        merged_folded_boundaries = merge_close_folded_domains(folded_boundaries, LINKER_THRESHOLD)
        
        # Step 2: Remove disordered regions that are now contained within merged folded domains
        filtered_disordered_boundaries = remove_contained_disordered_regions(
            merged_folded_boundaries, disordered_boundaries
        )
        
        # Step 3: Remove ALL remaining short disordered regions (< 30 AA)
        filtered_disordered_boundaries = remove_short_disordered_regions(
            filtered_disordered_boundaries, LINKER_THRESHOLD
        )
        
        # Step 4: Update counts based on filtered boundaries
        num_folded_merged = len(merged_folded_boundaries)
        num_disordered_filtered = len(filtered_disordered_boundaries)
        
        # Process any protein with 2+ folded domains and disordered regions AFTER filtering
        # (need at least 2 FDs to have a linker between them)
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

print(f"Found {len(proteins_with_linkers):,} proteins with 2+ folded domains and disordered regions (after filtering)")

def get_linker_lengths(folded_bounds, disordered_bounds):
    """
    Calculate lengths of disordered regions (linkers) between folded domains.
    Returns list of tuples: (start, end, length)
    """
    linkers = []
    
    # Sort both by start position
    folded_sorted = sorted(folded_bounds, key=lambda x: x[0])
    disordered_sorted = sorted(disordered_bounds, key=lambda x: x[0])
    
    # For each disordered region, check if it's between two folded domains
    for d_start, d_end in disordered_sorted:
        # Check if there's a folded domain before and after this disordered region
        has_folded_before = any(f_end < d_start for f_start, f_end in folded_sorted)
        has_folded_after = any(f_start > d_end for f_start, f_end in folded_sorted)
        
        if has_folded_before and has_folded_after:
            # This is a linker between folded domains
            linker_length = d_end - d_start + 1
            linkers.append((d_start, d_end, linker_length))
    
    return linkers

# Collect all linker data
print("\nAnalyzing linker regions between folded domains...")
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

# Calculate statistics
print(f"\n=== Linker Statistics (After Filtering, All Proteins) ===")
print(f"Total proteins analyzed: {len(proteins_with_linkers):,}")
print(f"Total linkers found: {len(all_linker_lengths):,}")
if all_linker_lengths:
    print(f"Mean linker length: {np.mean(all_linker_lengths):.1f} residues")
    print(f"Median linker length: {np.median(all_linker_lengths):.1f} residues")
    print(f"Min linker length: {min(all_linker_lengths)} residues")
    print(f"Max linker length: {max(all_linker_lengths)} residues")
    print(f"Std dev: {np.std(all_linker_lengths):.1f} residues")
    
    # Distribution by number of folded domains
    print(f"\n=== Proteins by Number of Folded Domains ===")
    fd_counts = {}
    for protein in proteins_with_linkers:
        nfd = protein['num_folded']
        fd_counts[nfd] = fd_counts.get(nfd, 0) + 1
    for nfd in sorted(fd_counts.keys()):
        print(f"  {nfd} FDs: {fd_counts[nfd]:,} proteins")
else:
    print("No linkers found!")

# Save linker data to CSV
output_csv = 'linker_lengths_all_proteins_filtered.csv'
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Protein_Index', 'Protein_Name', 'Num_Folded_Domains', 
                     'Linker_Start', 'Linker_End', 'Linker_Length'])
    
    for detail in linker_details:
        writer.writerow([
            detail['protein_idx'],
            detail['protein_name'],
            detail['num_folded'],
            detail['linker_start'],
            detail['linker_end'],
            detail['linker_length']
        ])

print(f"\nLinker data saved to '{output_csv}'")

# Create distribution plot
if all_linker_lengths:
    fig, ax = plt.subplots(figsize=(4, 4))

    # Create histogram with 50 bins between 0 and 500
    n, bins, patches = ax.hist(all_linker_lengths, bins=50, range=(0, 500), color='#2ecc71', 
                                edgecolor='black', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Linker length (residues)', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, max(n) * 1.1)  # Dynamic y-axis based on max frequency
    ax.tick_params(axis='both', labelsize=18)

    # Add grid for easier reading
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add text annotation
    ax.text(0.98, 0.98, f'n = {len(all_linker_lengths):,} linkers\n≥30 AA', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()

    # Save plot
    output_plot = 'linker_length_distribution_all_proteins_filtered.svg'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to '{output_plot}'")
    plt.close()
else:
    print("Cannot create plot: no linker data available")