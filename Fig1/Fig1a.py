import csv
import matplotlib.pyplot as plt
import ast

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

folded_proteins = []
idp_proteins = []
mixed_proteins = []

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
            # Disordered region is contained if it starts at or after fold_start
            # and ends at or before fold_end
            if dis_start >= fold_start and dis_end <= fold_end:
                is_contained = True
                break
        
        # Only keep disordered regions that are NOT contained
        if not is_contained:
            filtered_disordered.append([dis_start, dis_end])
    
    return filtered_disordered

def remove_short_disordered_regions(disordered_boundaries, threshold=30):
    """Remove ALL disordered regions that are shorter than threshold.
    This includes N-terminal, C-terminal, and any remaining internal short regions."""
    if not disordered_boundaries:
        return disordered_boundaries
    
    filtered_disordered = []
    
    for dis_start, dis_end in disordered_boundaries:
        dis_length = dis_end - dis_start + 1
        
        # Keep only disordered regions >= threshold length
        if dis_length >= threshold:
            filtered_disordered.append([dis_start, dis_end])
    
    return filtered_disordered

# Count total proteins in input file
total_input_proteins = 0
with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for _ in reader:
        total_input_proteins += 1

print(f"Processing {total_input_proteins:,} proteins from {input_file}...")

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        idx = int(row['idx'])
        name = row['name']
        seq_length = int(row['sequence_length'])
        
        # Parse the lists (they're stored as strings)
        folded_boundaries = ast.literal_eval(row['folded_boundaries'])
        disordered_boundaries = ast.literal_eval(row['disordered_boundaries'])
        disordered_domains = ast.literal_eval(row['disordered_domains'])
        
        # Step 1: Merge folded domains with short linkers
        merged_folded_boundaries = merge_close_folded_domains(folded_boundaries, LINKER_THRESHOLD)
        
        # Step 2: Remove disordered regions that are now contained within merged folded domains
        filtered_disordered_boundaries = remove_contained_disordered_regions(
            merged_folded_boundaries, disordered_boundaries
        )
        
        # Step 3: Remove ALL remaining short disordered regions (< 30 AA)
        # This catches N-terminal, C-terminal, and any other short disordered regions
        filtered_disordered_boundaries = remove_short_disordered_regions(
            filtered_disordered_boundaries, LINKER_THRESHOLD
        )
        
        # Step 4: Update counts based on merged and filtered boundaries
        num_folded_merged = len(merged_folded_boundaries)
        num_disordered_filtered = len(filtered_disordered_boundaries)
        
        # Calculate fraction disordered (using filtered disordered regions)
        total_disordered = sum(
            (d[1] - d[0] + 1) for d in filtered_disordered_boundaries
        )
        fraction_disordered = total_disordered / seq_length if seq_length > 0 else 0

        protein_data = {
            'idx': idx,
            'name': name,
            'length': seq_length,
            'num_folded': num_folded_merged,
            'num_disordered': num_disordered_filtered,
            'fraction_disordered': fraction_disordered
        }

        # Classify based on filtered counts
        if num_folded_merged == 0:
            # No folded domains → IDP
            idp_proteins.append(protein_data)
        elif num_disordered_filtered == 0:
            # No disordered domains (after filtering) → Folded
            folded_proteins.append(protein_data)
        else:
            # Has both → Mixed
            mixed_proteins.append(protein_data)

# Calculate overall statistics
total_proteins = len(folded_proteins) + len(idp_proteins) + len(mixed_proteins)
pct_folded = (len(folded_proteins) / total_proteins) * 100
pct_idp = (len(idp_proteins) / total_proteins) * 100
pct_mixed = (len(mixed_proteins) / total_proteins) * 100

print(f"\n=== Human Proteome Domain Architecture Analysis ===")
print(f"Input proteins: {total_input_proteins:,}")
print(f"Classified proteins: {total_proteins:,}")

if total_proteins != total_input_proteins:
    proteins_lost = total_input_proteins - total_proteins
    print(f"WARNING: {proteins_lost:,} proteins were removed during filtering!")
else:
    print(f"✓ All proteins classified successfully")

print(f"\nFolded: {len(folded_proteins):,} ({pct_folded:.1f}%)")
print(f"IDP: {len(idp_proteins):,} ({pct_idp:.1f}%)")
print(f"Mixed: {len(mixed_proteins):,} ({pct_mixed:.1f}%)")

# Save summary statistics
with open('proteome_classification_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Category', 'Count', 'Percentage'])
    writer.writerow(['Mixed', len(mixed_proteins), f"{pct_mixed:.2f}"])
    writer.writerow(['Folded', len(folded_proteins), f"{pct_folded:.2f}"])
    writer.writerow(['IDP', len(idp_proteins), f"{pct_idp:.2f}"])
    writer.writerow(['Total', total_proteins, "100.00"])

print("\nSummary saved to 'proteome_classification_summary.csv'")

# Create plot
fig, ax = plt.subplots(figsize=(4, 4))

categories = ['Mixed', 'Folded', 'IDP']
counts = [len(mixed_proteins), len(folded_proteins), len(idp_proteins)]
percentages = [pct_mixed, pct_folded, pct_idp]
colors = ['#4472C4', '#ED7D31', "#3EDB5B"]

bars = ax.bar(categories, percentages, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('% human proteome', fontsize=18)
ax.set_ylim(0, 100)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

# Add percentage labels on bars
for bar, pct in zip(bars, percentages):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('proteome_classification_merged.svg', dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved to 'proteome_classification_merged.svg'")