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
input_file = 'identify_disordered_regions/disorder_complete_metapredictv3_Dec17.csv'

folded_proteins = []
idp_proteins = []
mixed_proteins = []

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

total_input_proteins = 0
with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_input_proteins += 1

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        idx = int(row['idx'])
        name = row['name']
        
        folded_boundaries = ast.literal_eval(row['folded_boundaries'])
        disordered_boundaries = ast.literal_eval(row['disordered_boundaries'])
        disordered_domains = ast.literal_eval(row['disordered_domains'])
        
        # Merge folded domains with short linkers
        merged_folded_boundaries = merge_close_folded_domains(folded_boundaries, LINKER_THRESHOLD)
        
        # Remove idrs that are now within folded domains
        filtered_disordered_boundaries = remove_contained_disordered_regions(merged_folded_boundaries, disordered_boundaries)
        
        # Remove idrs < 30aa
        filtered_disordered_boundaries = remove_short_disordered_regions(filtered_disordered_boundaries, LINKER_THRESHOLD)
        
        # recount 
        num_folded_merged = len(merged_folded_boundaries)
        num_disordered_filtered = len(filtered_disordered_boundaries)
        

        protein_data = {
            'idx': idx,
            'name': name,
            'num_folded': num_folded_merged,
            'num_disordered': num_disordered_filtered,
        }

        # Classify 
        if num_folded_merged == 0: #all IDR
           
            idp_proteins.append(protein_data)
        elif num_disordered_filtered == 0: #folded 
            folded_proteins.append(protein_data)
        else: #mixed 
            mixed_proteins.append(protein_data)


total_proteins = len(folded_proteins) + len(idp_proteins) + len(mixed_proteins)
per_folded = (len(folded_proteins) / total_proteins) * 100
per_idp = (len(idp_proteins) / total_proteins) * 100
per_mixed = (len(mixed_proteins) / total_proteins) * 100

print(f"\nFolded: {len(folded_proteins):,} ({per_folded:.1f}%)")
print(f"IDP: {len(idp_proteins):,} ({per_idp:.1f}%)")
print(f"Mixed: {len(mixed_proteins):,} ({per_mixed:.1f}%)")

fig, ax = plt.subplots(figsize=(4, 4))

categories = ['Mixed', 'Folded', 'IDP']
counts = [len(mixed_proteins), len(folded_proteins), len(idp_proteins)]
percentages = [per_mixed, per_folded, per_idp]
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
plt.savefig('Fig1a.svg', dpi=300, bbox_inches='tight')
plt.close()

