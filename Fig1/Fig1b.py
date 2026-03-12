import csv
import matplotlib.pyplot as plt
import ast

# Apply same styling
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

mixed_proteins = []

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
        # This catches N-terminal, C-terminal, and any other short disordered regions
        filtered_disordered_boundaries = remove_short_disordered_regions(
            filtered_disordered_boundaries, LINKER_THRESHOLD
        )
        
        # Step 4: Update counts based on merged and filtered boundaries
        num_folded_merged = len(merged_folded_boundaries)
        num_disordered_filtered = len(filtered_disordered_boundaries)
        
        # Only process mixed proteins (has both folded and disordered domains after filtering)
        if num_folded_merged > 0 and num_disordered_filtered > 0:
            protein_data = {
                'idx': idx,
                'name': name,
                'length': seq_length,
                'num_folded': num_folded_merged,
                'num_disordered': num_disordered_filtered,
                'folded_boundaries': merged_folded_boundaries,
                'disordered_boundaries': filtered_disordered_boundaries
            }
            mixed_proteins.append(protein_data)

print(f"Analyzing {len(mixed_proteins):,} mixed proteins...")

# Analyze architecture patterns
def get_architecture_pattern(folded_bounds, disordered_bounds):
    """Determine the linear arrangement of folded and disordered regions"""
    all_regions = []
    for f in folded_bounds:
        all_regions.append(('F', f[0]))
    for d in disordered_bounds:
        all_regions.append(('D', d[0]))
    
    # Sort by start position
    all_regions.sort(key=lambda x: x[1])
    
    # Create pattern string
    pattern = '-'.join([r[0] for r in all_regions])
    return pattern

def classify_architecture(pattern):
    """Classify pattern into architectural categories"""
    parts = pattern.split('-')
    num_folded = parts.count('F')
    
    if num_folded == 1:
        return "1 FD & IDR(s)"
    elif num_folded == 2:
        return "2 FDs & IDRs"
    else:  # 3+ folded domains
        return "3+ FDs & IDRs"

architecture_patterns = {}
architecture_categories = {}
protein_pattern_list = []  # Store protein name and pattern pairs

for protein in mixed_proteins:
    pattern = get_architecture_pattern(
        protein['folded_boundaries'], 
        protein['disordered_boundaries']
    )
    
    # Store protein name and pattern
    protein_pattern_list.append({
        'name': protein['name'],
        'pattern': pattern,
        'category': classify_architecture(pattern)
    })
    
    # Count individual patterns
    if pattern not in architecture_patterns:
        architecture_patterns[pattern] = 0
    architecture_patterns[pattern] += 1
    
    # Count categories
    category = classify_architecture(pattern)
    if category not in architecture_categories:
        architecture_categories[category] = 0
    architecture_categories[category] += 1

# Save individual architecture patterns
with open('architecture_patterns_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Architecture_Pattern', 'Count', 'Percentage_of_Mixed', 'Category'])
    
    for pattern, count in sorted(architecture_patterns.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(mixed_proteins)) * 100
        category = classify_architecture(pattern)
        writer.writerow([pattern, count, f"{pct:.2f}", category])

# Save protein names with their patterns
with open('protein_patterns.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Protein_Name', 'Pattern', 'Category'])
    
    for protein_data in protein_pattern_list:
        writer.writerow([protein_data['name'], protein_data['pattern'], protein_data['category']])

# Define category order
category_order = [
    "1 FD & IDR(s)",
    "2 FDs & IDRs",
    "3+ FDs & IDRs"
]

# Save category summary
with open('architecture_categories_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Category', 'Count', 'Percentage_of_Mixed'])
    
    for category in category_order:
        if category in architecture_categories:
            count = architecture_categories[category]
            pct = (count / len(mixed_proteins)) * 100
            writer.writerow([category, count, f"{pct:.2f}"])

print(f"Architecture patterns saved to 'architecture_patterns_summary.csv'")
print(f"Protein patterns saved to 'protein_patterns.csv'")
print(f"Category summary saved to 'architecture_categories_summary.csv'")

# Print categories
print("\n=== Architecture Categories ===")

for category in category_order:
    if category in architecture_categories:
        count = architecture_categories[category]
        pct = (count / len(mixed_proteins)) * 100
        print(f"{category}: {count:,} proteins ({pct:.1f}%)")

# Create category plot
fig, ax = plt.subplots(figsize=(4, 4))

categories = [cat for cat in category_order if cat in architecture_categories]
# Simplified x-axis labels
simplified_labels = ["1 FD", "2 FDs", "3+ FDs"]
category_percentages = [(architecture_categories[cat] / len(mixed_proteins)) * 100 
                        for cat in categories]

# Use distinct colors
colors = ["#ae34db", '#e74c3c', "#ecf312"]

bars = ax.bar(simplified_labels, category_percentages, 
              color=colors[:len(categories)], edgecolor='black', linewidth=1.5)

ax.set_ylabel('% of mixed proteins', fontsize=18)
ax.set_ylim(0, 100)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

# Add percentage labels on bars
for bar, pct, cat in zip(bars, category_percentages, categories):
    count = architecture_categories[cat]
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('architecture_categories.svg', dpi=300, bbox_inches='tight')
print("\nPlot saved to 'architecture_categories.svg'")
plt.close()

# Also print top individual patterns
print("\n=== Top 15 Individual Patterns ===")
for pattern, count in sorted(architecture_patterns.items(), key=lambda x: x[1], reverse=True)[:15]:
    pct = (count / len(mixed_proteins)) * 100
    category = classify_architecture(pattern)
    print(f"{pattern}: {count:,} proteins ({pct:.1f}%) - {category}")