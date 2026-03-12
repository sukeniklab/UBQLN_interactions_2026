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

def remove_short_terminal_disordered_regions(disordered_boundaries, threshold=30):
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
seen_proteins = set()  # Track unique protein names

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        idx = int(row['idx'])
        name = row['name']
        seq_length = int(row['sequence_length'])
        
        # Parse the boundaries first
        folded_boundaries = ast.literal_eval(row['folded_boundaries'])
        disordered_boundaries = ast.literal_eval(row['disordered_boundaries'])
        
        # Step 1: Merge folded domains with short linkers
        merged_folded_boundaries = merge_close_folded_domains(folded_boundaries, LINKER_THRESHOLD)
        
        # Step 2: Remove disordered regions that are now contained within merged folded domains
        filtered_disordered_boundaries = remove_contained_disordered_regions(
            merged_folded_boundaries, disordered_boundaries
        )
        
        # Step 3: Update counts based on filtered boundaries
        num_folded_merged = len(merged_folded_boundaries)
        num_disordered_filtered = len(filtered_disordered_boundaries)
        
        # Only process mixed proteins (has both folded and disordered domains after merging and filtering)
        if num_folded_merged > 0 and num_disordered_filtered > 0:
            # Skip duplicates
            if name in seen_proteins:
                continue
            seen_proteins.add(name)
            
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

print(f"Analyzing {len(mixed_proteins):,} unique mixed proteins...")

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
    """Group patterns with their reverses together using consistent alphabetical ordering"""
    parts = pattern.split('-')
    reversed_parts = parts[::-1]
    reversed_pattern = '-'.join(reversed_parts)
    
    # Always use the alphabetically smaller pattern as the canonical form
    # This ensures consistent grouping: F-D and D-F both map to D-F
    return min(pattern, reversed_pattern)

architecture_patterns = {}
architecture_categories = {}
protein_pattern_list = []  # Store each protein with its pattern

for protein in mixed_proteins:
    pattern = get_architecture_pattern(
        protein['folded_boundaries'], 
        protein['disordered_boundaries']
    )
    
    # Store protein with its pattern
    category = classify_architecture(pattern)
    protein_pattern_list.append({
        'name': protein['name'],
        'pattern': pattern,
        'grouped_pattern': category,
        'num_folded': protein['num_folded'],
        'num_disordered': protein['num_disordered']
    })
    
    # Count individual patterns
    if pattern not in architecture_patterns:
        architecture_patterns[pattern] = 0
    architecture_patterns[pattern] += 1
    
    # Count categories
    if category not in architecture_categories:
        architecture_categories[category] = 0
    architecture_categories[category] += 1

# Save each protein with its pattern
with open('protein_architecture_list.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Protein_Name', 'Architecture_Pattern', 'Grouped_Pattern', 'Num_Folded_Domains', 'Num_Disordered_Regions'])
    
    for entry in protein_pattern_list:
        writer.writerow([entry['name'], entry['pattern'], entry['grouped_pattern'], 
                        entry['num_folded'], entry['num_disordered']])

print(f"Protein architecture list saved to 'protein_architecture_list.csv'")

# Save individual architecture patterns with grouping info
with open('architecture_patterns_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Architecture_Pattern', 'Count', 'Percentage_of_Mixed', 'Grouped_As', 'Note'])
    
    for pattern, count in sorted(architecture_patterns.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(mixed_proteins)) * 100
        grouped = classify_architecture(pattern)
        
        # Check if this pattern is the same as its grouped form
        if pattern == grouped:
            note = "Canonical form"
        else:
            note = f"Grouped with {grouped}"
        
        writer.writerow([pattern, count, f"{pct:.2f}", grouped, note])

print(f"Architecture patterns saved to 'architecture_patterns_summary.csv'")

# Save grouped patterns summary showing which individual patterns contribute
with open('architecture_grouped_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Grouped_Pattern', 'Total_Count', 'Percentage_of_Mixed', 'Contributing_Patterns'])
    
    for grouped_pattern, total_count in sorted(architecture_categories.items(), key=lambda x: x[1], reverse=True):
        pct = (total_count / len(mixed_proteins)) * 100
        
        # Find all individual patterns that map to this grouped pattern
        contributing = []
        for indiv_pattern in architecture_patterns.keys():
            if classify_architecture(indiv_pattern) == grouped_pattern:
                contributing.append(f"{indiv_pattern}({architecture_patterns[indiv_pattern]})")
        
        contributing_str = "; ".join(contributing)
        writer.writerow([grouped_pattern, total_count, f"{pct:.2f}", contributing_str])

print(f"Grouped patterns summary saved to 'architecture_grouped_summary.csv'")

# Verify counts
total_individual_patterns = sum(architecture_patterns.values())
total_grouped_patterns = sum(architecture_categories.values())
print(f"\n=== Count Verification ===")
print(f"Total mixed proteins: {len(mixed_proteins):,}")
print(f"Sum of individual pattern counts: {total_individual_patterns:,}")
print(f"Sum of grouped pattern counts: {total_grouped_patterns:,}")
print(f"Unique individual patterns: {len(architecture_patterns)}")
print(f"Unique grouped patterns: {len(architecture_categories)}")
if total_individual_patterns == len(mixed_proteins):
    print("✓ Individual pattern counts match!")
else:
    print(f"✗ WARNING: Mismatch of {total_individual_patterns - len(mixed_proteins)} proteins")

# Print top grouped patterns
print("\n=== Top 20 Grouped Architecture Patterns ===")
top_20_patterns = sorted(architecture_categories.items(), key=lambda x: x[1], reverse=True)[:20]

for pattern, count in top_20_patterns:
    pct = (count / len(mixed_proteins)) * 100
    print(f"{pattern}: {count:,} proteins ({pct:.1f}%)")

print(f"\nTotal unique grouped patterns: {len(architecture_categories)}")

# Show some examples of grouped patterns
print("\n=== Pattern Grouping Examples ===")
print("Patterns are grouped with their reverses (directionality ignored):")
example_pairs = []
for pattern in architecture_patterns.keys():
    grouped = classify_architecture(pattern)
    reversed_pattern = '-'.join(pattern.split('-')[::-1])
    if pattern != grouped and reversed_pattern in architecture_patterns:
        pair = tuple(sorted([pattern, reversed_pattern]))
        if pair not in example_pairs:
            example_pairs.append(pair)
            if len(example_pairs) <= 5:  # Show first 5 examples
                print(f"  '{pair[0]}' and '{pair[1]}' → grouped as '{grouped}'")

# Get ALL patterns for plotting (sorted by count)
all_patterns_plot = sorted(architecture_categories.items(), key=lambda x: x[1], reverse=True)
print(f"\nPlotting all {len(all_patterns_plot)} grouped patterns...")
print("(Patterns like F-D and D-F are combined in the plot)")

# Dynamically adjust figure width based on number of patterns
# Use at least 0.5 inches per bar, with a minimum of 16 inches for more spacing
fig_width = max(16, len(all_patterns_plot) * 0.5)
fig, ax = plt.subplots(figsize=(fig_width, 6))

categories = [p[0] for p in all_patterns_plot]
category_percentages = [(p[1] / len(mixed_proteins)) * 100 for p in all_patterns_plot]
category_counts = [p[1] for p in all_patterns_plot]

# Use tab20 colors, cycling if we have more than 20 patterns
colors = [plt.cm.tab20(i % 20) for i in range(len(categories))]

x_positions = range(len(categories))
bars = ax.bar(x_positions, category_percentages, 
              color=colors, edgecolor='black', linewidth=1.5, width=0.575)

ax.set_xticks(x_positions)
ax.set_xticklabels(categories, fontsize=14, rotation=45, ha='right')
ax.set_ylabel('% of mixed proteins', fontsize=18)
# Add note about grouping - move to right side
ax.text(0.98, 0.98, '*Reverse patterns grouped together', 
        transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right')
ax.set_ylim(0, max(category_percentages) * 1.15)
# Tighten x-axis limits to reduce space on left/right edges
ax.set_xlim(-0.7, len(categories) - 0.5)
ax.tick_params(axis='y', labelsize=18)

# Add count labels on bars
for bar, count in zip(bars, category_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('architecture_categories_corrected.svg', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to 'architecture_categories_corrected.svg' with all {len(all_patterns_plot)} grouped patterns")
print("(Reverse patterns like F-D and D-F are combined in the visualization)")
plt.close()

# Also print top individual patterns with grouping info
print("\n=== Top 15 Individual Patterns ===")
print("(Note: Patterns like F-D and D-F are grouped together)")
for pattern, count in sorted(architecture_patterns.items(), key=lambda x: x[1], reverse=True)[:15]:
    pct = (count / len(mixed_proteins)) * 100
    grouped = classify_architecture(pattern)
    
    if pattern == grouped:
        print(f"{pattern}: {count:,} proteins ({pct:.1f}%) [canonical form]")
    else:
        print(f"{pattern}: {count:,} proteins ({pct:.1f}%) → grouped as '{grouped}'")