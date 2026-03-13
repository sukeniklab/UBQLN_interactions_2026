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

def get_architecture_pattern(folded_bounds, disordered_bounds):
    all_regions = []
    for f in folded_bounds:
        all_regions.append(('F', f[0]))
    for d in disordered_bounds:
        all_regions.append(('D', d[0]))
    
    all_regions.sort(key=lambda x: x[1])
    
    pattern = '-'.join([r[0] for r in all_regions])
    return pattern

def classify_architecture(pattern):
    parts = pattern.split('-')
    reversed_parts = parts[::-1]
    reversed_pattern = '-'.join(reversed_parts)
    
    ##check for reverse pattern
    return min(pattern, reversed_pattern)


mixed_proteins = []
seen_proteins = set()  # Track unique architectures 

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        idx = int(row['idx'])
        name = row['name']
        
        folded_boundaries = ast.literal_eval(row['folded_boundaries'])
        disordered_boundaries = ast.literal_eval(row['disordered_boundaries'])
        
        # Merge folded domains with short linkers
        merged_folded_boundaries = merge_close_folded_domains(folded_boundaries, LINKER_THRESHOLD)
        
        # Remove idrs that are now within folded domains
        filtered_disordered_boundaries = remove_contained_disordered_regions(merged_folded_boundaries, disordered_boundaries)
        
        
        num_folded_merged = len(merged_folded_boundaries)
        num_disordered_filtered = len(filtered_disordered_boundaries)
        
        if num_folded_merged > 0 and num_disordered_filtered > 0:
            if name in seen_proteins:
                continue
            seen_proteins.add(name)
            
            protein_data = {
                'idx': idx,
                'name': name,
                'num_folded': num_folded_merged,
                'num_disordered': num_disordered_filtered,
                'folded_boundaries': merged_folded_boundaries,
                'disordered_boundaries': filtered_disordered_boundaries
            }
            mixed_proteins.append(protein_data)



architecture_patterns = {}
architecture_categories = {}
protein_pattern_list = [] 

for protein in mixed_proteins:
    pattern = get_architecture_pattern(protein['folded_boundaries'], protein['disordered_boundaries'])
    
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


total_individual_patterns = sum(architecture_patterns.values())
total_grouped_patterns = sum(architecture_categories.values())


# Merge small groups into "Other" category
MIN_GROUP_SIZE = 10

# Separate patterns into major groups and "Other"
major_patterns = {}
other_count = 0
other_patterns = []

for pattern, count in architecture_categories.items():
    if count >= MIN_GROUP_SIZE:
        major_patterns[pattern] = count
    else:
        other_count += count
        other_patterns.append(f"{pattern}({count})")

# Add "Other" category if there are any small groups
all_patterns_plot = sorted(major_patterns.items(), key=lambda x: x[1], reverse=True)
if other_count > 0:
    all_patterns_plot.append(('Other', other_count))

fig_width = max(16, len(all_patterns_plot) * 0.5)
fig, ax = plt.subplots(figsize=(fig_width, 6))

categories = [p[0] for p in all_patterns_plot]
category_percentages = [(p[1] / len(mixed_proteins)) * 100 for p in all_patterns_plot]
category_counts = [p[1] for p in all_patterns_plot]


colors = []
for i, cat in enumerate(categories):
    if cat == 'Other':
        colors.append('#808080')  # Gray for "Other"
    else:
        colors.append(plt.cm.tab20(i % 20))

x_positions = range(len(categories))
bars = ax.bar(x_positions, category_percentages, 
              color=colors, edgecolor='black', linewidth=1.5, width=0.575)

ax.set_xticks(x_positions)
ax.set_xticklabels(categories, fontsize=14, rotation=45, ha='right')
ax.set_ylabel('% of mixed proteins', fontsize=18)
note_text = f'*Reverse patterns grouped together'
ax.text(0.98, 0.98, note_text, 
        transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right')
ax.set_ylim(0, max(category_percentages) * 1.15)
ax.set_xlim(-0.7, len(categories) - 0.5)
ax.tick_params(axis='y', labelsize=18)

# Add count labels on bars
for bar, count in zip(bars, category_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('Fig_SI1.svg', dpi=300, bbox_inches='tight')
plt.close()
