import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['svg.fonttype'] = 'none'

# ---- Parameters ----
original_domains = [[75,145], [223,325]]
STI1_locs = [[146,222]]

# Load data from CSV
df = pd.read_csv('Dsk2_STI1_Occupancy_Fold_Change.csv')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Split data into two domains
for i, (domain_start, domain_end) in enumerate(original_domains):
    domain_data = df[(df['residue_number'] >= domain_start) & 
                     (df['residue_number'] <= domain_end)].copy()
    
    residue_number = domain_data['residue_number'].values
    weighted_mean = domain_data['weighted_mean'].values
    weighted_std = domain_data['weighted_std'].values
    
    axes[i].set_xlim(domain_start, domain_end)
    
    axes[i].scatter(residue_number, weighted_mean, color='blue', s=10, zorder=100)
    axes[i].plot(residue_number, weighted_mean, color='blue', linewidth=0.8, zorder=100)
    axes[i].fill_between(residue_number, weighted_mean - weighted_std, weighted_mean + weighted_std, 
                        color='blue', alpha=0.15, zorder=50)
    axes[i].set_ylim(0, 200)
    axes[i].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# Mark TH regions
axes[0].axvspan(114, 134, color='orange', alpha=0.3)
axes[1].axvspan(279, 291, color='orange', alpha=0.3)
axes[1].axvspan(303, 313, color='orange', alpha=0.3)

fig.supylabel("STI1 Occupancy Fold Change ($P_{\mathrm{g}}$/$P_{\mathrm{g,EV}}$)", x=0.05)
fig.supxlabel("Residue Number", y=-0.02)

plt.savefig('Fig3b.svg', dpi=300, bbox_inches="tight")
plt.close()