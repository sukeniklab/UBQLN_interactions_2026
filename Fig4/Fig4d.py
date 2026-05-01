import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['font.sans-serif'] = 'Arial'


variants = {
   
    "FL": {"mods": ["full", "full_bound"],
             "weight_open": 5.8,  
             "err_lo": 1.0,        
             "err_hi": 1.0},        

    "dTH1": {"mods": ["deltaTH1", "deltaTH1_bound"],
             "weight_open": 13.2,   
             "err_lo": 1.0,         
             "err_hi": 1.0},      

    "dTH2": {"mods": ["deltaTH2", "deltaTH2_bound"],
             "weight_open": 22.6,  
             "err_lo": 1.2,         
             "err_hi": 1.2},        
    
    
    "dTH3": {"mods": ["deltaTH3", "deltaTH3_bound"],
             "weight_open": 38.7,   
             "err_lo": 0.4,        
             "err_hi": 0.4},       

    "I45A": {"mods": ["full", "full_bound"],
             "weight_open": 24.4,   
             "err_lo": 0.8,         
             "err_hi": 0.8},       


}

colors_open   = "blue"
colors_closed = "purple"
face_open     = mcolors.to_rgba(colors_open,   alpha=0.4)
face_closed   = mcolors.to_rgba(colors_closed, alpha=0.4)
edge_open     = mcolors.to_rgba(colors_open,   alpha=1.0)
edge_closed   = mcolors.to_rgba(colors_closed, alpha=1.0)

weights_open, weights_closed = [], []
err_open_lo, err_open_hi     = [], []

for var_name, config in variants.items():


    wo = float(config["weight_open"])
    weights_open.append(wo)
    weights_closed.append(100.0 - wo)
    err_open_lo.append(float(config.get("err_lo", 0.0)))
    err_open_hi.append(float(config.get("err_hi", 0.0)))
    continue                         

var_names = list(variants.keys())
x         = np.arange(len(var_names))
width     = 0.5

fig, ax = plt.subplots(figsize=(4, 4))

bars_open = ax.bar(x, weights_open, width, label="Open",
                   facecolor=face_open, edgecolor=edge_open, linewidth=1.5)

bars_closed = ax.bar(x, weights_closed, width, bottom=weights_open, label="Closed",
                     facecolor=face_closed, edgecolor=edge_closed, linewidth=1.5)

ax.errorbar(x, weights_open,
            yerr=[err_open_lo, err_open_hi],
            fmt='none', color='k', capsize=5, capthick=1.25, linewidth=1.25,
            zorder=5)

for i in range(len(variants)):
    wo, wc = weights_open[i], weights_closed[i]
    if wo > 6:
        ax.text(x[i], wo / 2,      f"{wo:.0f}%", ha='center', va='center',
                fontsize=10, color='k')
    if wc > 6:
        ax.text(x[i], wo + wc / 2, f"{wc:.0f}%", ha='center', va='center',
                fontsize=10, color='k')

ax.set_xticks(x)
ax.set_xticklabels(var_names, fontsize=14)
ax.set_ylabel('Population (%)', fontsize=14)
ax.set_xlim(-0.5, len(var_names) - 0.5)
ax.set_ylim(0, 102)
ax.set_yticks([0, 25, 50, 75, 100])
ax.tick_params(axis='y', labelsize=12)
ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('Fig4D.svg', dpi=300, bbox_inches='tight')
plt.close()