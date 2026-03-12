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


def solve_weight(target, mean_unbound, mean_bound):

    w = (target - mean_bound) / (mean_unbound - mean_bound)
    return np.clip(w, 0, 1)

mods   = ["full", "full_bound"]
trials = list(range(1, 11))

variants = {
    "Dsk2 FL":   {"exp_rg": 37.2, "err": 0.15},
    "Dsk2 I45A": {"exp_rg": 39.1, "err": 0.2},
}

colors_open   = "blue"
colors_closed = "purple"
face_open     = mcolors.to_rgba(colors_open,   alpha=0.2)
face_closed   = mcolors.to_rgba(colors_closed, alpha=0.2)
edge_open     = mcolors.to_rgba(colors_open,   alpha=1.0)
edge_closed   = mcolors.to_rgba(colors_closed, alpha=1.0)

### load data 
avg_rgs = {}
for mod in mods:
    trial_means = [] 
    for trial in trials:
        Rg = np.load(f"../../Data/old_pipeline/CALVADOS3COM_2.0_MD_gpu_trial{trial}_Dsk2_{mod}/Dsk2_{mod}/0/Rg_traj.npy")
        mean_rg_tri = 10 * np.mean(Rg)
        trial_means.append(mean_rg_tri)
    
    avg_rgs[mod] = np.mean(trial_means)

mean_unbound = avg_rgs[mods[0]]
mean_bound   = avg_rgs[mods[1]]

# calc weights and errors 
weights_open, weights_closed = [], []
err_open_lo, err_open_hi     = [], []      

for var_name, config in variants.items():
    rg, err = config["exp_rg"], config["err"]

    w_mid  = solve_weight(rg, mean_unbound, mean_bound)
    w_hi   = solve_weight(rg + err, mean_unbound, mean_bound)
    w_lo   = solve_weight(rg - err, mean_unbound, mean_bound)

    #get percentage values 
    wo = w_mid * 100
    wc = (1 - w_mid) * 100
    weights_open.append(wo)
    weights_closed.append(wc)


    #get error of percentage 
    err_open_lo.append(wo - (w_lo  * 100))
    err_open_hi.append((w_hi * 100) - wo)   


## Plotting arrea: 
var_names = list(variants.keys())
x         = np.arange(len(var_names))
width     = 0.5

fig, ax = plt.subplots(figsize=(4, 4))

bars_open = ax.bar(x, weights_open, width, label="Open",
                   facecolor=face_open, edgecolor=edge_open, linewidth=1.5)

bars_closed = ax.bar(x, weights_closed, width, bottom=weights_open, label="Closed",
                     facecolor=face_closed, edgecolor=edge_closed, linewidth=1.5)

# Error bars on the boundary between open and closed (= top of open bar)
ax.errorbar(x, weights_open,
            yerr=[err_open_lo, err_open_hi],
            fmt='none', color='k', capsize=5, capthick=1.25, linewidth=1.25,
            zorder=5)

# Percentage labels
for i in range(len(variants)):
    wo, wc = weights_open[i], weights_closed[i]

    ax.text(x[i], wo / 2,      f"{wo:.0f}%", ha='center', va='center',
            fontsize=12, color='white')

    ax.text(x[i], wo + wc / 2, f"{wc:.0f}%", ha='center', va='center',
            fontsize=12, color='white')

ax.set_xticks(x)
ax.set_xticklabels(var_names, fontsize=14)
ax.set_ylabel('Population (%)', fontsize=14)
ax.set_xlim(-0.5, len(var_names) - 0.5)
ax.set_ylim(0, 102)
ax.set_yticks([0, 25, 50, 75, 100])
ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.savefig('Dsk2_weighted_bar.svg', dpi=300, bbox_inches='tight')
plt.close()
