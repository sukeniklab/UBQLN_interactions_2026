import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.sans-serif'] = 'Arial'

mods   = ["full", "full_bound"]
labels = ["Unbound Sim", "Bound Sim"]
trials = list(range(1, 11))

colors_base = ['blue', 'purple']

bins = np.arange(0, 85, 0.75)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

exp_rg_fl    = 37.2
ex_rg_fl_err = 0.5

XLIM     = (20, 80)
YLIM     = (0, 0.075)
YTICKS   = [0.0, 0.025, 0.05, 0.075]
FS_LABEL = 18

### load all data once
all_rg_data = {0: [], 1: []}
n_samples_per_trial = None

for j in range(len(mods)):
    for trial in trials:
        rg = 10 * np.load(f"raw_rg_data/Dsk2_{mods[j]}_trial{trial}.npy")
        all_rg_data[j].append(rg)
        n_samples_per_trial = len(rg)

### compute histograms — h/h.sum() (probability mass), std of per-trial means
sim_mean_dists = []
sim_std_dists  = []
sim_means      = []
sim_stds       = []

for j in range(len(mods)):
    trial_histograms = []
    trial_means      = []

    for trial_rg in all_rg_data[j]:
        h, _ = np.histogram(trial_rg, bins=bins)
        trial_histograms.append(h / h.sum())        # probability mass, not density
        trial_means.append(np.mean(trial_rg))       # per-trial mean

    trial_histograms = np.array(trial_histograms)
    sim_mean_dists.append(np.mean(trial_histograms, axis=0))
    sim_std_dists.append(np.std(trial_histograms, axis=0))
    sim_means.append(np.mean(trial_means))
    sim_stds.append(np.std(trial_means))            # std of per-trial means

### plot
fig, ax = plt.subplots(1, 1, figsize=(4, 4), gridspec_kw={'hspace': 0}, constrained_layout=False)

for j in range(len(mods)):
    ax.plot(bin_centers, sim_mean_dists[j], color=colors_base[j],
            label=f"{labels[j]} ({sim_means[j]:.1f} Å)", linewidth=1.5)
    ax.fill_between(bin_centers,
                    sim_mean_dists[j] - sim_std_dists[j],
                    sim_mean_dists[j] + sim_std_dists[j],
                    color=colors_base[j], alpha=0.2, linewidth=0, zorder=50)
    ax.fill_between(bin_centers, sim_mean_dists[j], color=colors_base[j], alpha=0.1)
    # vertical dashed line at simulation mean Rg
    ax.axvline(sim_means[j], color=colors_base[j], linestyle='dashed', linewidth=1.75,
               ymin=0, ymax=0.1/0.065)

# experimental Rg vertical line
ax.axvline(exp_rg_fl, color='black', linestyle='dashed', linewidth=1.75)

ax.text(48, 0.060, f"Open={sim_means[0]:.1f}±{sim_stds[0]:.1f} Å",  color=colors_base[0], fontsize=12)
ax.text(48, 0.055, f"Closed={sim_means[1]:.1f}±{sim_stds[1]:.1f} Å", color=colors_base[1], fontsize=12)
ax.text(48, 0.050, f"Exp. FL={exp_rg_fl}±{ex_rg_fl_err} Å",          color='k',             fontsize=12)

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_yticks(YTICKS)
ax.set_ylabel('Probability', fontsize=FS_LABEL)
ax.set_xlabel('$R_g$ ($\\AA$)', fontsize=FS_LABEL)
ax.tick_params(axis='both', labelsize=FS_LABEL)

plt.subplots_adjust(hspace=0)
plt.savefig('Fig2a.svg', dpi=300, bbox_inches='tight')
plt.close()