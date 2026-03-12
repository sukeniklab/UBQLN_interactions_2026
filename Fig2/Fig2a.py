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


def solve_weight(target, mean_unbound, mean_bound):
    w = (target - mean_bound) / (mean_unbound - mean_bound)
    return np.clip(w, 0, 1)


def make_weighted_ensemble(all_rg_data, optimal_weight, n_samples_per_trial, trials):
    histograms = []
    means = []
    for i in range(len(trials)):
        rg_unbound = all_rg_data[0][i]
        rg_bound   = all_rg_data[1][i]
        n_unbound  = int(optimal_weight * n_samples_per_trial)
        n_bound    = n_samples_per_trial - n_unbound
        sampled = np.concatenate([
            np.random.choice(rg_unbound, size=n_unbound, replace=True),
            np.random.choice(rg_bound,   size=n_bound,   replace=True),
        ])
        h, _ = np.histogram(sampled, bins=bins)
        histograms.append(h / h.sum())
        means.append(np.mean(sampled))
    return np.array(histograms), means


mods   = ["full", "full_bound"]
labels = ["Unbound Sim", "Bound Sim"]
trials = list(range(1, 11))

colors_base = ['blue', 'purple']    # unbound, bound
color_fl    = 'dimgray'             # FL weighted ensemble
color_i45a  = 'red'                 # I45A weighted ensemble

bins = np.arange(0, 85, 0.75)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

exp_rg_fl   = 35.56
exp_rg_i45a = 37.4

# Shared axis limits
XLIM      = (20, 80)
YLIM      = (0, 0.07)
YTICKS    = [0.0, 0.025, 0.05]
FS_LABEL  = 18
FS_LEGEND = 10

### load all data once
all_rg_data = {0: [], 1: []}
avg_rgs = []
n_samples_per_trial = None

for j in range(len(mods)):
    avg_rg = []
    for trial in trials:
        rg = 10 * np.load(f"../../Data/old_pipeline/CALVADOS3COM_2.0_MD_gpu_trial{trial}_Dsk2_{mods[j]}/0/Rg_traj.npy")
        all_rg_data[j].append(rg)
        avg_rg.append(np.mean(rg))
        n_samples_per_trial = len(rg)
    avg_rgs.append(np.mean(avg_rg))

# compute optimal weights
w_fl   = solve_weight(exp_rg_fl,   avg_rgs[0], avg_rgs[1])
w_i45a = solve_weight(exp_rg_i45a, avg_rgs[0], avg_rgs[1])

hist_fl,   means_fl   = make_weighted_ensemble(all_rg_data, w_fl,   n_samples_per_trial, trials)
hist_i45a, means_i45a = make_weighted_ensemble(all_rg_data, w_i45a, n_samples_per_trial, trials)

mean_fl = hist_fl.mean(axis=0)
std_fl = hist_fl.std(axis=0)

mean_i45a = hist_i45a.mean(axis=0)
std_i45a =  hist_i45a.std(axis=0)


sim_means = []
sim_stds = []
sim_mean_dists = []
sim_std_dists = []

for j in range(len(mods)):
    avg_rg, histo_rg = [], []
    for trial in trials:
        rg = 10 * np.load(
            f"../../Data/old_pipeline/CALVADOS3COM_2.0_MD_gpu_trial{trial}_Dsk2_{mods[j]}/0/Rg_traj.npy")
        h, _ = np.histogram(rg, bins=bins)
        histo_rg.append(h / h.sum())
        avg_rg.append(np.mean(rg))
    sim_means.append(np.mean(avg_rg))
    sim_stds.append(np.std(avg_rg))
    sim_mean_dists.append(np.mean(histo_rg, axis=0))
    sim_std_dists.append(np.std(histo_rg, axis=0))

fig, axes = plt.subplots(3, 1, figsize=(4, 12), gridspec_kw={'hspace': 0}, constrained_layout=False)

# Panel 1: raw unbound + bound
ax = axes[0]
for j in range(len(mods)):
    ax.plot(bin_centers, sim_mean_dists[j], color=colors_base[j], 
            label=f"{labels[j]} ({sim_means[j]:.1f} Å)", linewidth=1.5)
    ax.fill_between(bin_centers, sim_mean_dists[j] - sim_std_dists[j], 
                    sim_mean_dists[j] + sim_std_dists[j], 
                    color=colors_base[j], alpha=0.2, linewidth=0, zorder=50)
    ax.fill_between(bin_centers, sim_mean_dists[j], color=colors_base[j], alpha=0.1)

axes[0].text(48, 0.06, f"Open={sim_means[0]:.1f}±{sim_stds[0]:.1f} Å", color=colors_base[0], fontsize=12)
axes[0].text(48, 0.055, f"Closed={sim_means[1]:.1f}±{sim_stds[1]:.1f} Å", color=colors_base[1], fontsize=12)

# Add vertical lines for simulation average Rg
for j in range(len(mods)):
    ax.axvline(sim_means[j], color=colors_base[j], linestyle='dashed', linewidth=1.75,
               ymin=0, ymax=0.1/0.065)

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_yticks(YTICKS)
ax.set_ylabel('Probability', fontsize=FS_LABEL)
ax.tick_params(axis='both', labelsize=FS_LABEL)
ax.set_xticklabels([])

# Panel 2: FL weighted ensemble
ax = axes[1]
ax.plot(bin_centers, mean_fl, color=color_fl, 
        label=f"FL Weighted Ensemble ({np.mean(means_fl):.1f}±{np.std(means_fl):.1f} Å)", linewidth=2)
ax.fill_between(bin_centers, mean_fl - std_fl, mean_fl + std_fl, 
                color=color_fl, alpha=0.2, linewidth=0, zorder=50)
ax.fill_between(bin_centers, mean_fl, color=color_fl, alpha=0.1)

axes[1].text(53, 0.06, f"FL={np.mean(means_fl):.1f}±{np.std(means_fl):.1f} Å", color=color_fl, fontsize=12)
axes[1].axvline(36.56, color=color_fl, linestyle='dashed', linewidth=1.5, label="Exp. FL = 36.6 Å")

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_yticks(YTICKS)
ax.set_ylabel('Probability', fontsize=FS_LABEL)
ax.tick_params(axis='both', labelsize=FS_LABEL)
ax.set_xticklabels([])

# Panel 3: I45A weighted ensemble
ax = axes[2]
ax.plot(bin_centers, mean_i45a, color=color_i45a, 
        label=f"I45A Weighted Ensemble ({np.mean(means_i45a):.1f}±{np.std(means_i45a):.1f} Å)", linewidth=2)
ax.fill_between(bin_centers, mean_i45a - std_i45a, mean_i45a + std_i45a, 
                color=color_i45a, alpha=0.2, linewidth=0, zorder=50)
ax.fill_between(bin_centers, mean_i45a, color=color_i45a, alpha=0.1)

axes[2].axvline(38.53, color=color_i45a, linestyle='dashed', linewidth=1.5, label="Exp. I45A = 38.5 Å")
axes[2].text(50, 0.06, f"I45A={np.mean(means_i45a):.1f}±{np.std(means_i45a):.1f} Å", color=color_i45a, fontsize=12)

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_yticks(YTICKS)
ax.set_ylabel('Probability', fontsize=FS_LABEL)
ax.tick_params(axis='both', labelsize=FS_LABEL)
ax.set_xlabel('$R_g$ ($\\AA$)', fontsize=FS_LABEL)

plt.subplots_adjust(hspace=0)
plt.savefig('Dsk2_rg_stacked_panels.svg', dpi=300, bbox_inches='tight')
plt.close()