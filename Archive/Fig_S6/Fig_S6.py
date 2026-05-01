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


def load_mod(mod, trials):
    arrays, trial_means = [], []
    for trial in trials:
        rg = 10 * np.load(f"raw_rg_data/{mod}_trial{trial}.npy")
        arrays.append(rg)
        trial_means.append(np.mean(rg))
    return arrays, np.mean(trial_means), np.std(trial_means)


def rg_distribution(arrays, bins):
    histos = []
    for rg in arrays:
        h, _ = np.histogram(rg, bins=bins)
        histos.append(h / h.sum())
    return np.mean(histos, axis=0), np.std(histos, axis=0)


def make_weighted_ensemble(arrays_unbound, arrays_bound, weight, bins):
    n = len(arrays_unbound[0])
    n_unbound = int(weight * n)
    n_bound   = n - n_unbound
    histos, means = [], []
    for ru, rb in zip(arrays_unbound, arrays_bound):
        ensemble = np.concatenate([
            np.random.choice(ru, size=n_unbound, replace=True),
            np.random.choice(rb, size=n_bound,   replace=True),
        ])
        h, _ = np.histogram(ensemble, bins=bins)
        histos.append(h / h.sum())
        means.append(np.mean(ensemble))
    return np.mean(histos, axis=0), np.std(histos, axis=0), np.mean(means), np.std(means)


trials = list(range(1, 11))

variants = {
    # "FL": {"mods": ["full", "full_bound"], "exp_rg": 37.2, "err": 0.5, "color": "grey"},

    "ΔHS1": {"mods": ["deltaTH1", "deltaTH1_bound"], "exp_rg": 36.2, "err": 0.2, "color": "green"},
    "ΔHS2": {"mods": ["deltaTH2", "deltaTH2_bound"], "exp_rg": 38.5, "err": 0.2, "color": "fuchsia"},
    "ΔHS3": {"mods": ["deltaTH3", "deltaTH3_bound"], "exp_rg": 41.1, "err": 0.2, "color": "orange"},
}

bins = np.arange(0, 85, 0.75)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

color_unbound = "blue"
color_bound   = "purple"

XLIM     = (20, 80)
YLIM     = (0, 0.07)
YTICKS   = [0.0, 0.025, 0.05]
FS_LABEL = 16

## Plotting area:
n_variants = len(variants)
fig, axes  = plt.subplots(n_variants, 1, figsize=(4, 4 * n_variants),
                           gridspec_kw={'hspace': 0},
                           constrained_layout=False)
if n_variants == 1:
    axes = [axes]

for ax, (var_name, config) in zip(axes, variants.items()):
    last = (var_name == list(variants.keys())[-1])
    
    mods = config["mods"]

    ### load data
    arr_u, mean_u, std_u = load_mod(mods[0], trials)
    arr_b, mean_b, std_b = load_mod(mods[1], trials)

    # Distributions
    dist_u_mean, dist_u_std = rg_distribution(arr_u, bins)
    dist_b_mean, dist_b_std = rg_distribution(arr_b, bins)

    # Weighted ensemble (central + ±err)
    w_mid = solve_weight(config["exp_rg"], mean_u, mean_b)
    w_hi  = solve_weight(config["exp_rg"] + config["err"], mean_u, mean_b)
    w_lo  = solve_weight(config["exp_rg"] - config["err"], mean_u, mean_b)

    ens_mean, ens_std, ens_rg, ens_rg_std = make_weighted_ensemble(arr_u, arr_b, w_mid, bins)

    # Plot distributions
    ax.plot(bin_centers, dist_u_mean, color=color_unbound, alpha=0.8, 
            label=f"Unbound ({mean_u:.1f}±{std_u:.1f} Å)", linewidth=1.5)
    ax.fill_between(bin_centers, dist_u_mean - dist_u_std, dist_u_mean + dist_u_std,
                    color=color_unbound, alpha=0.2, linewidth=0, zorder=50)
    ax.fill_between(bin_centers, dist_u_mean, color=color_unbound, alpha=0.1)
    
    ax.plot(bin_centers, dist_b_mean, color=color_bound, alpha=0.8, 
            label=f"Bound   ({mean_b:.1f}±{std_b:.1f} Å)", linewidth=1.5)
    ax.fill_between(bin_centers, dist_b_mean - dist_b_std, dist_b_mean + dist_b_std,
                    color=color_bound, alpha=0.2, linewidth=0, zorder=50)
    ax.fill_between(bin_centers, dist_b_mean, color=color_bound, alpha=0.1)
    

    ax.text(52, 0.06, f"Open={mean_u:.1f}±{std_u:.1f} Å",
            color=color_unbound, fontsize=12)
    ax.text(52, 0.055, f"Closed={mean_b:.1f}±{std_b:.1f} Å",
            color=color_bound, fontsize=12)
    ax.text(50, 0.05, f"{var_name}={config['exp_rg']:.1f}±{config['err']:.1f} Å",
            color=config["color"], fontsize=12)

    # Vertical lines
    ax.axvline(mean_u, color=color_unbound, linestyle='dashed', linewidth=1.5,
               ymin=0, ymax=0.1 / 0.065)
    ax.axvline(mean_b, color=color_bound, linestyle='dashed', linewidth=1.5,
               ymin=0, ymax=0.1 / 0.065)
    ax.axvline(config["exp_rg"], color=config["color"], linestyle='dashed', linewidth=1.75,
               ymin=0, ymax=0.1 / 0.065,
               label=f"Exp. {var_name} = {config['exp_rg']:.1f}±{config['err']:.1f} Å")
    
    ax.axvspan(config["exp_rg"] - config["err"], config["exp_rg"] + config["err"],
               color=config["color"], alpha=0.3, linewidth=0, zorder=1)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    ax.set_ylabel('Probability', fontsize=FS_LABEL)
    ax.tick_params(axis='both', labelsize=FS_LABEL)
    if last:
        ax.set_xlabel('$R_g$ ($\\AA$)', fontsize=FS_LABEL)
    else:
        ax.set_xticklabels([])

plt.subplots_adjust(hspace=0)
plt.savefig('Fig_S6.svg', dpi=300, bbox_inches='tight')
plt.close()