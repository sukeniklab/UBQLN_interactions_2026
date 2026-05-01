import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
plt.rcParams['svg.fonttype'] = 'none'

# ── Helpers ───────────────────────────────────────────────────────────────────
plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['font.size'] = 14
plt.rcParams['font.sans-serif'] = 'Arial'

def shift_idr_range(idr_range, deletions_list):
    start, end = idr_range
    deletions_before = 0
    deletions_within = 0
    for del_start, del_end in deletions_list:
        if del_end < start:
            deletions_before += (del_end - del_start + 1)
        elif del_start <= end and del_end >= start:
            deletions_within += (min(del_end, end) - max(del_start, start) + 1)
    return [start - deletions_before, end - deletions_before - deletions_within]

def create_original_to_shifted_mapping(original_range, deletions_list):
    original_residues = np.arange(original_range[0], original_range[1] + 1)
    shifted_positions, original_positions = [], []
    for orig_res in original_residues:
        if any(del_start <= orig_res <= del_end for del_start, del_end in deletions_list):
            continue
        deletions_before = sum((del_end - del_start + 1)
                               for del_start, del_end in deletions_list if del_end < orig_res)
        shifted_positions.append(orig_res - deletions_before)
        original_positions.append(orig_res)
    return np.array(original_positions), np.array(shifted_positions)

def smooth_log_histogram(log_histogram, smooth):
    valid_mask = ~np.isnan(log_histogram)
    if np.sum(valid_mask) < 4:
        return log_histogram
    x_coords = np.arange(len(log_histogram))
    valid_x  = x_coords[valid_mask]
    valid_y  = log_histogram[valid_mask]
    spline   = UnivariateSpline(valid_x, valid_y, s=smooth, k=3)
    smoothed = np.full_like(log_histogram, np.nan)
    mn, mx   = valid_x.min(), valid_x.max()
    mask     = (x_coords >= mn) & (x_coords <= mx)
    smoothed[mask] = spline(x_coords[mask])
    return smoothed

def plot_segments(ax, dist_STI1, original_positions, original_section,
                  current_deletions, mean_y, std_y, color):
   
    def _draw(sx, sy, ss):
        if len(sx) == 0:
            return
        ax.scatter(sx, sy, color=color, s=10, zorder=100)
        ax.plot(sx, sy, color=color, linewidth=0.8, zorder=100)
        ax.fill_between(sx, sy - ss, sy + ss,
                        color=color, alpha=0.15, zorder=50, linewidth=0)

    breaks = np.where(np.diff(original_positions) > 1)[0]

    start_idx = 0
    for break_idx in breaks:
        end_idx = break_idx + 1  # exclusive, so last point before gap is included
        _draw(dist_STI1[start_idx:end_idx], mean_y[start_idx:end_idx], std_y[start_idx:end_idx])
        start_idx = end_idx

    _draw(dist_STI1[start_idx:], mean_y[start_idx:], std_y[start_idx:])


numReplicas    = 20
sim_time       = 70
print_interval = 0.01
snaps = numReplicas * int((sim_time / print_interval) - (0.05 * (sim_time / print_interval)))

trials           = list(range(1, 11))
original_domains = [[75, 145], [223, 325]]

deletions = {
    "Dsk2_full":     [],
    "Dsk2_deltaTH1": [[113, 133]],
    "Dsk2_deltaTH2": [[278, 290]],
    "Dsk2_deltaTH3": [[302, 312]],
}
directory = ["Dsk2_full", "Dsk2_deltaTH1", "Dsk2_deltaTH2", "Dsk2_deltaTH3"]
STI1_locs = [[146, 222]]

C_OPEN  = 'blue'
C_BOUND = 'purple'
C_EVRAW = 'black'
C_EVFIT = 'red'

fig, axes = plt.subplots(len(directory), 2,
                         figsize=(10, 4 * len(directory)),
                         sharex='col', sharey=True)

for j, th in enumerate(directory):
    current_deletions = deletions.get(th, [])

    for i, original_section in enumerate(original_domains):
        ax = axes[j, i]

        shifted_section = shift_idr_range(original_section, current_deletions)
        if shifted_section[0] > shifted_section[1]:
            print(f"Entire section {original_section} deleted in {th}")
            continue

        original_positions, _ = create_original_to_shifted_mapping(
            original_section, current_deletions)
        n = len(original_positions)

        all_open  = np.full((len(trials), n), np.nan)
        all_bound = np.full((len(trials), n), np.nan)
        all_evraw = np.full((len(trials), n), np.nan)
        all_evfit = np.full((len(trials), n), np.nan)

        smoothing_factor = 1 if i == 0 else 2.4

        for trial in trials:
            try:
                hist_open  = np.load(f"Histos/{th}_trial{trial}_innerVol_idr{shifted_section[0]}_{shifted_section[1]}.npy") / snaps
                hist_bound = np.load(f"Histos/{th}_bound_trial{trial}_innerVol_idr{shifted_section[0]}_{shifted_section[1]}.npy") / snaps
                hist_ev    = np.load(f"Histos/{th}_EV_trial{trial}_innerVol_idr{shifted_section[0]}_{shifted_section[1]}.npy") / (10 * snaps)

                with np.errstate(divide='ignore', invalid='ignore'):
                    log_open  = np.log10(hist_open)
                    log_bound = np.log10(hist_bound)
                    log_ev    = np.log10(hist_ev)

                log_open [~np.isfinite(log_open)]  = np.nan
                log_bound[~np.isfinite(log_bound)] = np.nan
                log_ev   [~np.isfinite(log_ev)]    = np.nan

                log_ev_fit = smooth_log_histogram(log_ev, smoothing_factor)

                all_open [trial - 1, :] = log_open
                all_bound[trial - 1, :] = log_bound
                all_evraw[trial - 1, :] = log_ev
                all_evfit[trial - 1, :] = log_ev_fit

            except FileNotFoundError as e:
                print(f"  Missing: {e}")
                continue

        mean_open  = np.nanmean(all_open,  axis=0)
        std_open   = np.nanstd(all_open,   axis=0)
        mean_bound = np.nanmean(all_bound, axis=0)
        std_bound  = np.nanstd(all_bound,  axis=0)
        mean_evraw = np.nanmean(all_evraw, axis=0)
        std_evraw  = np.nanstd(all_evraw,  axis=0)
        mean_evfit = np.nanmean(all_evfit, axis=0)
        std_evfit  = np.nanstd(all_evfit,  axis=0)

        dist_STI1 = original_positions - (STI1_locs[0][0] if i == 0 else STI1_locs[0][1])

        plot_segments(ax, dist_STI1, original_positions, original_section,
                      current_deletions, mean_open,  std_open,  C_OPEN)
        plot_segments(ax, dist_STI1, original_positions, original_section,
                      current_deletions, mean_bound, std_bound, C_BOUND)
        plot_segments(ax, dist_STI1, original_positions, original_section,
                      current_deletions, mean_evraw, std_evraw, C_EVRAW)
        plot_segments(ax, dist_STI1, original_positions, original_section,
                      current_deletions, mean_evfit, std_evfit, C_EVFIT)

        ax.set_ylim(-6, -1.5)
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        if i == 0:
            ax.set_xlim(dist_STI1[0], -5)
            # ax.set_ylabel(f"{th}\nlog($P_g$)", fontsize=14)
            ax.text(120 - STI1_locs[0][0], -1.85, "HS1", fontsize=14)
            ax.text(-68, -1.85, f"{th}", fontsize=14)

            if j == 0:
                ax.secondary_xaxis('top', functions=(lambda x: 146 + x, lambda x: x - 146))
        else:
            ax.set_xlim(5, dist_STI1[-1])
            ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            ax.text(279 - STI1_locs[0][1], -1.85, "HS2", fontsize=14)
            ax.text(302 - STI1_locs[0][1], -1.85, "HS3", fontsize=14)
            if j == 0:
                ax.secondary_xaxis('top', functions=(lambda x: x + 223, lambda x: x + 325))



for j in range(len(directory)):
    axes[j, 0].axvspan(-13, -33, color='orange', alpha=0.3)
    axes[j, 1].axvspan(56, 68,   color='orange', alpha=0.3)
    axes[j, 1].axvspan(80, 90,   color='orange', alpha=0.3)


fig.supylabel("log($P_{\\mathrm{g}}$)", x=-0.02, fontsize='large')
fig.text(0.53, 1.0, "Residue Number", ha='center', va='center', fontsize='large')
fig.text(0.53, -0.01, "Distance from STI1", ha='center', va='center', fontsize='large')

plt.tight_layout()
plt.savefig('Fig_S7.svg', bbox_inches='tight')
plt.close()