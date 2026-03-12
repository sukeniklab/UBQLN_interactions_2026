import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import csv

plt.rcParams['svg.fonttype'] = 'none'

def smooth_log_histogram(log_histogram, smooth):
    valid_mask = ~np.isnan(log_histogram)
    if np.sum(valid_mask) < 4:
        return log_histogram
    x_coords = np.arange(len(log_histogram))
    valid_x = x_coords[valid_mask]
    valid_y = log_histogram[valid_mask]
    spline = UnivariateSpline(valid_x, valid_y, s=smooth, k=3)
    smoothed = np.full_like(log_histogram, np.nan)
    min_valid, max_valid = valid_x.min(), valid_x.max()
    interp_mask = (x_coords >= min_valid) & (x_coords <= max_valid)
    smoothed[interp_mask] = spline(x_coords[interp_mask])
    return smoothed

def create_original_to_shifted_mapping(original_range, deletions_list):
    original_residues = np.arange(original_range[0], original_range[1]+1)
    shifted_positions = []
    original_positions = []
    for orig_res in original_residues:
        if any(del_start <= orig_res <= del_end for del_start, del_end in deletions_list):
            continue
        deletions_before = sum((del_end - del_start + 1) for del_start, del_end in deletions_list if del_end < orig_res)
        shifted_positions.append(orig_res - deletions_before)
        original_positions.append(orig_res)
    return np.array(original_positions), np.array(shifted_positions)

def calculate_optimal_weight(variant_base, exp_rg, trials):
    """Calculate optimal weight for a given variant to match experimental Rg"""
    mods = [variant_base, f"{variant_base}_bound"]
    avg_rgs = []
    
    for mod in mods:
        avg_rg = []
        for trial in trials:
            rg = 10 * np.load(f"../../Data/old_pipeline/CALVADOS3COM_2.0_MD_gpu_trial{trial}_{mod}/{mod}/0/Rg_traj.npy")
            avg_rg.append(np.mean(rg))
        avg_rgs.append(np.mean(avg_rg))
    
    def weighted_rg(weight):
        return weight * avg_rgs[0] + (1 - weight) * avg_rgs[1]
    
    def objective(weight):
        return (weighted_rg(weight) - exp_rg)**2
    
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    optimal_weight_unbound = result.x
    optimal_weight_bound = 1 - optimal_weight_unbound
    
    print(f"\n=== {variant_base} Optimal Weighting ===")
    print(f"Target experimental Rg: {exp_rg:.1f} Å")
    print(f"Optimal weight for unbound: {optimal_weight_unbound:.3f}")
    print(f"Optimal weight for bound: {optimal_weight_bound:.3f}")
    print(f"Resulting weighted Rg: {weighted_rg(optimal_weight_unbound):.2f} Å")
    
    return optimal_weight_unbound, optimal_weight_bound

# ---- Parameters ----
trials = list(range(1, 11))
original_domains = [[75,145], [223,325]]
deletions = {
    "full": [],
    "deltaTH1": [[114,134]],
    "deltaTH2": [[279,291]],
    "deltaTH3": [[303,313]]
}
STI1_locs = [[146,222]]
numReplicas=20
sim_time = 70 #ns
print_interval = 0.01

snaps = numReplicas * int((sim_time/print_interval) - (0.05 * (sim_time/print_interval)))

# ===== Define which variants to analyze and their experimental Rg values =====
weighted_variants_config = [
    {
        "name": "Dsk2_full",
        "label": "Dsk2 Full",
        "color": "k",
        "exp_rg": 38.2,
        "deletion_key": "full"
    },

    {
        "name": "Dsk2_deltaTH1",
        "label": "Dsk2 dTH1",
        "color": "green",
        "exp_rg": 35.74,  # Change this to your experimental value for deltaTH3
        "deletion_key": "deltaTH1"
    },

        {
        "name": "Dsk2_deltaTH2",
        "label": "Dsk2 dTH2",
        "color": "magenta",
        "exp_rg":37.47,  # Change this to your experimental value for deltaTH3
        "deletion_key": "deltaTH2"
    },


    {
        "name": "Dsk2_deltaTH3",
        "label": "Dsk2 dTH3",
        "color": "orange",
        "exp_rg": 40.2,  # Change this to your experimental value for deltaTH3
        "deletion_key": "deltaTH3"
    },




    # Add more variants here as needed:
    # {
    #     "name": "Dsk2_deltaTH1",
    #     "label": "Dsk2 ΔTH1",
    #     "color": "darkorange",
    #     "exp_rg": 36.0,
    #     "deletion_key": "deltaTH1"
    # },
]

# ===== Calculate optimal weights for each variant =====
weights = {}
for config in weighted_variants_config:
    weight_unbound, weight_bound = calculate_optimal_weight(
        config["name"], 
        config["exp_rg"], 
        trials
    )
    weights[config["name"]] = {
        "unbound": weight_unbound,
        "bound": weight_bound
    }

# ===== Calculate shifted domains for each variant =====
shifted_domains = {}
for config in weighted_variants_config:
    variant_name = config["name"]
    current_deletions = deletions[config["deletion_key"]]
    shifted_domains[variant_name] = []
    
    for original_section in original_domains:
        deletions_before = sum((del_end - del_start + 1) 
                              for del_start, del_end in current_deletions 
                              if del_end < original_section[0])
        deletions_within = sum((min(del_end, original_section[1]) - max(del_start, original_section[0]) + 1)
                              for del_start, del_end in current_deletions 
                              if del_start <= original_section[1] and del_end >= original_section[0])
        
        shifted_start = original_section[0] - deletions_before
        shifted_end = original_section[1] - deletions_before - deletions_within
        shifted_domains[variant_name].append([shifted_start, shifted_end])
    
    print(f"{variant_name}: {shifted_domains[variant_name]}")

# ===== Load and process data =====
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for j, config in enumerate(weighted_variants_config):
    variant_name = config["name"]
    current_deletions = deletions[config["deletion_key"]]
    variant_domains = shifted_domains[variant_name]
    weight_unbound = weights[variant_name]["unbound"]
    weight_bound = weights[variant_name]["bound"]
    
    # Store data for CSV output - dictionary to map residue to data
    # Format: [dist_STI1, weighted_mean, weighted_std, unbound_mean, unbound_std, bound_mean, bound_std]
    csv_data_dict = {}

    for i, (original_section, shifted_section) in enumerate(zip(original_domains, variant_domains)):
        original_positions, shifted_positions = create_original_to_shifted_mapping(
            original_section, current_deletions
        )
        if len(shifted_positions) == 0:
            continue

        all_histograms = []
        all_histograms_ex = []
        all_histograms_unbound = []
        all_histograms_ex_unbound = []
        all_histograms_bound = []
        all_histograms_ex_bound = []
        
        for trial in trials:
            try:
                # Load unbound data
                histogram_unbound = np.load(
                    f"Histos/{variant_name}_trial{trial}_innerVol_idr{shifted_section[0]}_{shifted_section[1]}.npy"
                ) / snaps
                histogram_ex_unbound = np.load(
                    f"Histos/{variant_name}_EV_trial{trial}_innerVol_idr{shifted_section[0]}_{shifted_section[1]}.npy"
                ) / (10*snaps)
                
                # Load bound data
                histogram_bound = np.load(
                    f"Histos/{variant_name}_bound_trial{trial}_innerVol_idr{shifted_section[0]}_{shifted_section[1]}.npy"
                ) / snaps
                histogram_ex_bound = np.load(
                    f"Histos/{variant_name}_EV_trial{trial}_innerVol_idr{shifted_section[0]}_{shifted_section[1]}.npy"
                ) / (10*snaps)
                
                # Weight the histograms
                histogram = weight_unbound * histogram_unbound + weight_bound * histogram_bound
                histogram_ex = weight_unbound * histogram_ex_unbound + weight_bound * histogram_ex_bound
                
                # Remove deleted residues
                mask_keep = np.ones_like(histogram, dtype=bool)
                for del_start, del_end in current_deletions:
                    mask_keep[(original_positions >= del_start) & (original_positions <= del_end)] = False
                histogram = histogram[mask_keep]
                histogram_ex = histogram_ex[mask_keep]
                histogram_unbound = histogram_unbound[mask_keep]
                histogram_ex_unbound = histogram_ex_unbound[mask_keep]
                histogram_bound = histogram_bound[mask_keep]
                histogram_ex_bound = histogram_ex_bound[mask_keep]
                original_positions_trimmed = original_positions[mask_keep]

                # Trim edges
                if i == 0:
                    histogram = histogram[:-4]
                    histogram_ex = histogram_ex[:-4]
                    histogram_unbound = histogram_unbound[:-4]
                    histogram_ex_unbound = histogram_ex_unbound[:-4]
                    histogram_bound = histogram_bound[:-4]
                    histogram_ex_bound = histogram_ex_bound[:-4]
                    original_positions_trimmed = original_positions_trimmed[:-4]
                    smoothing_factor = 1
                else:
                    histogram = histogram[4:]
                    histogram_ex = histogram_ex[4:]
                    histogram_unbound = histogram_unbound[4:]
                    histogram_ex_unbound = histogram_ex_unbound[4:]
                    histogram_bound = histogram_bound[4:]
                    histogram_ex_bound = histogram_ex_bound[4:]
                    original_positions_trimmed = original_positions_trimmed[4:]
                    smoothing_factor = 2.4

                # Smooth EV histograms
                log_ex = np.log10(histogram_ex)
                log_ex[np.isinf(log_ex)] = np.nan
                smoothed_log = smooth_log_histogram(log_ex, smoothing_factor)
                smooth_ev = 10**smoothed_log

                log_ex_unbound = np.log10(histogram_ex_unbound)
                log_ex_unbound[np.isinf(log_ex_unbound)] = np.nan
                smoothed_log_unbound = smooth_log_histogram(log_ex_unbound, smoothing_factor)
                smooth_ev_unbound = 10**smoothed_log_unbound

                log_ex_bound = np.log10(histogram_ex_bound)
                log_ex_bound[np.isinf(log_ex_bound)] = np.nan
                smoothed_log_bound = smooth_log_histogram(log_ex_bound, smoothing_factor)
                smooth_ev_bound = 10**smoothed_log_bound

                all_histograms.append(histogram)
                all_histograms_ex.append(smooth_ev)
                all_histograms_unbound.append(histogram_unbound)
                all_histograms_ex_unbound.append(smooth_ev_unbound)
                all_histograms_bound.append(histogram_bound)
                all_histograms_ex_bound.append(smooth_ev_bound)

            except FileNotFoundError as e:
                print(f"  File not found for {variant_name}, trial {trial}: {e}")
                continue

        if len(all_histograms) == 0:
            print(f"  No valid data for {variant_name}, domain {i}")
            continue

        # Convert to arrays
        all_histograms = np.array(all_histograms)
        all_histograms_ex = np.array(all_histograms_ex)
        all_histograms_unbound = np.array(all_histograms_unbound)
        all_histograms_ex_unbound = np.array(all_histograms_ex_unbound)
        all_histograms_bound = np.array(all_histograms_bound)
        all_histograms_ex_bound = np.array(all_histograms_ex_bound)

        # Calculate mean and std for weighted ensemble
        mean_histogram = np.nanmean(all_histograms, axis=0)
        std_histogram = np.nanstd(all_histograms, axis=0, ddof=1)
        mean_histogram_ex = np.nanmean(all_histograms_ex, axis=0)
        std_histogram_ex = np.nanstd(all_histograms_ex, axis=0, ddof=1)
        mean_ratio = mean_histogram / mean_histogram_ex
        relative_error_squared = (std_histogram / mean_histogram)**2 + (std_histogram_ex / mean_histogram_ex)**2
        std_ratio = mean_ratio * np.sqrt(relative_error_squared)
        std_ratio[np.isinf(std_ratio)] = np.nan
        mean_ratio[np.isinf(mean_ratio)] = np.nan

        # Calculate mean and std for unbound ensemble
        mean_histogram_unbound = np.nanmean(all_histograms_unbound, axis=0)
        std_histogram_unbound = np.nanstd(all_histograms_unbound, axis=0, ddof=1)
        mean_histogram_ex_unbound = np.nanmean(all_histograms_ex_unbound, axis=0)
        std_histogram_ex_unbound = np.nanstd(all_histograms_ex_unbound, axis=0, ddof=1)
        mean_ratio_unbound = mean_histogram_unbound / mean_histogram_ex_unbound
        relative_error_squared_unbound = (std_histogram_unbound / mean_histogram_unbound)**2 + (std_histogram_ex_unbound / mean_histogram_ex_unbound)**2
        std_ratio_unbound = mean_ratio_unbound * np.sqrt(relative_error_squared_unbound)
        std_ratio_unbound[np.isinf(std_ratio_unbound)] = np.nan
        mean_ratio_unbound[np.isinf(mean_ratio_unbound)] = np.nan

        # Calculate mean and std for bound ensemble
        mean_histogram_bound = np.nanmean(all_histograms_bound, axis=0)
        std_histogram_bound = np.nanstd(all_histograms_bound, axis=0, ddof=1)
        mean_histogram_ex_bound = np.nanmean(all_histograms_ex_bound, axis=0)
        std_histogram_ex_bound = np.nanstd(all_histograms_ex_bound, axis=0, ddof=1)
        mean_ratio_bound = mean_histogram_bound / mean_histogram_ex_bound
        relative_error_squared_bound = (std_histogram_bound / mean_histogram_bound)**2 + (std_histogram_ex_bound / mean_histogram_ex_bound)**2
        std_ratio_bound = mean_ratio_bound * np.sqrt(relative_error_squared_bound)
        std_ratio_bound[np.isinf(std_ratio_bound)] = np.nan
        mean_ratio_bound[np.isinf(mean_ratio_bound)] = np.nan

        # Distance from STI1
        if i == 0:
            dist_STI1 = original_positions_trimmed - STI1_locs[0][0]
            axes[i].set_xlim(dist_STI1[0], -5) 
        else:
            dist_STI1 = original_positions_trimmed - STI1_locs[0][1]
            axes[i].set_xlim(5, dist_STI1[-1])
        
        # Store data for CSV in dictionary
        for res, d, m, s, m_ub, s_ub, m_b, s_b in zip(original_positions_trimmed, dist_STI1, 
                                                        mean_ratio, std_ratio,
                                                        mean_ratio_unbound, std_ratio_unbound,
                                                        mean_ratio_bound, std_ratio_bound):
            csv_data_dict[int(res)] = [d, m, s, m_ub, s_ub, m_b, s_b] 

        # Plot
        axes[i].scatter(dist_STI1, mean_ratio, color=config["color"], s=10, zorder=100)
        axes[i].plot(dist_STI1, mean_ratio, color=config["color"], linewidth=0.8, zorder=100)
        axes[i].fill_between(dist_STI1, mean_ratio - std_ratio, mean_ratio + std_ratio, 
                            color=config["color"], alpha=0.15, zorder=50)
        axes[i].set_ylim(0, 200)
        axes[i].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    # Write CSV for this variant (one file per construct)
    # Include all residues from original domains, with NaN for deleted regions
    if csv_data_dict:
        csv_filename = f"{variant_name}_excess_prob.csv"
        csv_data = []
        
        for domain_idx, original_section in enumerate(original_domains):
            for res in range(original_section[0], original_section[1] + 1):
                if res in csv_data_dict:
                    csv_data.append([res] + csv_data_dict[res])
                else:
                    # Residue is deleted, calculate dist_STI1 but add NaN for mean/std values
                    if domain_idx == 0:
                        dist_sti1 = res - STI1_locs[0][0]
                    else:
                        dist_sti1 = res - STI1_locs[0][1]
                    csv_data.append([res, dist_sti1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['residue_number', 'dist_STI1', 'weighted_mean', 'weighted_std', 
                           'unbound_mean', 'unbound_std', 'bound_mean', 'bound_std'])
            writer.writerows(csv_data)
        print(f"Wrote data to {csv_filename}")

# Mark TH regions
axes[0].axvspan(-13, -33, color='orange', alpha=0.3)
axes[1].axvspan(56, 68, color='orange', alpha=0.3)
axes[1].axvspan(80, 90, color='orange', alpha=0.3)

# Legend
for config in weighted_variants_config:
    axes[0].scatter(1000, 1000, color=config["color"], s=10, 
                   label=f"{config['label']}", 
                   zorder=100)

# fig.suptitle('Weighted Ensemble Excess Probability')

fig.legend(bbox_to_anchor=(1.05, 0.55))
fig.supylabel("Excess Probability ($P_{\mathrm{g}}$/$P_{\mathrm{g,EV}}$)", x=0.05)
fig.supxlabel("Distance from STI1", y=-0.02)

plt.savefig('excess_probability_weighted_ensembles_newFL.png', dpi=300, bbox_inches="tight")
plt.close()