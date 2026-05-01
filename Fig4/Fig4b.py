import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d as _interp1d

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.sans-serif'] = 'Arial'

# ── Config ────────────────────────────────────────────────────────────────────
CONSTRUCTS = [
    dict(label="Dsk2_full",  exp="exp_data/S5E_FL_I(q)_20260307.csv",   color="grey", display="Dsk2 FL"),
    dict(label="Dsk2_dTH1",  exp="exp_data/S5B_dHS1_I(q)_20260213.csv", color="green",    display="ΔHS1"),
    dict(label="Dsk2_dTH2",  exp="exp_data/S5C_dHS2_I(q)_20260213.csv", color="fuchsia",  display="ΔHS2"),
    dict(label="Dsk2_dTH3",  exp="exp_data/S5F_dHS3_I(q)_20260307.csv", color="orange",   display="ΔHS3"),
    dict(label="Dsk2_I45A",  exp="exp_data/S5D_I45A_I(q)_20260213.csv", color="red",  display="Dsk2 I45A"),

]

TRIALS       = ["trial1", "trial2", "trial3"]
FITS_BASE    = "theoretical_scattering"
TRIAL_ALPHA  = 0.5
TRIAL_STYLES = ["-", "--", ":"]
N_SCATTER    = 50

QMIN, QMAX   = 0.0, 0.30
FS_LABEL     = 14
FS_LEGEND    = 9

def load_exp(path, qmin=QMIN, qmax=QMAX):
    delim = "," if path.endswith(".csv") else None
    skip = 0
    with open(path) as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                float(s.split(delim)[0])
                skip = i
            except ValueError:
                skip = i + 1
            break
    data = np.loadtxt(path, comments="#", delimiter=delim, skiprows=skip)
    q, I, sigma = data[:, 0], data[:, 1], data[:, 2]
    mask = (q >= qmin) & (q <= qmax) & (sigma > 0) & (I > 0)
    q, I, sigma = q[mask], I[mask], sigma[mask]
    return q, I / I[0], sigma / I[0]


def load_recon_trials(label, trials=TRIALS, fits_base=FITS_BASE):
    all_I        = []
    q_ref        = None
    loaded_trials = []

    for trial in trials:
        recon_path   = f"{fits_base}/{trial}/{label}/{label}_reconstructed_corrected.dat"
        results_path = f"{fits_base}/{trial}/{label}/{label}_fit_results.txt"

        try:
            d = np.loadtxt(recon_path, comments="#")
        except FileNotFoundError:
            print(f"  WARNING: not found, skipping — {recon_path}")
            continue

        q_src, I_src = d[:, 0], d[:, 1]

        if q_ref is None:
            q_ref = q_src
        elif not np.allclose(q_src, q_ref, rtol=1e-4):
            print(f"  WARNING: q-grid mismatch for {trial}/{label} — skipping")
            continue

        # Load sigma_scale for this trial
        sigma_scale = 1.0
        try:
            with open(results_path) as f:
                for line in f:
                    if line.strip().startswith("sigma_scale"):
                        sigma_scale = float(line.split("=")[1].split("±")[0].strip())
                        break
        except FileNotFoundError:
            print(f"  WARNING: results file not found, using sigma_scale=1.0")

  
        all_I.append(sigma_scale * I_src)
        loaded_trials.append(trial)

    if not all_I:
        raise RuntimeError(f"No trial data found for {label}")

    all_I = np.array(all_I)   
    return q_ref, all_I, all_I.mean(axis=0), all_I.std(axis=0, ddof=1), loaded_trials


def load_fit_results(label, fits_base=FITS_BASE, trials=TRIALS):
    x_opens, chi2s = [], []

    for trial in trials:
        path = f"{fits_base}/{trial}/{label}/{label}_fit_results.txt"
        try:
            with open(path) as f:
                for line in f:
                    if line.strip().startswith("x_open           ="):
                        x_opens.append(float(line.split("=")[1].split("±")[0].strip()))
                    if line.strip().startswith("chi2_red"):
                        chi2s.append(float(line.split("=")[1].strip()))
        except FileNotFoundError:
            print(f"  WARNING: results file not found — {path}")
            continue

    x_open   = np.mean(x_opens)
    x_std    = np.std(x_opens, ddof=1)    
    chi2_red = np.mean(chi2s)


    return x_open, x_std, chi2_red


n = len(CONSTRUCTS)
height_ratios = [3, 1] * n

fig, axes = plt.subplots(
    2 * n, 1,
    figsize=(6, 4.5 * n),
    gridspec_kw={"height_ratios": height_ratios, "hspace": 0},
)

if n == 1:
    axes = [axes[0], axes[1]]

for i, c in enumerate(CONSTRUCTS):
    ax_top = axes[2 * i]
    ax_bot = axes[2 * i + 1]

    label   = c["label"]
    color   = c["color"]
    display = c["display"]


    # Load experimental data 
    q_exp, I_exp, sigma_exp = load_exp(c["exp"])

    # Load reconstructed profiles 
    q_rec, all_I_trials, I_rec_mean, I_rec_std, loaded_trials = load_recon_trials(label)

    # Load x_open mean and std across trials
    x_open, x_std, chi2_red = load_fit_results(label)

    eb_step = max(1, len(q_rec) // N_SCATTER)
    eb_idx  = np.arange(0, len(q_rec), eb_step)
    yerr_lo = np.minimum(I_rec_std[eb_idx], I_rec_mean[eb_idx] * 0.999)


    REPLICATE_COLORS = ["grey", "red"]

    # Experimental data
    ax_top.errorbar(q_exp, I_exp, yerr=sigma_exp,
                    fmt="o", ms=2, color=color,
                    elinewidth=0.5, capsize=0, alpha=0.35,
                    label="Experimental", zorder=4)

    # # Mean ± std as scatter 
    ax_top.errorbar(q_rec[eb_idx], I_rec_mean[eb_idx],
                    yerr=[yerr_lo, I_rec_std[eb_idx]],
                    fmt="s", ms=4, color='k',
                    elinewidth=1.0, capsize=2, alpha=0.9,
                    ecolor="k", label="Reconstructed Ensemble", zorder=5)

    ax_top.set_yscale("log")
    ax_top.set_ylim(1e-4, 2)
    ax_top.set_xlim(QMIN, QMAX)
    ax_top.set_ylabel(r"$I(q)$", fontsize=FS_LABEL, labelpad=4)
    ax_top.tick_params(axis="both", labelsize=FS_LABEL)
    ax_top.set_xticklabels([])
    ax_top.legend(fontsize=FS_LEGEND, loc="lower left", frameon=False)
    ax_top.text(0.97, 0.97,
                f"{display}\n"
                f"$f_{{open}}$ = {x_open:.3f} ± {x_std:.3f}\n"
                f"$f_{{close}}$ = {(1-x_open):.3f} ± {x_std:.3f}",
                transform=ax_top.transAxes,
                ha="right", va="top", fontsize=FS_LEGEND + 1, color=color)

###Residuals panel

    fi     = _interp1d(q_rec, I_rec_mean, kind="cubic",
                       bounds_error=False, fill_value="extrapolate")
    resid  = (I_exp - fi(q_exp)) / sigma_exp

    ax_bot.axhline(0, color="k", linestyle="--", lw=1.5, zorder=1)
    ax_bot.scatter(q_exp, resid, s=4, color=color, alpha=0.4, zorder=3)
    ax_bot.set_ylim(-8, 8)
    ax_bot.set_xlim(QMIN, QMAX)
    ax_bot.set_yticks([-5, 0, 5])
    ax_bot.tick_params(axis="both", labelsize=FS_LABEL - 2)
    ax_bot.set_ylabel(r"$\Delta/\sigma$", fontsize=FS_LABEL - 2, labelpad=4)

    if i < n - 1:
        ax_bot.set_xticklabels([])
    else:
        ax_bot.set_xlabel(r"$q$ (Å$^{-1}$)", fontsize=FS_LABEL)

plt.subplots_adjust(hspace=0)
plt.savefig("Fig4D.svg", dpi=300, bbox_inches="tight")
plt.savefig("Fig4D.png", dpi=150, bbox_inches="tight")
plt.close()


