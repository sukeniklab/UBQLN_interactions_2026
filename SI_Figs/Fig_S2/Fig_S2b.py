import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.sans-serif'] = 'Arial'


FL_EXP       = "S5E_FL_I(q)_20260307.csv"
OPEN_NAME    = "Dsk2_full"
CLOSED_NAME  = "Dsk2_full_bound"
FOXS_BASE    = "foxs_profile"
TRIAL        = "trial1"

OUTPUT_BASE  = "fits"
QMIN, QMAX   = 0.0, 0.30
FS_LABEL     = 18
FS_LEGEND    = 12
COLOR_OPEN   = "blue"
COLOR_CLOSED = "purple"


def normalize_to_exp_lowq(I_sim, I_exp, idx_start=1, idx_end=5):
    """Anchor sim to the mean of exp over a small low-q window."""
    return I_sim * (np.mean(I_exp[idx_start:idx_end]) / np.mean(I_sim[idx_start:idx_end]))

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


def load_pool_mean(construct_name, q_target, foxs_base=FOXS_BASE, trial=TRIAL):
    path = f"{foxs_base}/{trial}/{construct_name}_{trial}_avg.dat"
    print(f"  Loading {path}")
    d = np.loadtxt(path, comments="#")
    q_src = d[:, 0]
    I_src = d[:, 1] / d[0, 1]   # normalize to I(0)=1
    fi = interp1d(q_src, I_src, kind="cubic",
                  bounds_error=False, fill_value="extrapolate")
    return fi(q_target)


fig, ax0 = plt.subplots(1, 1, figsize=(4, 4))


q_fl, I_fl, sig_fl = load_exp(FL_EXP)


I_open_mean   = normalize_to_exp_lowq(load_pool_mean(OPEN_NAME,   q_fl), I_fl, idx_start=1, idx_end=5)
I_closed_mean = normalize_to_exp_lowq(load_pool_mean(CLOSED_NAME, q_fl), I_fl, idx_start=1, idx_end=5)


ax0.errorbar(q_fl, I_fl, yerr=sig_fl,
             fmt="o", ms=2, color="black",
             elinewidth=0.5, capsize=0, alpha=0.35,
             label="Exp. FL", zorder=3)
ax0.semilogy(q_fl,I_open_mean,   lw=2, color=COLOR_OPEN,
             label="Simulated Open",   zorder=4)
ax0.semilogy(q_fl, I_closed_mean,  lw=2, color=COLOR_CLOSED,
             label="Simulated Closed", zorder=4)

ax0.set_yscale("log")
ax0.set_ylim(1e-4, 2)
ax0.set_xlim(QMIN, 0.3)
ax0.set_xlabel(r"$q$ (Å$^{-1}$)", fontsize=FS_LABEL)   # now visible
ax0.set_ylabel(r"$I(q)$", fontsize=FS_LABEL, labelpad=4)
ax0.tick_params(axis="both", labelsize=FS_LABEL)
ax0.legend(fontsize=FS_LEGEND, loc="lower left", frameon=False)


plt.tight_layout()
plt.savefig("Fig_S2b.svg", dpi=300, bbox_inches="tight")
plt.savefig("Fig_S2b.png", dpi=150, bbox_inches="tight")
plt.close()
