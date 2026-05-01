import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_dat(path):
    with open(path) as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            if s and not s.startswith("#"):
                delim = "," if "," in s else None
                try:
                    float(s.split(delim)[0])
                    skip = 0
                except ValueError:
                    skip = i + 1
                break
    data = np.loadtxt(path, comments="#", delimiter=delim, skiprows=skip)
    return data[:, 0], data[:, 1], data[:, 2]


def pool_frames(glob_pattern):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files found: {glob_pattern}")
    print(f"  Loading {len(files)} frames...")
    all_I, q, n_skipped = [], None, 0
    valid_paths = []
    for f in files:
        try:
            data = np.loadtxt(f, comments="#")
        except Exception as e:
            print(f"  WARNING: could not load {f}: {e}")
            n_skipped += 1
            continue
        if q is None:
            q = data[:, 0]
        if data.shape[0] != len(q) or not np.allclose(data[:, 0], q, rtol=1e-4):
            fi = interp1d(data[:, 0], data[:, 1], kind="cubic",
                          bounds_error=False, fill_value="extrapolate")
            all_I.append(fi(q))
        else:
            all_I.append(data[:, 1])
        pdb = f[:-4] if f.endswith(".dat") else f
        valid_paths.append(pdb)
    if not all_I:
        raise ValueError(f"No valid frames loaded from {glob_pattern}")
    all_I = np.array(all_I)
    print(f"  Pool shape: {all_I.shape}  ({all_I.nbytes / 1e6:.1f} MB)")
    if n_skipped:
        print(f"  Skipped:    {n_skipped} files")
    return q, all_I, valid_paths


def normalize_I0(I, err=None):
    I0 = I[0]
    if I0 <= 0:
        raise ValueError(f"I(0) = {I0:.3g} ≤ 0")
    return (I / I0, err / I0) if err is not None else I / I0


def normalize_matrix(I_mat):
    I0 = I_mat[:, 0][:, None]
    if np.any(I0 <= 0):
        raise ValueError("Some frames have I(0) ≤ 0.")
    return I_mat / I0


def interp_to_grid(q_target, q_src, I_src):
    f = interp1d(q_src, I_src, kind="cubic",
                 bounds_error=False, fill_value="extrapolate")
    return f(q_target)


def interp_matrix_to_grid(q_target, q_src, I_mat):
    out = np.zeros((I_mat.shape[0], len(q_target)))
    for i, row in enumerate(I_mat):
        out[i] = interp_to_grid(q_target, q_src, row)
        if i % 50_000 == 0 and i > 0:
            print(f"    interpolated {i}/{I_mat.shape[0]} frames", flush=True)
    return out


# ── Fitting ───────────────────────────────────────────────────────────────────

def two_state_fit(q, I_exp, sigma_exp, I_open, I_closed):
    """Fit c * (x * I_open + (1-x) * I_closed) to I_exp.
    Returns x_open, sigma_scale, and their 1-sigma errors from the covariance matrix."""
    def model(q_, x, scale):
        return scale * (x * I_open + (1.0 - x) * I_closed)

    try:
        popt, pcov = curve_fit(
            model, q, I_exp,
            p0=[0.5, 1.0],
            sigma=sigma_exp,
            absolute_sigma=True,
            bounds=([0.0, 1e-9], [1.0, np.inf]),
            method="trf",
            maxfev=10_000,
        )
        perr = np.sqrt(np.diag(pcov))
        x, scale = popt
        x_err, scale_err = perr
        I_fit = model(q, x, scale)
        return x, scale, x_err, scale_err, I_fit
    except Exception as e:
        print(f"  WARNING: fit failed — {e}")
        return None, None, None, None, None


def reduced_chi2(I_exp, I_fit, sigma_exp, n_params=2):
    return np.sum(((I_exp - I_fit) / sigma_exp) ** 2) / (len(I_exp) - n_params)


# ── Ensemble reconstruction ───────────────────────────────────────────────────

def reconstruct_ensemble(I_open_all, I_closed_all, x_open, n_draws, seed=42):
    """Single random draw of frames in proportion x_open : (1-x_open).
    Returns the mean reconstructed I(q) and the selection counts per frame."""
    rng = np.random.default_rng(seed)

    n_open_pool   = I_open_all.shape[0]
    n_closed_pool = I_closed_all.shape[0]

    n_open_draw   = int(round(x_open * n_draws))
    n_closed_draw = n_draws - n_open_draw

    open_idx   = rng.integers(0, n_open_pool,   size=max(n_open_draw,   1))
    closed_idx = rng.integers(0, n_closed_pool, size=max(n_closed_draw, 1))

    open_counts   = np.zeros(n_open_pool,   dtype=np.int64)
    closed_counts = np.zeros(n_closed_pool, dtype=np.int64)
    np.add.at(open_counts,   open_idx,   1)
    np.add.at(closed_counts, closed_idx, 1)

    selected = np.vstack([I_open_all[open_idx], I_closed_all[closed_idx]])
    I_recon  = selected.mean(axis=0)

    print(f"  Drew {n_open_draw} open + {n_closed_draw} closed frames "
          f"(x_open = {x_open:.4f})")

    return I_recon, open_counts, closed_counts


# ── Trajectory writer ─────────────────────────────────────────────────────────

def write_ensemble_trajectory(open_paths, closed_paths,
                               open_counts, closed_counts,
                               n_frames, output_path):
    open_ranked   = np.argsort(open_counts)[::-1]
    closed_ranked = np.argsort(closed_counts)[::-1]
    open_ranked   = [i for i in open_ranked   if open_counts[i]   > 0]
    closed_ranked = [i for i in closed_ranked if closed_counts[i] > 0]

    total_open   = open_counts.sum()
    total_closed = closed_counts.sum()
    total        = total_open + total_closed
    x_open_actual = total_open / total if total > 0 else 0.5

    n_open_out   = max(1, int(round(x_open_actual * n_frames)))
    n_closed_out = n_frames - n_open_out
    n_open_out   = min(n_open_out,   len(open_ranked))
    n_closed_out = min(n_closed_out, len(closed_ranked))

    open_selected   = [(open_counts[i],   "open",   open_paths[i])   for i in open_ranked[:n_open_out]]
    closed_selected = [(closed_counts[i], "closed", closed_paths[i]) for i in closed_ranked[:n_closed_out]]

    open_selected.sort(key=lambda x: x[0],   reverse=True)
    closed_selected.sort(key=lambda x: x[0], reverse=True)

    selected = []
    oi, ci = 0, 0
    open_step   = 1.0 / n_open_out   if n_open_out   > 0 else np.inf
    closed_step = 1.0 / n_closed_out if n_closed_out > 0 else np.inf
    open_acc, closed_acc = 0.0, 0.0
    while oi < n_open_out or ci < n_closed_out:
        if open_acc <= closed_acc and oi < n_open_out:
            selected.append(open_selected[oi]);  open_acc += open_step;   oi += 1
        elif ci < n_closed_out:
            selected.append(closed_selected[ci]); closed_acc += closed_step; ci += 1
        else:
            selected.append(open_selected[oi]);  open_acc += open_step;   oi += 1

    n_open_out   = sum(1 for _, kind, _ in selected if kind == "open")
    n_closed_out = len(selected) - n_open_out

    print(f"\n── Writing ensemble trajectory ─────────────────────────────────")
    print(f"  Top {len(selected)} frames: {n_open_out} open, {n_closed_out} closed")
    print(f"  Output: {output_path}")

    n_missing, model_num = 0, 1
    with open(output_path, "w") as out:
        out.write(f"REMARK  Reconstructed SAXS ensemble — {len(selected)} frames\n")
        out.write(f"REMARK  {n_open_out} open, {n_closed_out} closed\n")
        out.write(f"REMARK  Ranked by selection frequency\n")
        for count, kind, pdb_path in selected:
            if not os.path.exists(pdb_path):
                n_missing += 1
                continue
            out.write(f"MODEL     {model_num:4d}\n")
            out.write(f"REMARK  source={pdb_path}  state={kind}  count={count}\n")
            with open(pdb_path) as pdb:
                for line in pdb:
                    if not line.strip().startswith("END"):
                        out.write(line)
            out.write("ENDMDL\n")
            model_num += 1
        out.write("END\n")

    if n_missing:
        print(f"  WARNING: {n_missing} PDB files not found (skipped)")
    print(f"  Written {model_num - 1} models to {output_path}")

    manifest_path = output_path.replace(".pdb", "_manifest.txt")
    with open(manifest_path, "w") as mf:
        mf.write("model  state   count  pdb_path\n")
        m = 1
        for count, kind, pdb_path in selected:
            if os.path.exists(pdb_path):
                mf.write(f"{m:5d}  {kind:6s}  {count:6d}  {pdb_path}\n")
                m += 1
    print(f"  Manifest → {manifest_path}")


# ── Plot ──────────────────────────────────────────────────────────────────────

COLORS = {"exp": "black", "open": "royalblue", "closed": "mediumorchid", "recon": "tomato"}


def plot_saxs(q, I_exp, sigma_exp, I_open, I_closed, I_recon,
              sigma_scale, x_open, x_err, chi2_red, label, output_path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    ax1.errorbar(q, I_exp, yerr=sigma_exp,
                 fmt="o", ms=2, color=COLORS["exp"],
                 elinewidth=0.6, capsize=0, alpha=0.5,
                 label="Experimental", zorder=3)
    ax1.semilogy(q, sigma_scale * I_open,   lw=1.8, ls="--",
                 color=COLORS["open"],   zorder=4,
                 label=f"Open pool mean  ($f$ = {x_open:.3f} ± {x_err:.3f})")
    ax1.semilogy(q, sigma_scale * I_closed, lw=1.8, ls="--",
                 color=COLORS["closed"], zorder=4,
                 label=f"Closed pool mean  ($f$ = {1-x_open:.3f} ± {x_err:.3f})")
    ax1.semilogy(q, sigma_scale * I_recon,  lw=2,
                 color=COLORS["recon"],  zorder=5,
                 label="Reconstructed ensemble")

    ax1.set_ylabel(r"$I(q)\,/\,I(0)$")
    ax1.set_ylim(1e-4, 2)
    ax1.set_title(label)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.text(0.97, 0.03, f"$\\chi^2_{{red}}$ = {chi2_red:.3f}",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    resid = (I_exp - sigma_scale * I_recon) / sigma_exp
    ax2.axhline(0, color=COLORS["recon"], lw=1.5)
    ax2.scatter(q, resid, s=6, color=COLORS["exp"], alpha=0.5)
    ax2.axhspan(-1, 1, color="grey", alpha=0.12)
    ax2.set_xlabel(r"$q$ (Å$^{-1}$)")
    ax2.set_ylabel(r"$(I_\mathrm{exp} - I_\mathrm{rec})\,/\,\sigma$")
    ax2.set_ylim(-10, 10)
    ax2.grid(ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-state SAXS ensemble reconstruction — single random draw."
    )
    parser.add_argument("--open-glob",   required=True)
    parser.add_argument("--closed-glob", required=True)
    parser.add_argument("--exp",         required=True)
    parser.add_argument("--qmin",        type=float, default=None)
    parser.add_argument("--qmax",        type=float, default=None)
    parser.add_argument("--n-draws",     type=int,   default=50_000)
    parser.add_argument("--n-ensemble",  type=int,   default=500)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--label",       default="construct")
    parser.add_argument("--output-dir",  default="fits")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    label = args.label

    # ── Load pools ────────────────────────────────────────────────────────
    print("\nLoading open pool...")
    q_open,   I_open_all,   open_pdb_paths   = pool_frames(args.open_glob)
    print("Loading closed pool...")
    q_closed, I_closed_all, closed_pdb_paths = pool_frames(args.closed_glob)

    I_open_all   = normalize_matrix(I_open_all)
    I_closed_all = normalize_matrix(I_closed_all)

    # ── Load experimental data ────────────────────────────────────────────
    print("\nLoading experimental data...")
    q_exp, I_exp, sigma_exp = load_dat(args.exp)
    qmin = args.qmin or q_exp.min()
    qmax = args.qmax or q_exp.max()
    mask = (q_exp >= qmin) & (q_exp <= qmax) & (sigma_exp > 0) & (I_exp > 0)
    q_exp, I_exp, sigma_exp = q_exp[mask], I_exp[mask], sigma_exp[mask]
    I_exp, sigma_exp = normalize_I0(I_exp, sigma_exp)

    # ── Interpolate pool means onto experimental q-grid ───────────────────
    I_open_g   = interp_to_grid(q_exp, q_open,   I_open_all.mean(axis=0))
    I_closed_g = interp_to_grid(q_exp, q_closed, I_closed_all.mean(axis=0))

    # ── Interpolate full pools onto experimental q-grid ───────────────────
    print("\nInterpolating open pool onto exp q-grid...")
    I_open_all_g   = interp_matrix_to_grid(q_exp, q_open,   I_open_all)
    print("Interpolating closed pool onto exp q-grid...")
    I_closed_all_g = interp_matrix_to_grid(q_exp, q_closed, I_closed_all)

    # ── Deterministic two-state fit ───────────────────────────────────────
    print("\nFitting two-state model...")
    x_open, sigma_scale, x_err, sigma_err, I_fit = two_state_fit(
        q_exp, I_exp, sigma_exp, I_open_g, I_closed_g
    )
    if x_open is None:
        raise RuntimeError("Two-state fit failed — cannot continue.")

    chi2_red = reduced_chi2(I_exp, I_fit, sigma_exp)
    print(f"  x_open      = {x_open:.6f} ± {x_err:.6f}")
    print(f"  sigma_scale = {sigma_scale:.6f} ± {sigma_err:.6f}")
    print(f"  chi2_red    = {chi2_red:.6f}")

    # ── Single random draw ────────────────────────────────────────────────
    print("\nDrawing ensemble...")
    I_recon, open_counts, closed_counts = reconstruct_ensemble(
        I_open_all_g, I_closed_all_g,
        x_open, args.n_draws, seed=args.seed
    )

    # ── Save results ──────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, f"{label}_fit_results.txt")
    with open(results_path, "w") as f:
        f.write(f"construct        = {label}\n")
        f.write(f"\n── Deterministic fit ──\n")
        f.write(f"x_open           = {x_open:.6f} ± {x_err:.6f}\n")
        f.write(f"x_closed         = {1-x_open:.6f}\n")
        f.write(f"sigma_scale      = {sigma_scale:.6f} ± {sigma_err:.6f}\n")
        f.write(f"chi2_red         = {chi2_red:.6f}\n")
        f.write(f"\n── Ensemble draw ──\n")
        f.write(f"n_draws          = {args.n_draws}\n")
        f.write(f"seed             = {args.seed}\n")
        f.write(f"n_open_frames    = {I_open_all.shape[0]}\n")
        f.write(f"n_closed_frames  = {I_closed_all.shape[0]}\n")
    print(f"\n  Results → {results_path}")

    recon_path = os.path.join(args.output_dir, f"{label}_reconstructed_corrected.dat")
    np.savetxt(
        recon_path,
        np.column_stack([q_exp, I_recon]),
        header="q  I_recon"
    )
    print(f"  Reconstructed profile → {recon_path}")

    # ── Write ensemble trajectory ─────────────────────────────────────────
    traj_path = os.path.join(args.output_dir, f"{label}_ensemble.pdb")
    write_ensemble_trajectory(
        open_pdb_paths, closed_pdb_paths,
        open_counts, closed_counts,
        n_frames=args.n_ensemble,
        output_path=traj_path,
    )

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_saxs(
        q_exp, I_exp, sigma_exp,
        I_open_g, I_closed_g, I_recon,
        sigma_scale, x_open, x_err, chi2_red,
        label,
        os.path.join(args.output_dir, f"{label}_saxs_fit.png")
    )


if __name__ == "__main__":
    main()




# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_full/trial3/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_full_bound/trial3/*.pdb.dat" \
#     --exp         "exp_data/S5E_FL_I(q)_20260307.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 42 \
#     --label       "Dsk2_full" --output-dir fits_v2/trial3/Dsk2_full \
#     >> dsk2_full_pooled.log 2>&1 &

# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_full/trial3/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_full_bound/trial3/*.pdb.dat" \
#     --exp         "exp_data/S5D_I45A_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 42 \
#     --label       "Dsk2_I45A" --output-dir fits_v2/trial3/Dsk2_I45A \
#     >> dsk2_i45A_pooled.log 2>&1 &

# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH2/trial3/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH2_bound/trial3/*.pdb.dat" \
#     --exp         "exp_data/S5C_dHS2_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 42 \
#     --label       "Dsk2_dTH2" --output-dir fits_v2/trial3/Dsk2_dTH2 \
#     >> dsk2_dTH2_pooled.log 2>&1 &

# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH1/trial3/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH1_bound/trial3/*.pdb.dat" \
#     --exp         "exp_data/S5B_dHS1_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 42 \
#     --label       "Dsk2_dTH1" --output-dir fits_v2/trial3/Dsk2_dTH1 \
#     >> dsk2_dTH1_pooled.log 2>&1 &

# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH3/trial3/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH3_bound/trial3/*.pdb.dat" \
#     --exp         "exp_data/S5F_dHS3_I(q)_20260307.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 42 \
#     --label       "Dsk2_dTH3" --output-dir fits_v2/trial3/Dsk2_dTH3 \
#     >> dsk2_dTH3_pooled.log 2>&1 &




# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_full/trial1/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_full_bound/trial1/*.pdb.dat" \
#     --exp         "exp_data/S5E_FL_I(q)_20260307.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 1356 \
#     --label       "Dsk2_full" --output-dir fits_v2/trial1/Dsk2_full \
#     >> dsk2_full_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_full/trial1/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_full_bound/trial1/*.pdb.dat" \
#     --exp         "exp_data/S5D_I45A_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 6521 \
#     --label       "Dsk2_I45A" --output-dir fits_v2/trial1/Dsk2_I45A \
#     >> dsk2_i45A_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH2/trial1/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH2_bound/trial1/*.pdb.dat" \
#     --exp         "exp_data/S5C_dHS2_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 81991 \
#     --label       "Dsk2_dTH2" --output-dir fits_v2/trial1/Dsk2_dTH2 \
#     >> dsk2_dTH2_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH1/trial1/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH1_bound/trial1/*.pdb.dat" \
#     --exp         "exp_data/S5B_dHS1_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 9182 \
#     --label       "Dsk2_dTH1" --output-dir fits_v2/trial1/Dsk2_dTH1 \
#     >> dsk2_dTH1_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH3/trial1/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH3_bound/trial1/*.pdb.dat" \
#     --exp         "exp_data/S5F_dHS3_I(q)_20260307.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 281 \
#     --label       "Dsk2_dTH3" --output-dir fits_v2/trial1/Dsk2_dTH3 \
#     >> dsk2_dTH3_pooled.log 2>&1 &


#     python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_full/trial2/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_full_bound/trial2/*.pdb.dat" \
#     --exp         "exp_data/S5E_FL_I(q)_20260307.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 21228 \
#     --label       "Dsk2_full" --output-dir fits_v2/trial2/Dsk2_full \
#     >> dsk2_full_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_full/trial2/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_full_bound/trial2/*.pdb.dat" \
#     --exp         "exp_data/S5D_I45A_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 33373 \
#     --label       "Dsk2_I45A" --output-dir fits_v2/trial2/Dsk2_I45A \
#     >> dsk2_i45A_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH2/trial2/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH2_bound/trial2/*.pdb.dat" \
#     --exp         "exp_data/S5C_dHS2_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 81908 \
#     --label       "Dsk2_dTH2" --output-dir fits_v2/trial2/Dsk2_dTH2 \
#     >> dsk2_dTH2_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH1/trial2/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH1_bound/trial2/*.pdb.dat" \
#     --exp         "exp_data/S5B_dHS1_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 59102 \
#     --label       "Dsk2_dTH1" --output-dir fits_v2/trial2/Dsk2_dTH1 \
#     >> dsk2_dTH1_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH3/trial2/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH3_bound/trial2/*.pdb.dat" \
#     --exp         "exp_data/S5F_dHS3_I(q)_20260307.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 41789 \
#     --label       "Dsk2_dTH3" --output-dir fits_v2/trial2/Dsk2_dTH3 \
#     >> dsk2_dTH3_pooled.log 2>&1 &

# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_full/trial3/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_full_bound/trial3/*.pdb.dat" \
#     --exp         "exp_data/S5E_FL_I(q)_20260307.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 7463 \
#     --label       "Dsk2_full" --output-dir fits_v2/trial3/Dsk2_full \
#     >> dsk2_full_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_full/trial3/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_full_bound/trial3/*.pdb.dat" \
#     --exp         "exp_data/S5D_I45A_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 13206 \
#     --label       "Dsk2_I45A" --output-dir fits_v2/trial3/Dsk2_I45A \
#     >> dsk2_i45A_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH2/trial3/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH2_bound/trial3/*.pdb.dat" \
#     --exp         "exp_data/S5C_dHS2_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 22902 \
#     --label       "Dsk2_dTH2" --output-dir fits_v2/trial3/Dsk2_dTH2 \
#     >> dsk2_dTH2_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH1/trial3/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH1_bound/trial3/*.pdb.dat" \
#     --exp         "exp_data/S5B_dHS1_I(q)_20260213.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 68990 \
#     --label       "Dsk2_dTH1" --output-dir fits_v2/trial3/Dsk2_dTH1 \
#     >> dsk2_dTH1_pooled.log 2>&1 &
 
# python pool_replicates_rebuild.py \
#     --open-glob   "foxs_profiles/Dsk2_deltaTH3/trial3/*.pdb.dat" \
#     --closed-glob "foxs_profiles/Dsk2_deltaTH3_bound/trial3/*.pdb.dat" \
#     --exp         "exp_data/S5F_dHS3_I(q)_20260307.csv" \
#     --n-draws     50000 --n-ensemble 50000 --seed 311345 \
#     --label       "Dsk2_dTH3" --output-dir fits_v2/trial3/Dsk2_dTH3 \
#     >> dsk2_dTH3_pooled.log 2>&1 &
