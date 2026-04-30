"""Post-hoc analysis for the volumetric hourglass sweep (single-waist).

Reads runs/<timestamp>_volumetric_sweep/<condition>/aggregate_history.npz
files and produces figures into runs/<timestamp>_volumetric_sweep/figures/.

Usage:
    python3 volumetric_hourglass_analyze.py [sweep_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def find_latest_sweep(base: Path) -> Path:
    runs = [p for p in base.iterdir()
            if p.is_dir() and p.name.endswith("_volumetric_sweep")]
    if not runs:
        raise FileNotFoundError("no *_volumetric_sweep/ directories under runs/")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_condition(cdir: Path):
    cfg_path = cdir / "config.json"
    agg_path = cdir / "aggregate_history.npz"
    if not cfg_path.exists() or not agg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    agg = dict(np.load(agg_path))
    return {"name": cdir.name, "cfg": cfg, "agg": agg, "dir": cdir}


def safe_mean(x, axis=0):
    return np.nanmean(x, axis=axis)


def safe_sem(x, axis=0):
    n = x.shape[axis]
    return np.nanstd(x, axis=axis, ddof=1) / max(1, np.sqrt(max(1, n - 1)))


def waist_index(width_profile):
    return int(np.argmin(width_profile))


def fig_learning_curves(conditions, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    cmap = mpl.cm.viridis
    n = len(conditions)
    for i, c in enumerate(conditions):
        col = cmap(i / max(1, n - 1))
        agg = c["agg"]
        it = agg["iter"][0]
        acc = safe_mean(agg["acc"], 0)
        acc_s = safe_sem(agg["acc"], 0)
        loss = safe_mean(agg["loss"], 0)
        mi = safe_mean(agg["mi_F_waist"], 0)
        K = c["cfg"]["waist_k"]
        label = f"K={K}"
        axes[0].plot(it, acc, color=col, label=label)
        axes[0].fill_between(it, acc - acc_s, acc + acc_s, color=col, alpha=0.15)
        axes[1].plot(it, loss, color=col, label=label)
        axes[2].plot(it, mi, color=col, label=label)
    R = conditions[0]["cfg"]["r_task"]
    axes[0].axhline(1.0 / R, color="k", lw=0.6, ls=":", label=f"chance (1/R={1/R:.3f})")
    axes[0].axhline(1.0, color="k", lw=0.4, ls=":", alpha=0.5)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("future accuracy")
    axes[0].set_ylim(0, 1); axes[0].set_title("Learning curves"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("cross-entropy loss")
    axes[1].set_title("Loss")
    axes[2].axhline(np.log2(R), color="k", lw=0.6, ls=":", label=f"$\\log_2 R$={np.log2(R):.2f}")
    axes[2].set_xlabel("iteration"); axes[2].set_ylabel(r"$I(M_{\rm waist}; F)$ (bits)")
    axes[2].set_title("Waist task information"); axes[2].legend(fontsize=7)
    fig.savefig(out_path, dpi=140); plt.close(fig)


def fig_radial_d_eff(conditions, out_path):
    """D_eff(depth) at the per-seed peak-accuracy iteration: volumetric collapse + re-expansion."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    cmap = mpl.cm.viridis
    n = len(conditions)
    for i, c in enumerate(conditions):
        col = cmap(i / max(1, n - 1))
        K = c["cfg"]["waist_k"]
        # Per-seed peak iteration; index d_eff at that iter.
        peak_idx = np.argmax(c["agg"]["acc"], axis=1)        # (n_seeds,)
        n_seeds = c["agg"]["acc"].shape[0]
        d_eff = c["agg"]["d_eff"][np.arange(n_seeds), peak_idx, :]   # (n_seeds, n_depths)
        m = safe_mean(d_eff, 0)
        s = safe_sem(d_eff, 0)
        depths = np.arange(m.shape[0])
        axes[0].plot(depths, m, "-o", color=col, label=f"K={K}", lw=1.5, ms=3)
        axes[0].fill_between(depths, m - s, m + s, color=col, alpha=0.10)
        axes[1].plot(depths, m, "-o", color=col, label=f"K={K}", lw=1.5, ms=3)
        axes[1].fill_between(depths, m - s, m + s, color=col, alpha=0.10)
    # Mark the waist depth (waist_state index in states list = wi + 1, since state[0]=bulk)
    wi = waist_index(conditions[0]["cfg"]["width_profile"])
    waist_state_idx = wi + 1
    for ax in axes:
        ax.axvline(waist_state_idx, color="k", lw=0.6, ls=":", alpha=0.4)
        ax.set_xlabel("radial depth (state index)")
        ax.set_ylabel(r"$D_{\rm eff}$ (participation ratio)")
    axes[0].set_title("Radial $D_{\\rm eff}$ profile (linear, at peak)")
    axes[1].set_yscale("log"); axes[1].set_title("Radial $D_{\\rm eff}$ profile (log, at peak)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.savefig(out_path, dpi=140); plt.close(fig)


def fig_waist_floor(conditions, out_path):
    """Prop-6 cardinality floor: peak accuracy across seeds vs nominal K.

    We report PEAK accuracy across training because Gumbel-Softmax with finite
    tau_min has late-stage training instability — many seeds reach near-task-
    limit peaks then collapse to lower-K_eff basins. Peak captures the
    architecture's achievable capacity (the relevant quantity for Prop 6),
    while final captures where SGD landed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    Ks = [c["cfg"]["waist_k"] for c in conditions]
    R = conditions[0]["cfg"]["r_task"]
    peak_acc_m, peak_acc_s = [], []
    final_acc_m, final_acc_s = [], []
    peak_mi_m, peak_mi_s = [], []
    peak_K_eff_m, peak_K_eff_s = [], []
    for c in conditions:
        acc_traj = c["agg"]["acc"]                         # (n_seeds, T)
        H_traj = c["agg"]["H_waist"]                       # (n_seeds, T)
        mi_traj = c["agg"]["mi_F_waist"]                   # (n_seeds, T)
        K = c["cfg"]["waist_k"]
        # peak per seed = max over time
        peak_idx = np.argmax(acc_traj, axis=1)             # (n_seeds,)
        n_seeds = acc_traj.shape[0]
        peak_acc = acc_traj[np.arange(n_seeds), peak_idx]
        peak_H = H_traj[np.arange(n_seeds), peak_idx]
        peak_mi = mi_traj[np.arange(n_seeds), peak_idx]
        peak_K_eff = 2 ** (peak_H * np.log2(max(K, 2)))
        final_acc = acc_traj[:, -1]
        peak_acc_m.append(safe_mean(peak_acc));   peak_acc_s.append(safe_sem(peak_acc))
        final_acc_m.append(safe_mean(final_acc)); final_acc_s.append(safe_sem(final_acc))
        peak_mi_m.append(safe_mean(peak_mi));     peak_mi_s.append(safe_sem(peak_mi))
        peak_K_eff_m.append(safe_mean(peak_K_eff)); peak_K_eff_s.append(safe_sem(peak_K_eff))
    Ks_arr = np.asarray(Ks)
    peak_acc_m = np.asarray(peak_acc_m); peak_acc_s = np.asarray(peak_acc_s)
    final_acc_m = np.asarray(final_acc_m); final_acc_s = np.asarray(final_acc_s)
    peak_mi_m = np.asarray(peak_mi_m); peak_mi_s = np.asarray(peak_mi_s)
    peak_K_eff_m = np.asarray(peak_K_eff_m); peak_K_eff_s = np.asarray(peak_K_eff_s)
    # (a) accuracy vs K (peak + final)
    axes[0].errorbar(Ks_arr, peak_acc_m, yerr=peak_acc_s, fmt="-o", lw=1.5,
                     ms=5, label="peak (achievable)", color="C0")
    axes[0].errorbar(Ks_arr, final_acc_m, yerr=final_acc_s, fmt="--s", lw=1.0,
                     ms=4, label="final (where SGD landed)", color="C1", alpha=0.6)
    axes[0].axvline(R, color="k", lw=0.6, ls="--", label=f"R={R}")
    axes[0].axhline(1 / R, color="k", lw=0.5, ls=":", label="chance")
    axes[0].plot(Ks_arr, np.minimum(Ks_arr / R, 1.0), "k--", alpha=0.5,
                 label="Prop-6 floor: min(K/R, 1)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("waist alphabet $K$"); axes[0].set_ylabel("future-task accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("(a) Cardinality floor vs nominal K")
    axes[0].legend(fontsize=8)
    # (b) peak MI vs K
    axes[1].errorbar(Ks_arr, peak_mi_m, yerr=peak_mi_s, fmt="-o", lw=1.5, ms=5)
    axes[1].axvline(R, color="k", lw=0.6, ls="--")
    axes[1].axhline(np.log2(R), color="k", lw=0.6, ls=":", label="$\\log_2 R$")
    axes[1].plot(Ks_arr, np.minimum(np.log2(np.maximum(Ks_arr, 2)), np.log2(R)),
                 "k--", alpha=0.5, label=r"$\min(\log_2 K, \log_2 R)$")
    axes[1].set_xscale("log", base=2); axes[1].set_xlabel("waist alphabet $K$")
    axes[1].set_ylabel(r"peak $I(M_{\rm waist}; F)$ (bits)")
    axes[1].set_title("(b) Task information saturates at $\\log_2 R$")
    axes[1].legend(fontsize=8)
    # (c) effective K used vs nominal K
    axes[2].errorbar(Ks_arr, peak_K_eff_m, yerr=peak_K_eff_s, fmt="-o", lw=1.5, ms=5)
    axes[2].plot(Ks_arr, Ks_arr, "k:", alpha=0.5, label="K (full use)")
    axes[2].axhline(R, color="k", lw=0.6, ls="--", label=f"R={R}")
    axes[2].set_xscale("log", base=2); axes[2].set_yscale("log", base=2)
    axes[2].set_xlabel("waist alphabet $K$"); axes[2].set_ylabel(r"$K_{\rm eff}=2^{H(M)}$ at peak")
    axes[2].set_title("(c) Used codeword count saturates at R")
    axes[2].legend(fontsize=8)
    fig.savefig(out_path, dpi=140); plt.close(fig)


def fig_keff_collapse(conditions, out_path):
    """The cleanest Prop-6 figure: scatter accuracy vs K_eff for every seed in
    every condition. All points should fall on the same curve regardless of
    nominal K, showing that the binding constraint is K_eff (the codes
    actually used), not K (the codes available).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    R = conditions[0]["cfg"]["r_task"]
    cmap = mpl.cm.viridis
    for i, c in enumerate(conditions):
        col = cmap(i / max(1, len(conditions) - 1))
        K = c["cfg"]["waist_k"]
        acc_traj = c["agg"]["acc"]
        H_traj = c["agg"]["H_waist"]
        # for each seed, take the iteration with peak accuracy and read K_eff
        peak_idx = np.argmax(acc_traj, axis=1)
        n_seeds = acc_traj.shape[0]
        peak_acc = acc_traj[np.arange(n_seeds), peak_idx]
        peak_H = H_traj[np.arange(n_seeds), peak_idx]
        peak_K_eff = 2 ** (peak_H * np.log2(max(K, 2)))
        ax.scatter(peak_K_eff, peak_acc, s=70, color=col,
                   label=f"K_nom={K}", edgecolor="black", lw=0.5, zorder=3)
    # Theoretical Prop-6 curves for reference
    K_grid = np.logspace(0, np.log2(2048), 200, base=2)
    ax.plot(K_grid, np.minimum(K_grid / R, 1.0), "k--", alpha=0.5,
            label=r"Prop-6 floor: $\min(K_{\rm eff}/R, 1)$")
    # task-intrinsic limit observed at ~0.78
    task_limit = 0.78
    ax.axhline(task_limit, color="C3", lw=1, ls=":",
               label=fr"task limit $\approx {task_limit:.2f}$ (drift noise)")
    ax.axvline(R, color="k", lw=0.6, ls="--", alpha=0.5, label=f"R={R}")
    ax.axhline(1 / R, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"effective alphabet $K_{\rm eff} = 2^{H(M_{\rm waist})}$ at peak")
    ax.set_ylabel("peak future-task accuracy")
    ax.set_title("Prop-6 cardinality floor in $K_{\\rm eff}$ collapses all conditions")
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out_path, dpi=140); plt.close(fig)


def fig_phase_modes(conditions, out_path):
    """For each condition, MI(M_waist; phase_k) — does longest-tau survive?"""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    Ks = [c["cfg"]["waist_k"] for c in conditions]
    n_modes = conditions[0]["agg"]["mi_phase_waist"].shape[-1]
    taus = conditions[0]["cfg"]["phase_taus"]
    cmap = mpl.cm.plasma
    width = 0.8 / n_modes
    x = np.arange(len(conditions))
    # Per-seed peak iteration; index mi_phase_waist at that iter.
    peak_arrays = []
    for c in conditions:
        peak_idx = np.argmax(c["agg"]["acc"], axis=1)        # (n_seeds,)
        n_seeds = c["agg"]["acc"].shape[0]
        # mi_phase_waist shape: (n_seeds, T, n_modes)
        peak_mi = c["agg"]["mi_phase_waist"][np.arange(n_seeds), peak_idx, :]   # (n_seeds, n_modes)
        peak_arrays.append(peak_mi)
    for k in range(n_modes):
        col = cmap(k / max(1, n_modes - 1))
        m = np.array([safe_mean(arr[:, k]) for arr in peak_arrays])
        s = np.array([safe_sem(arr[:, k]) for arr in peak_arrays])
        ax.bar(x + (k - n_modes / 2 + 0.5) * width, m, width=width,
               yerr=s, color=col, label=fr"$\tau_{k}={taus[k]:.0f}$")
    ax.set_xticks(x); ax.set_xticklabels([f"K={K}" for K in Ks], rotation=30)
    ax.set_ylabel(r"$I(M_{\rm waist}; \phi_k)$ (bits)")
    ax.set_title("Task-relevance selectivity at the waist (only $\\phi_3$ is task-relevant)")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=140); plt.close(fig)


def fig_d_eff_dynamics(conditions, out_path):
    """Heatmap: D_eff(depth, iteration) for each condition."""
    n = len(conditions)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n),
                             constrained_layout=True, sharex=True)
    if n == 1:
        axes = [axes]
    waist_idx = waist_index(conditions[0]["cfg"]["width_profile"])
    for ax, c in zip(axes, conditions):
        d_eff = c["agg"]["d_eff"]                  # (n_seeds, T, n_depths)
        m = safe_mean(d_eff, 0)                    # (T, n_depths)
        it = c["agg"]["iter"][0]
        im = ax.imshow(
            m.T, aspect="auto", origin="lower",
            extent=[it[0], it[-1], -0.5, m.shape[1] - 0.5],
            cmap="magma", norm=mpl.colors.LogNorm(vmin=max(0.5, m.min())),
        )
        ax.axhline(waist_idx + 0.5, color="cyan", lw=0.6, ls=":", alpha=0.7)
        K = c["cfg"]["waist_k"]
        ax.set_title(f"K={K}: $D_{{\\rm eff}}(r,t)$", fontsize=10)
        ax.set_ylabel("depth")
        plt.colorbar(im, ax=ax, label=r"$D_{\rm eff}$")
    axes[-1].set_xlabel("iteration")
    fig.savefig(out_path, dpi=140); plt.close(fig)


def main():
    base = Path(__file__).resolve().parent.parent / "runs"
    if len(sys.argv) > 1:
        sweep_dir = Path(sys.argv[1])
    else:
        sweep_dir = find_latest_sweep(base)
    print(f"sweep_dir = {sweep_dir}")
    fig_dir = sweep_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    conditions = []
    for cdir in sorted(sweep_dir.iterdir()):
        if not cdir.is_dir():
            continue
        c = load_condition(cdir)
        if c is None:
            print(f"  skip (incomplete): {cdir.name}")
            continue
        n_seeds = c["agg"]["acc"].shape[0]
        n_meas = c["agg"]["acc"].shape[1]
        K = c["cfg"]["waist_k"]
        print(f"  {cdir.name}: K={K}, {n_seeds} seeds × {n_meas} measurements")
        conditions.append(c)
    if not conditions:
        print("no completed conditions; nothing to analyze yet")
        return
    # sort by K
    conditions.sort(key=lambda c: c["cfg"]["waist_k"])
    print("generating figures...")
    fig_learning_curves(conditions, fig_dir / "fig_learning_curves.pdf")
    fig_radial_d_eff(conditions, fig_dir / "fig_radial_d_eff.pdf")
    fig_waist_floor(conditions, fig_dir / "fig_waist_floor.pdf")
    fig_keff_collapse(conditions, fig_dir / "fig_keff_collapse.pdf")
    fig_phase_modes(conditions, fig_dir / "fig_phase_modes.pdf")
    fig_d_eff_dynamics(conditions, fig_dir / "fig_d_eff_dynamics.pdf")
    print(f"figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
