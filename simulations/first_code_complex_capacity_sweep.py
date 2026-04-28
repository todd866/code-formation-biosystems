#!/usr/bin/env python3
"""Capacity / bottleneck sweep for the first-code emergence sim.

Whereas first_code_complex_sweep.py varies substrate dimension N at fixed
channel capacity (K codewords, R response programs), this script varies the
channel capacity K at fixed N and R. It directly tests the boundary-code
mechanism: codes form because high-D dynamics must cross a finite-capacity
boundary. As K grows, the bottleneck loosens; the predicted curve is

  - K << R: coarse partition, viability limited by inability to address all
    actions;
  - K ~ R:  best fit between channel capacity and task structure;
  - K >> R: redundant codewords, viability plateaus; selection still beats
    the random-population baseline but with diminishing returns.

The control labelled "random search" is a memoryless random-population
baseline (genotypes redrawn each generation, not best-of-all). It should
track the same shape but flat-lower: random codes can stumble onto useful
partitions when K is small (lucky), but cannot exploit a wider channel
without retention pressure.

Default grid:
  K        in {2, 4, 8, 16, 32}
  N        in {128}
  R        = 8 (held fixed; channel capacity is the only varied resource)
  seeds    in {321, 322, 323, 324, 325}
  conditions: selection, random_search

Output:
  figures/sweep_K_capacity.{png,pdf}
  figures/sweep_K_capacity.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from first_code_complex_sim import Config, World, run_condition


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


# Base config (matches the medium preset). n_codewords (K) and seed are
# overridden per cell. We fix n_actions=8 so the task complexity is constant
# while the channel capacity varies.
BASE_CONFIG = dict(
    latent_dim=10,
    n_actions=8,
    pop_size=160,
    generations=450,
    samples_per_gen=768,
    eval_samples=12000,
    mutation_std=0.045,
    bias_mutation_std=0.018,
    decoder_flip_prob=0.08,
    elite_frac=0.45,
    slow_noise=0.14,
    substrate_noise=0.28,
    future_delta=1.1,
    run_drift=False,            # drift is a degraded baseline; skip it
    run_random_search=True,
)

CONDITIONS = ["selection", "random_search"]


def make_config(tag: str, seed: int, n_dim: int, n_codewords: int, quick: bool) -> Config:
    cfg_kwargs = dict(BASE_CONFIG)
    if quick:
        cfg_kwargs["generations"] = max(100, BASE_CONFIG["generations"] // 4)
        cfg_kwargs["samples_per_gen"] = max(256, BASE_CONFIG["samples_per_gen"] // 2)
        cfg_kwargs["eval_samples"] = max(2048, BASE_CONFIG["eval_samples"] // 4)
    cfg_kwargs["n_dim"] = n_dim
    cfg_kwargs["n_codewords"] = n_codewords
    cfg_kwargs["tag"] = tag
    cfg_kwargs["seed"] = seed
    return Config(**cfg_kwargs)


def best_metrics(result: dict) -> tuple[float, float, float]:
    fit = result["eval_fit"]
    acc = result["eval_acc"]
    ent = result["eval_entropy"]
    best = int(np.argmax(fit))
    return float(fit[best]), float(acc[best]), float(ent[best])


def t95(n_trials: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(n_trials - 1, 1.96)


def run_grid(k_values: list[int], n_values: list[int], seeds: list[int], quick: bool) -> dict:
    grid = {}
    t0 = time.time()
    cells = [(n, k, s) for n in n_values for k in k_values for s in seeds]
    total = len(cells) * len(CONDITIONS)
    cell_index = 0
    for n_dim, k, seed in cells:
        tag = f"capacity_N{n_dim}_K{k}_seed{seed}"
        cfg = make_config(tag=tag, seed=seed, n_dim=n_dim, n_codewords=k, quick=quick)
        world = World(cfg)
        for ci, condition in enumerate(CONDITIONS):
            cell_index += 1
            t_cell = time.time()
            res = run_condition(cfg, world, condition, seed_offset=1000 * (ci + 1))
            via, acc, ent = best_metrics(res)
            secs = time.time() - t_cell
            grid[(n_dim, k, seed, condition)] = {
                "viability": via,
                "accuracy": acc,
                "entropy": ent,
                "runtime_seconds": secs,
            }
            elapsed = time.time() - t0
            print(
                f"[{cell_index:3d}/{total}] N={n_dim:3d} K={k:>3d} seed={seed} "
                f"{condition:<14s} via={via:+.3f} acc={acc:.3f} ent={ent:.3f}  "
                f"({secs:.1f}s, total {elapsed:.0f}s)",
                flush=True,
            )
    return grid


def aggregate(grid: dict, k_values: list[int], n_values: list[int], seeds: list[int]) -> dict:
    agg = {}
    for n_dim in n_values:
        for k in k_values:
            for cond in CONDITIONS:
                arr_via = np.array([grid[(n_dim, k, s, cond)]["viability"] for s in seeds])
                arr_acc = np.array([grid[(n_dim, k, s, cond)]["accuracy"] for s in seeds])
                arr_ent = np.array([grid[(n_dim, k, s, cond)]["entropy"] for s in seeds])
                ntrials = len(seeds)
                tt = t95(ntrials)
                agg[(n_dim, k, cond)] = {
                    "via_mean": float(arr_via.mean()),
                    "via_ci": float(tt * arr_via.std(ddof=1) / np.sqrt(ntrials)) if ntrials > 1 else 0.0,
                    "acc_mean": float(arr_acc.mean()),
                    "acc_ci": float(tt * arr_acc.std(ddof=1) / np.sqrt(ntrials)) if ntrials > 1 else 0.0,
                    "ent_mean": float(arr_ent.mean()),
                    "ent_ci": float(tt * arr_ent.std(ddof=1) / np.sqrt(ntrials)) if ntrials > 1 else 0.0,
                    "n_trials": ntrials,
                }
    return agg


def write_summary(grid, agg, k_values, n_values, seeds, quick, runtime_seconds) -> Path:
    lines = [
        "# Capacity / bottleneck sweep for first-code emergence",
        f"# quick={quick}  runtime_seconds={runtime_seconds:.1f}",
        f"# K_values={k_values}  N_values={n_values}  seeds={seeds}",
        f"# conditions={CONDITIONS}",
        f"# base_config={json.dumps(BASE_CONFIG)}",
        "",
        "## Per-run metrics",
        f"{'N':>4} {'K':>4} {'seed':>5} {'condition':<14} {'viability':>10} {'accuracy':>10} {'entropy':>10} {'secs':>7}",
    ]
    for n_dim in n_values:
        for k in k_values:
            for s in seeds:
                for cond in CONDITIONS:
                    r = grid[(n_dim, k, s, cond)]
                    lines.append(
                        f"{n_dim:>4} {k:>4} {s:>5} {cond:<14} {r['viability']:>10.4f} "
                        f"{r['accuracy']:>10.4f} {r['entropy']:>10.4f} {r['runtime_seconds']:>7.1f}"
                    )

    lines.append("")
    lines.append("## Aggregate (mean +/- 95% CI across seeds)")
    lines.append(
        f"{'N':>4} {'K':>4} {'condition':<14} {'via_mean':>10} {'via_ci':>10} "
        f"{'acc_mean':>10} {'acc_ci':>10} {'ent_mean':>10} {'ent_ci':>10}"
    )
    for n_dim in n_values:
        for k in k_values:
            for cond in CONDITIONS:
                a = agg[(n_dim, k, cond)]
                lines.append(
                    f"{n_dim:>4} {k:>4} {cond:<14} {a['via_mean']:>10.4f} {a['via_ci']:>10.4f} "
                    f"{a['acc_mean']:>10.4f} {a['acc_ci']:>10.4f} {a['ent_mean']:>10.4f} {a['ent_ci']:>10.4f}"
                )
    out = FIG_DIR / "sweep_K_capacity.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


def plot_capacity(agg: dict, k_values: list[int], n_values: list[int], n_seeds: int) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    palette = {"selection": "#1f7770", "random_search": "#52616b"}
    labels = {"selection": "selection", "random_search": "random-population baseline"}
    markers = {"selection": "o", "random_search": "^"}

    for which, ax, ylabel, title in [
        ("via", axes[0], "normalized viability\n(0 = mean action, 1 = oracle)",
         f"(a) Viability vs channel capacity"),
        ("acc", axes[1], "best-of-population action accuracy",
         f"(b) Oracle-action agreement"),
        ("ent", axes[2], "best-of-population message entropy\n(normalized to log2(K))",
         f"(c) Codeword usage entropy"),
    ]:
        for cond in CONDITIONS:
            for n_dim in n_values:
                means = np.array([agg[(n_dim, k, cond)][f"{which}_mean"] for k in k_values])
                cis = np.array([agg[(n_dim, k, cond)][f"{which}_ci"] for k in k_values])
                style = "-" if n_dim == n_values[0] else "--"
                ax.fill_between(k_values, means - cis, means + cis, color=palette[cond], alpha=0.12)
                ax.plot(
                    k_values, means,
                    color=palette[cond], lw=2, ls=style, marker=markers[cond], ms=6,
                    label=f"{labels[cond]} (N={n_dim})" if len(n_values) > 1 else labels[cond],
                )

        ax.set_xscale("log", base=2)
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel("channel capacity $K$ (codewords)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, ls=":", alpha=0.3, which="both")
        ax.legend(loc="best", frameon=False, fontsize=8)

    n_actions = BASE_CONFIG["n_actions"]
    axes[1].axhline(1.0 / n_actions, color="#27313a", lw=1, ls=":", alpha=0.7)
    axes[1].text(k_values[-1], 1.0 / n_actions + 0.012, "chance", ha="right", va="bottom",
                 fontsize=8, color="#27313a")
    axes[1].set_ylim(0, 1.0)
    axes[0].axvline(n_actions, color="#27313a", lw=1, ls=":", alpha=0.4)
    axes[0].text(n_actions, axes[0].get_ylim()[0] + 0.02, f" $K=R={n_actions}$",
                 fontsize=8, color="#27313a", ha="left", va="bottom")

    fig.suptitle(f"Channel capacity sweep, fixed $R=8$ actions, mean +/- 95% CI ({n_seeds} seeds)",
                 fontsize=11, y=1.02)
    fig.tight_layout(pad=1.0)
    out_pdf = FIG_DIR / "sweep_K_capacity.pdf"
    out_png = FIG_DIR / "sweep_K_capacity.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", default="2,4,8,16,32", type=str)
    parser.add_argument("--n-values", default="128", type=str)
    parser.add_argument("--seeds", default="321,322,323,324,325", type=str)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = [int(s) for s in args.k_values.split(",") if s.strip()]
    n_values = [int(s) for s in args.n_values.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"capacity sweep: K={k_values} N={n_values} seeds={seeds} quick={args.quick}")
    t0 = time.time()
    grid = run_grid(k_values, n_values, seeds, quick=args.quick)
    runtime = time.time() - t0
    agg = aggregate(grid, k_values, n_values, seeds)
    summary_path = write_summary(grid, agg, k_values, n_values, seeds, args.quick, runtime)
    fig_path = plot_capacity(agg, k_values, n_values, n_seeds=len(seeds))
    print(f"\nwrote {summary_path}")
    print(f"wrote {fig_path}")
    print(f"total runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
