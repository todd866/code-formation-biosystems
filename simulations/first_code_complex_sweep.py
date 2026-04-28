#!/usr/bin/env python3
"""Grid sweep over substrate dimension and seed for the first-code emergence sim.

Imports the simulation harness from ``first_code_complex_sim.py`` and runs a
(N, seed, condition) grid. For each cell, evolves selection, drift, and a
memoryless random-population baseline (genotypes redrawn each generation; not
a best-of-all matched random search) and records final-evaluation metrics.
Saves per-run summaries plus an aggregate scaling figure with mean and 95% CI
bands across seeds for each condition.

Default grid:
  N        in {64, 128, 256, 512}
  seeds    in {321, 322, 323, 324, 325}
  conditions: selection, drift, random_search

Output:
  figures/sweep_N_scaling.{png,pdf}
  figures/sweep_N_scaling.txt   (per-run table + per-N aggregate)

Runtime estimate
----------------
medium preset (450 generations, pop 160, samples 768): roughly 20s per condition
at N=128, scaling roughly linearly with N. So
  4 * 5 * 3 = 60 condition runs.
At an average ~30s each, ~30 min total. Use --quick to halve generations and
sample counts for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the harness module importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from first_code_complex_sim import Config, World, run_condition


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


# Base config matching the "medium" preset, with n_dim and seed overridden per cell.
BASE_CONFIG = dict(
    latent_dim=10,
    n_actions=8,
    n_codewords=8,
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
    run_drift=True,
    run_random_search=True,
)


CONDITIONS = ["selection", "drift", "random_search"]


def make_config(tag: str, seed: int, n_dim: int, quick: bool) -> Config:
    cfg_kwargs = dict(BASE_CONFIG)
    if quick:
        cfg_kwargs["generations"] = max(80, BASE_CONFIG["generations"] // 4)
        cfg_kwargs["samples_per_gen"] = max(256, BASE_CONFIG["samples_per_gen"] // 2)
        cfg_kwargs["eval_samples"] = max(2048, BASE_CONFIG["eval_samples"] // 4)
    cfg_kwargs["n_dim"] = n_dim
    cfg_kwargs["tag"] = tag
    cfg_kwargs["seed"] = seed
    return Config(**cfg_kwargs)


def best_metrics(result: dict) -> tuple[float, float, float]:
    fit = result["eval_fit"]
    acc = result["eval_acc"]
    ent = result["eval_entropy"]
    best = int(np.argmax(fit))
    return float(fit[best]), float(acc[best]), float(ent[best])


def run_grid(n_values: list[int], seeds: list[int], quick: bool) -> dict:
    """Run the full grid; return a nested dict of results keyed by (n_dim, seed, condition)."""
    grid_results = {}
    t0 = time.time()
    total_cells = len(n_values) * len(seeds) * len(CONDITIONS)
    cell_index = 0
    for n_dim in n_values:
        for seed in seeds:
            tag = f"sweep_N{n_dim}_seed{seed}"
            cfg = make_config(tag=tag, seed=seed, n_dim=n_dim, quick=quick)
            world = World(cfg)
            for ci, condition in enumerate(CONDITIONS):
                cell_index += 1
                t_cell = time.time()
                result = run_condition(cfg, world, condition, seed_offset=1000 * (ci + 1))
                viability, accuracy, entropy = best_metrics(result)
                cell_secs = time.time() - t_cell
                elapsed = time.time() - t0
                grid_results[(n_dim, seed, condition)] = {
                    "viability": viability,
                    "accuracy": accuracy,
                    "entropy": entropy,
                    "runtime_seconds": cell_secs,
                }
                print(
                    f"[{cell_index:3d}/{total_cells}] N={n_dim:4d} seed={seed} "
                    f"{condition:<14s} via={viability:+.3f} acc={accuracy:.3f} "
                    f"ent={entropy:.3f}  ({cell_secs:.1f}s, total {elapsed:.0f}s)",
                    flush=True,
                )
    return grid_results


def aggregate(grid_results: dict, n_values: list[int], seeds: list[int]) -> dict:
    """Aggregate per-cell results into per-(N, condition) mean and 95% CI."""
    agg = {}
    for n_dim in n_values:
        for condition in CONDITIONS:
            via = np.array([grid_results[(n_dim, seed, condition)]["viability"] for seed in seeds])
            acc = np.array([grid_results[(n_dim, seed, condition)]["accuracy"] for seed in seeds])
            ent = np.array([grid_results[(n_dim, seed, condition)]["entropy"] for seed in seeds])
            n_trials = len(seeds)
            # 95% CI from the sample SD (small-sample t-approximation factor).
            t95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}.get(
                n_trials - 1, 1.96
            )
            agg[(n_dim, condition)] = {
                "via_mean": float(via.mean()),
                "via_sd": float(via.std(ddof=1)) if n_trials > 1 else 0.0,
                "via_ci": float(t95 * via.std(ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0,
                "acc_mean": float(acc.mean()),
                "acc_sd": float(acc.std(ddof=1)) if n_trials > 1 else 0.0,
                "acc_ci": float(t95 * acc.std(ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0,
                "ent_mean": float(ent.mean()),
                "ent_sd": float(ent.std(ddof=1)) if n_trials > 1 else 0.0,
                "n_trials": n_trials,
            }
    return agg


def write_summary(
    grid_results: dict,
    agg: dict,
    n_values: list[int],
    seeds: list[int],
    quick: bool,
    runtime_seconds: float,
) -> Path:
    lines = [
        "# Grid sweep over (N, seed, condition) for first-code emergence",
        f"# quick={quick}  runtime_seconds={runtime_seconds:.1f}",
        f"# N_values={n_values}",
        f"# seeds={seeds}",
        f"# conditions={CONDITIONS}",
        f"# base_config={json.dumps(BASE_CONFIG)}",
        "",
        "## Per-run metrics",
        f"{'N':>5} {'seed':>5} {'condition':<14} {'viability':>10} {'accuracy':>10} {'entropy':>10} {'secs':>8}",
    ]
    for n_dim in n_values:
        for seed in seeds:
            for cond in CONDITIONS:
                r = grid_results[(n_dim, seed, cond)]
                lines.append(
                    f"{n_dim:>5} {seed:>5} {cond:<14} {r['viability']:>10.4f} {r['accuracy']:>10.4f} "
                    f"{r['entropy']:>10.4f} {r['runtime_seconds']:>8.1f}"
                )

    lines.append("")
    lines.append("## Aggregate (mean +/- 95% CI across seeds)")
    lines.append(
        f"{'N':>5} {'condition':<14} {'via_mean':>10} {'via_ci':>10} {'acc_mean':>10} {'acc_ci':>10} "
        f"{'ent_mean':>10}"
    )
    for n_dim in n_values:
        for cond in CONDITIONS:
            a = agg[(n_dim, cond)]
            lines.append(
                f"{n_dim:>5} {cond:<14} {a['via_mean']:>10.4f} {a['via_ci']:>10.4f} "
                f"{a['acc_mean']:>10.4f} {a['acc_ci']:>10.4f} {a['ent_mean']:>10.4f}"
            )

    out = FIG_DIR / "sweep_N_scaling.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


def plot_scaling(agg: dict, n_values: list[int], n_seeds: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    palette = {
        "selection": "#1f7770",
        "drift": "#b65a34",
        "random_search": "#52616b",
    }
    labels = {
        "selection": "selection",
        "drift": "no selection (drift)",
        "random_search": "random-population baseline",
    }
    markers = {"selection": "o", "drift": "s", "random_search": "^"}

    for which, ax, ylabel, title in [
        ("via", axes[0], "normalized viability\n(0 = mean action, 1 = oracle action)",
         f"(a) Viability vs substrate dimension (mean +/- 95% CI, {n_seeds} seeds)"),
        ("acc", axes[1], "best-of-population action accuracy",
         f"(b) Oracle-action agreement vs substrate dimension"),
    ]:
        for cond in CONDITIONS:
            means = np.array([agg[(n, cond)][f"{which}_mean"] for n in n_values])
            cis = np.array([agg[(n, cond)][f"{which}_ci"] for n in n_values])
            ax.fill_between(n_values, means - cis, means + cis, color=palette[cond], alpha=0.15)
            ax.plot(
                n_values, means,
                color=palette[cond], lw=2, marker=markers[cond], ms=6,
                label=labels[cond],
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.set_xlabel("substrate dimension $N$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, ls=":", alpha=0.3, which="both")
        ax.legend(loc="lower left", frameon=False, fontsize=8)

    # accuracy axis: chance baseline
    n_actions = BASE_CONFIG["n_actions"]
    axes[1].axhline(1.0 / n_actions, color="#27313a", lw=1, ls=":", alpha=0.7)
    axes[1].text(n_values[-1], 1.0 / n_actions + 0.01, "chance", ha="right", va="bottom",
                 fontsize=8, color="#27313a")
    axes[1].set_ylim(0, 1.0)

    fig.tight_layout(pad=1.1)
    out_pdf = FIG_DIR / "sweep_N_scaling.pdf"
    out_png = FIG_DIR / "sweep_N_scaling.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-values",
        type=str,
        default="64,128,256,512",
        help="comma-separated list of substrate dimensions",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="321,322,323,324,325",
        help="comma-separated list of seeds",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduce generations and sample counts for a quick smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = [int(s) for s in args.n_values.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"running grid: N={n_values} seeds={seeds} quick={args.quick}")
    t0 = time.time()
    grid_results = run_grid(n_values, seeds, quick=args.quick)
    runtime = time.time() - t0
    agg = aggregate(grid_results, n_values, seeds)
    summary_path = write_summary(grid_results, agg, n_values, seeds, args.quick, runtime)
    fig_path = plot_scaling(agg, n_values, n_seeds=len(seeds))
    print(f"\nwrote {summary_path}")
    print(f"wrote {fig_path}")
    print(f"total runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
