#!/usr/bin/env python3
"""Stochastic-resonance sweep for the bulk-to-bulk synchronization thesis.

Tests the prediction that biological codes are finite boundary handles by which
two high-dimensional bulks remain mutually synchronized, with the synchronization
itself sustained through stochastic-resonant entrainment to a slow environmental
signal.

Setup
-----
  E_t                      : slow environmental signal, cos(omega_E t).
  B1: N1 Kuramoto oscillators, weakly forced by E, with noise std sigma.
  M_t = discretize(mean phase of B1) into K codewords.
  B2: N2 Kuramoto oscillators, driven by codeword-keyed target phases
      (selected to align with E shifted by Delta).
  B2's mean phase is read out as the receiver's prediction of E_{t+Delta}.

Sweep
-----
  Noise amplitude sigma over a log-spaced range.
  For each sigma, four conditions:
    * SELECTED      M_t correctly tracks B1 phase; B2 targets phase aligned
                    with E shifted by Delta.
    * RANDOM        Codeword target phases are randomized.
    * SHUFFLED_EVAL Simulation runs with the codeword forcing intact; B2's
                    mean-phase trajectory is then shuffled in time before
                    scoring. This is a synchronization-only ablation: the
                    (B1, B2, E) phase relationship is destroyed at readout
                    while the codeword sequence and B1 dynamics are
                    unchanged. It is not a causal shuffle of the codeword
                    input fed to B2.
    * NO_CODE       B2 receives no codeword forcing.

Predictions
-----------
  - SELECTED viability and phase-locking peak at intermediate sigma
    (stochastic resonance signature).
  - RANDOM, SHUFFLED_EVAL, NO_CODE remain low across all sigma.

Outputs
-------
  figures/stochastic_resonance.{pdf,png}
  figures/stochastic_resonance.txt
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


# Hyperparameters. To produce a stochastic-resonance signature rather than a
# deterministic-locking signature, B1's intrinsic frequency is mismatched to
# the environmental signal so that weak K_E forcing alone cannot entrain;
# noise enhances entrainment up to an optimum, beyond which it destroys it.
N1 = 64                     # B1 oscillator count
N2 = 64                     # B2 oscillator count
OMEGA_E = 0.04              # environmental slow frequency (radians per dt)
OMEGA_BAR_1 = 0.075         # B1 mean intrinsic frequency (mismatched to E)
OMEGA_BAR_2 = 0.04          # B2 mean intrinsic frequency
SPREAD = 0.005              # per-oscillator frequency spread (tight)
K_INTRA_1 = 0.20            # internal coupling within B1 (low: each osc more
                            # individually noise-driven)
K_INTRA_2 = 0.50            # internal coupling within B2 (high: tight readout)
K_E = 0.030                 # weak E -> B1 coupling (just below the
                            # deterministic locking threshold |omega_diff|=0.035)
K_M = 0.55                  # code -> B2 coupling
N_CODEWORDS = 4             # codeword alphabet size
DELTA = np.pi / 2           # future horizon (radians)
T_STEPS = 4000              # timesteps per run (long for cleaner statistics)
T_BURN = 800                # discard initial transient
DT = 1.0
SIGMAS = np.array([0.005, 0.02, 0.05, 0.10, 0.18, 0.30, 0.50, 0.80, 1.3, 2.5])
SEEDS = [11, 12, 13, 14, 15, 16, 17]
CONDITIONS = ["selected", "random", "shuffled_eval", "no_code"]


def codeword_target_phases(rng: np.random.Generator, mode: str) -> np.ndarray:
    """K target phases for B2, keyed by codeword index.

    SELECTED: codeword k centered at phase k * 2pi/K, shifted forward by DELTA
              so that B2 leads B1 by Delta and tracks E_{t+Delta}.
    RANDOM:   target phases drawn uniform on [0, 2pi).
    """
    if mode == "random":
        return rng.uniform(0, 2 * np.pi, size=N_CODEWORDS)
    bin_centers = (np.arange(N_CODEWORDS) + 0.5) * (2 * np.pi / N_CODEWORDS)
    return (bin_centers + DELTA) % (2 * np.pi)


def simulate(sigma: float, condition: str, seed: int) -> dict:
    """Simulate the coupled system and return summary metrics."""
    rng = np.random.default_rng(seed)

    omega_1 = OMEGA_BAR_1 + rng.normal(0, SPREAD, size=N1)
    omega_2 = OMEGA_BAR_2 + rng.normal(0, SPREAD, size=N2)

    phi_1 = rng.uniform(0, 2 * np.pi, size=N1)
    phi_2 = rng.uniform(0, 2 * np.pi, size=N2)

    target_phases = codeword_target_phases(
        rng,
        mode="random" if condition == "random" else "selected",
    )

    history_b1_mean = np.zeros(T_STEPS)
    history_b2_mean = np.zeros(T_STEPS)
    history_msg = np.zeros(T_STEPS, dtype=np.int32)
    history_e = np.zeros(T_STEPS)

    for t in range(T_STEPS):
        e_t = OMEGA_E * t  # phase of E at time t (cos(e_t) is the signal)

        # Mean field of B1 (Kuramoto order parameter direction).
        b1_mean = np.angle(np.exp(1j * phi_1).mean())
        # Discretize into a codeword
        msg = int((b1_mean % (2 * np.pi)) // (2 * np.pi / N_CODEWORDS))

        # B1 dynamics: intrinsic + intra-coupling + weak E forcing + noise
        coupling_1 = (K_INTRA_1 / N1) * np.sum(np.sin(phi_1[None, :] - phi_1[:, None]), axis=1)
        forcing_e = K_E * np.sin(e_t - phi_1)
        noise_1 = rng.normal(0, sigma, size=N1)
        phi_1 = (phi_1 + DT * (omega_1 + coupling_1 + forcing_e) + np.sqrt(DT) * noise_1) % (2 * np.pi)

        # B2 dynamics: intrinsic + intra-coupling + code forcing + noise
        coupling_2 = (K_INTRA_2 / N2) * np.sum(np.sin(phi_2[None, :] - phi_2[:, None]), axis=1)
        if condition == "no_code":
            code_force = np.zeros(N2)
        else:
            tp = target_phases[msg]
            code_force = K_M * np.sin(tp - phi_2)
        noise_2 = rng.normal(0, sigma, size=N2)
        phi_2 = (phi_2 + DT * (omega_2 + coupling_2 + code_force) + np.sqrt(DT) * noise_2) % (2 * np.pi)

        history_b1_mean[t] = b1_mean
        history_b2_mean[t] = np.angle(np.exp(1j * phi_2).mean())
        history_msg[t] = msg
        history_e[t] = e_t

    # ------------------------------------------------------------------
    # Post-process. Drop burn-in transient.
    b1 = history_b1_mean[T_BURN:]
    b2 = history_b2_mean[T_BURN:]
    msg = history_msg[T_BURN:]
    e = history_e[T_BURN:]

    # SHUFFLED_EVAL: shuffle B2's mean-phase trajectory in time before
    # scoring. B2 is the receiver oscillator network whose target phases
    # are keyed by the codeword; shuffling its trajectory at evaluation
    # breaks the (B1, B2, E) synchronization while preserving the codeword
    # sequence and B1's dynamics. This is a synchronization-only ablation,
    # not a causal shuffle of the codeword input itself.
    if condition == "shuffled_eval":
        perm = rng.permutation(b2.shape[0])
        b2_for_eval = b2[perm]
    else:
        b2_for_eval = b2

    # Phase locking metrics (Kuramoto-style).
    # R_b1_e = magnitude of <exp(i (b1 - e))>
    R_b1_e = np.abs(np.mean(np.exp(1j * (b1 - e))))
    R_b2_e = np.abs(np.mean(np.exp(1j * (b2_for_eval - e))))
    # Future locking: B2 phase to E shifted by Delta
    e_future = e + DELTA
    R_b2_efut = np.abs(np.mean(np.exp(1j * (b2_for_eval - e_future))))

    # Viability score: how well B2 phase predicts cos(E_{t+Delta}).
    # Action = sign of cos(B2 phase). Target = sign of cos(E + Delta).
    action = np.sign(np.cos(b2_for_eval))
    target = np.sign(np.cos(e_future))
    accuracy = float((action == target).mean())

    return {
        "sigma": sigma,
        "condition": condition,
        "seed": seed,
        "R_b1_e": float(R_b1_e),
        "R_b2_e": float(R_b2_e),
        "R_b2_efut": float(R_b2_efut),
        "accuracy": accuracy,
    }


def t95(n: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
    return table.get(n - 1, 1.96)


def aggregate(results: list[dict]) -> dict:
    by_cell = {}
    for r in results:
        key = (r["sigma"], r["condition"])
        by_cell.setdefault(key, []).append(r)
    agg = {}
    for (sig, cond), runs in by_cell.items():
        n = len(runs)
        for metric in ("R_b1_e", "R_b2_e", "R_b2_efut", "accuracy"):
            arr = np.array([r[metric] for r in runs])
            agg[(sig, cond, metric, "mean")] = float(arr.mean())
            agg[(sig, cond, metric, "ci")] = (
                float(t95(n) * arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            )
    return agg


def plot_results(agg: dict, sigmas: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    palette = {
        "selected": "#1f7770",
        "random": "#9b3d2f",
        "shuffled_eval": "#b65a34",
        "no_code": "#52616b",
    }
    labels = {
        "selected": "selected code",
        "random": "random code",
        "shuffled_eval": "selected code, bulk shuffled at eval",
        "no_code": "no code",
    }
    markers = {"selected": "o", "random": "s", "shuffled_eval": "^", "no_code": "x"}

    panels = [
        ("R_b1_e", axes[0], r"$R(B_1, E)$",
         "(a) B$_1$ to E phase-locking"),
        ("R_b2_efut", axes[1], r"$R(B_2, E_{t+\Delta})$",
         r"(b) B$_2$ to $E_{t+\Delta}$ phase-locking"),
        ("accuracy", axes[2], "future-class accuracy",
         "(c) Future viability (B$_2$ predicts $E_{t+\\Delta}$)"),
    ]
    for metric, ax, ylabel, title in panels:
        for cond in CONDITIONS:
            means = np.array([agg[(s, cond, metric, "mean")] for s in sigmas])
            cis = np.array([agg[(s, cond, metric, "ci")] for s in sigmas])
            ax.fill_between(sigmas, means - cis, means + cis, color=palette[cond], alpha=0.12)
            ax.plot(
                sigmas, means,
                color=palette[cond], lw=2, marker=markers[cond], ms=6,
                label=labels[cond],
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"oscillator noise std $\sigma$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, ls=":", alpha=0.3, which="both")
        ax.legend(loc="best", frameon=False, fontsize=7)

    axes[2].axhline(0.5, color="#27313a", lw=1, ls=":", alpha=0.7)
    axes[2].text(sigmas[-1], 0.51, "chance", ha="right", va="bottom", fontsize=8, color="#27313a")
    axes[2].set_ylim(0.4, 1.0)

    fig.suptitle(
        f"Stochastic resonance in coupled-bulk synchronization "
        f"({len(SEEDS)} seeds, K={N_CODEWORDS} codewords, $\\Delta={DELTA:.2f}$)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout(pad=1.0)
    out_pdf = FIG_DIR / "stochastic_resonance.pdf"
    out_png = FIG_DIR / "stochastic_resonance.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_png


def write_summary(results: list[dict], agg: dict, sigmas: np.ndarray, runtime: float) -> Path:
    lines = [
        "# Stochastic resonance sweep for bulk-to-bulk synchronization",
        f"# runtime_seconds={runtime:.1f}",
        f"# N1={N1} N2={N2} K_codewords={N_CODEWORDS} omega_E={OMEGA_E} Delta={DELTA:.4f}",
        f"# K_intra={K_INTRA_1} K_E={K_E} K_M={K_M} t_steps={T_STEPS} t_burn={T_BURN}",
        f"# sigmas={sigmas.tolist()}  seeds={SEEDS}  conditions={CONDITIONS}",
        "",
        "## Aggregate (mean +/- 95% CI)",
        f"{'sigma':>8} {'condition':<16} "
        f"{'R(B1,E)':>10} {'R(B2,Ef)':>10} {'accuracy':>10}",
    ]
    for sig in sigmas:
        for cond in CONDITIONS:
            r1 = agg[(sig, cond, 'R_b1_e', 'mean')]
            r1c = agg[(sig, cond, 'R_b1_e', 'ci')]
            rf = agg[(sig, cond, 'R_b2_efut', 'mean')]
            rfc = agg[(sig, cond, 'R_b2_efut', 'ci')]
            ac = agg[(sig, cond, 'accuracy', 'mean')]
            acc = agg[(sig, cond, 'accuracy', 'ci')]
            lines.append(
                f"{sig:>8.4f} {cond:<16} "
                f"{r1:>5.3f}+/-{r1c:>3.3f} {rf:>5.3f}+/-{rfc:>3.3f} {ac:>5.3f}+/-{acc:>3.3f}"
            )
    out = FIG_DIR / "stochastic_resonance.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    t0 = time.time()
    results = []
    total = len(SIGMAS) * len(CONDITIONS) * len(SEEDS)
    cell_index = 0
    for sigma in SIGMAS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                cell_index += 1
                t_cell = time.time()
                r = simulate(sigma, condition, seed)
                cell_secs = time.time() - t_cell
                results.append(r)
                elapsed = time.time() - t0
                if cell_index % 5 == 0 or cell_index == total:
                    print(
                        f"[{cell_index:3d}/{total}] sigma={sigma:.4f} {condition:<16s} "
                        f"R_B1E={r['R_b1_e']:.3f} R_B2Ef={r['R_b2_efut']:.3f} "
                        f"acc={r['accuracy']:.3f}  ({cell_secs:.1f}s, total {elapsed:.0f}s)",
                        flush=True,
                    )
    runtime = time.time() - t0
    agg = aggregate(results)
    summary_path = write_summary(results, agg, SIGMAS, runtime)
    fig_path = plot_results(agg, SIGMAS)
    print(f"\nwrote {summary_path}")
    print(f"wrote {fig_path}")
    print(f"runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
