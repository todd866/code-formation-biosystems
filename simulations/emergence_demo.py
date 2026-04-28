#!/usr/bin/env python3
"""Selection-driven partition emergence.

Demonstrates that low-loss boundary protocols (encoder/decoder pairs) emerge
from random initial conditions under mutation and selection on coordination
loss. Complements the one-shot demo in generate_figures.py: that figure shows
the optimal one-bit partition exists; this one shows the optimal multi-bit
partition is reachable from random initial conditions without any pre-specified
alphabet.

Setup
-----
- Substrate X in R^2: a slow phase phi drives X = (cos phi, sin phi) + noise.
- Future task F is a 4-class label determined by phi shifted by Delta.
- Boundary channel carries log2(K) bits (K = 4 codewords).
- Encoder pi_theta: x -> argmax_m (theta_m . x), parameters theta in R^{K x N}.
- Decoder delta: m -> predicted class, a lookup table in {0, ..., K-1}^K.
- Loss: 0/1 misclassification of F.

Selection
---------
- Population of 64 (theta, delta) pairs.
- Each generation: evaluate on 800 fresh substrate samples; top 32 survive;
  refill by Gaussian mutation on theta plus occasional decoder-table swap.
- Run 200 generations.

Controls
--------
- High-bandwidth: oracle classifier with full access to phi. Upper bound.
- No-selection: same mutation operator, but parents picked uniformly at
  random instead of by fitness. Verifies the partition is selected, not
  drift-emergent.

Output
------
- figures/fig3_emergence_demo.{pdf,png} with four panels.
- figures/emergence_results.txt with summary numbers.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


# Hyperparameters --- fixed by RNG seed for reproducibility.
N = 6                    # substrate dimension; only first two carry phi
K = 4                    # number of codewords (2 bits)
DELTA = np.pi / 2        # future horizon: a quarter cycle ahead
NOISE = 0.45             # gaussian noise on the slow-mode coordinates
DISTRACTOR_NOISE = 0.7   # gaussian noise on the N-2 distractor dimensions
N_SAMPLES_PER_EVAL = 1200
N_GENERATIONS = 250
POP_SIZE = 64
MUTATION_STD = 0.16      # std for theta mutation
DECODER_FLIP_PROB = 0.12 # per-child prob of one decoder-entry mutation
RNG_SEED = 7


def sample_substrate(rng, n):
    """Sample n substrate states from a slow-phase + noise distribution.

    Returns phi (uniform on [0, 2pi)) and x in R^N. The first two coordinates
    are (cos phi, sin phi) plus noise; the remaining N-2 are pure Gaussian
    noise (distractors). The encoder must learn to ignore the distractors
    and project onto the slow-mode-relevant subspace.
    """
    phi = rng.uniform(0, 2 * np.pi, n)
    slow = np.column_stack((np.cos(phi), np.sin(phi))) + rng.normal(0, NOISE, size=(n, 2))
    distractors = rng.normal(0, DISTRACTOR_NOISE, size=(n, N - 2))
    x = np.concatenate([slow, distractors], axis=1)
    return phi, x


def future_class(phi):
    """K-class future label: which quadrant phi + Delta lies in."""
    fphi = (phi + DELTA) % (2 * np.pi)
    return (fphi // (2 * np.pi / K)).astype(int)


def encode(theta, x):
    """theta is (K, N); returns argmax codeword for each x."""
    return np.argmax(x @ theta.T, axis=1)


def fitness(theta, decoder, x, future):
    m = encode(theta, x)
    pred = decoder[m]
    return float(np.mean(pred == future))


def mutate(parent, rng):
    theta, decoder = parent
    new_theta = theta + rng.normal(0, MUTATION_STD, size=theta.shape)
    new_decoder = decoder.copy()
    if rng.uniform() < DECODER_FLIP_PROB:
        i = rng.integers(K)
        new_decoder[i] = rng.integers(K)
    return (new_theta, new_decoder)


def random_individual(rng):
    return (rng.normal(0, 0.5, size=(K, N)), rng.integers(0, K, size=K))


def fiber_coherence(theta, decoder, x, future):
    """Fraction of fiber mass whose assigned response matches the true class.

    1.0 = every fiber is perfectly coherent under its assigned response.
    Random partitions give ~ 1/K.
    """
    m = encode(theta, x)
    correct = (decoder[m] == future)
    return float(np.mean(correct))


def run_evolution(rng, with_selection=True):
    """Returns (history, final_population). history is a list of dicts.

    with_selection=True: top half by fitness survives.
    with_selection=False: parents drawn uniformly at random (drift control).
    """
    population = [random_individual(rng) for _ in range(POP_SIZE)]
    history = []

    for gen in range(N_GENERATIONS):
        phi, x = sample_substrate(rng, N_SAMPLES_PER_EVAL)
        future = future_class(phi)

        fits = np.array([fitness(t, d, x, future) for (t, d) in population])
        best_idx = int(np.argmax(fits))
        coh_best = fiber_coherence(*population[best_idx], x, future)

        history.append({
            "gen": gen,
            "best_fit": float(np.max(fits)),
            "mean_fit": float(np.mean(fits)),
            "fiber_coh": coh_best,
        })

        order = np.argsort(fits)[::-1]
        if with_selection:
            survivors = [population[i] for i in order[: POP_SIZE // 2]]
        else:
            # drift control: parents from anywhere in the population
            survivors = [population[i] for i in rng.choice(POP_SIZE, POP_SIZE // 2, replace=False)]

        new_pop = list(survivors)
        while len(new_pop) < POP_SIZE:
            parent = survivors[rng.integers(len(survivors))]
            new_pop.append(mutate(parent, rng))
        population = new_pop

    return history, population


def best_individual(population, x, future):
    fits = [fitness(t, d, x, future) for (t, d) in population]
    return population[int(np.argmax(fits))], max(fits)


def plot_partition(ax, theta, decoder, x, future, title):
    """Scatter substrate samples colored by the decoder's predicted future class.

    Plots the first two substrate coordinates (the slow-mode-relevant ones).
    The remaining N-2 coordinates are distractor noise the encoder may or
    may not have learned to ignore. Color = decoded future class (one of K).
    """
    m = encode(theta, x)
    palette = ["#1f7770", "#b65a34", "#5b8db8", "#a55caf"]
    for k in range(K):
        mask = m == k
        ax.scatter(
            x[mask, 0], x[mask, 1],
            c=palette[decoder[k]], s=4, alpha=0.45, linewidths=0,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel(r"$X_1$ (slow-mode)")
    ax.set_ylabel(r"$X_2$ (slow-mode)")
    ax.set_title(title, fontsize=10)


def main():
    rng = np.random.default_rng(RNG_SEED)
    rng_drift = np.random.default_rng(RNG_SEED + 1)

    # Capture initial population for "gen 0" plot, then run evolution.
    init_pop = [random_individual(rng) for _ in range(POP_SIZE)]

    # Re-seed and reuse same initial pop for fair comparison
    rng_sel = np.random.default_rng(RNG_SEED + 100)
    population_sel = [(t.copy(), d.copy()) for (t, d) in init_pop]
    history_sel = []
    for gen in range(N_GENERATIONS):
        phi, x = sample_substrate(rng_sel, N_SAMPLES_PER_EVAL)
        future = future_class(phi)
        fits = np.array([fitness(t, d, x, future) for (t, d) in population_sel])
        best_idx = int(np.argmax(fits))
        history_sel.append({
            "gen": gen,
            "best_fit": float(np.max(fits)),
            "mean_fit": float(np.mean(fits)),
        })
        order = np.argsort(fits)[::-1]
        survivors = [population_sel[i] for i in order[: POP_SIZE // 2]]
        new_pop = list(survivors)
        while len(new_pop) < POP_SIZE:
            parent = survivors[rng_sel.integers(len(survivors))]
            new_pop.append(mutate(parent, rng_sel))
        population_sel = new_pop

    # Drift control: same init pop, no selection
    rng_drift = np.random.default_rng(RNG_SEED + 200)
    population_dr = [(t.copy(), d.copy()) for (t, d) in init_pop]
    history_dr = []
    for gen in range(N_GENERATIONS):
        phi, x = sample_substrate(rng_drift, N_SAMPLES_PER_EVAL)
        future = future_class(phi)
        fits = np.array([fitness(t, d, x, future) for (t, d) in population_dr])
        history_dr.append({
            "gen": gen,
            "best_fit": float(np.max(fits)),
            "mean_fit": float(np.mean(fits)),
        })
        # No selection: parents from anywhere
        idx = rng_drift.choice(POP_SIZE, POP_SIZE // 2, replace=False)
        survivors = [population_dr[i] for i in idx]
        new_pop = list(survivors)
        while len(new_pop) < POP_SIZE:
            parent = survivors[rng_drift.integers(len(survivors))]
            new_pop.append(mutate(parent, rng_drift))
        population_dr = new_pop

    # Final-evaluation samples for the partition panels
    rng_eval = np.random.default_rng(RNG_SEED + 999)
    phi_eval, x_eval = sample_substrate(rng_eval, 4000)
    future_eval = future_class(phi_eval)

    # Panel (a): a single random individual (not best-of-pop), to show
    # what an unselected encoder/decoder looks like at start.
    init_single = init_pop[0]
    fit_init = fitness(init_single[0], init_single[1], x_eval, future_eval)
    best_sel, fit_sel = best_individual(population_sel, x_eval, future_eval)
    best_dr, fit_dr = best_individual(population_dr, x_eval, future_eval)
    best_init = init_single

    # Oracle: classify by true future class directly
    fit_oracle = 1.0

    # ---- Figure ----
    fig = plt.figure(figsize=(11.0, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    ax_init = fig.add_subplot(gs[0, 0])
    ax_sel = fig.add_subplot(gs[0, 1])
    ax_dr = fig.add_subplot(gs[0, 2])
    ax_curve = fig.add_subplot(gs[1, :])

    plot_partition(ax_init, *best_init, x_eval, future_eval,
                   f"(a) Random individual, generation 0\naccuracy {fit_init:.2f}")
    plot_partition(ax_sel, *best_sel, x_eval, future_eval,
                   f"(b) After {N_GENERATIONS} generations with selection\naccuracy {fit_sel:.2f}")
    plot_partition(ax_dr, *best_dr, x_eval, future_eval,
                   f"(c) After {N_GENERATIONS} generations, no selection\naccuracy {fit_dr:.2f}")

    gens = np.arange(N_GENERATIONS)
    ax_curve.plot(gens, [h["best_fit"] for h in history_sel],
                  color="#1f7770", lw=2.0, label="selection (best of population)")
    ax_curve.plot(gens, [h["best_fit"] for h in history_dr],
                  color="#b65a34", lw=2.0, label="drift (best of population)")
    ax_curve.axhline(1.0 / K, color="#27313a", ls=":", lw=1.0, alpha=0.7,
                     label=f"chance ($1/K = {1.0/K:.2f}$)")
    ax_curve.axhline(fit_oracle, color="#27313a", ls=":", lw=1.0, alpha=0.4)
    ax_curve.text(N_GENERATIONS - 1, fit_oracle - 0.04, "oracle",
                  ha="right", va="top", fontsize=8, color="#27313a", alpha=0.7)
    ax_curve.set_xlim(0, N_GENERATIONS - 1)
    ax_curve.set_ylim(0.15, 1.05)
    ax_curve.set_xlabel("generation")
    ax_curve.set_ylabel("future-class accuracy")
    ax_curve.set_title("(d) Selection drives convergence to the slow-mode-aligned partition", fontsize=10)
    # Legend above the plot to avoid crowding the data
    ax_curve.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                    ncol=3, frameon=False, fontsize=9)

    fig.tight_layout(pad=1.2)
    fig.savefig(FIG_DIR / "fig3_emergence_demo.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_emergence_demo.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # ---- Numerical summary ----
    summary = (
        f"# Boundary-code emergence under selection (deterministic; rng_seed={RNG_SEED})\n"
        f"N_substrate_dim={N}\n"
        f"K_codewords={K}\n"
        f"Delta={DELTA:.6f}\n"
        f"noise={NOISE}\n"
        f"population_size={POP_SIZE}\n"
        f"generations={N_GENERATIONS}\n"
        f"samples_per_eval={N_SAMPLES_PER_EVAL}\n"
        f"\n"
        f"# Best-of-population future-class accuracy on a held-out 4000-sample evaluation set\n"
        f"random_init_accuracy={fit_init:.4f}\n"
        f"with_selection_final_accuracy={fit_sel:.4f}\n"
        f"no_selection_final_accuracy={fit_dr:.4f}\n"
        f"oracle_accuracy={fit_oracle:.4f}\n"
        f"chance_baseline={1.0 / K:.4f}\n"
    )
    (FIG_DIR / "emergence_results.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
