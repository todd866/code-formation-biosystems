#!/usr/bin/env python3
"""Code-vs-bulk ablation for the boundary-code framework.

This sim tests the synchronization-collapse signature of the latent-context
proposition (Latent anticipation through context):

  A system trained to use a code + bulk-context channel loses viability
  when bulk-environment synchronization is destroyed at evaluation, while
  a code-only system trained without that channel is unaffected. The
  ablation does not require code+bulk to dominate code-only in absolute
  viability; in observed runs the two are statistically indistinguishable
  because, given enough channel capacity, the encoder can extract slow-mode
  information directly from the substrate.

Four architectures are evolved under the same compute budget against the
same hidden nonlinear world used in first_code_complex_sim.py:

  1. CODE_ONLY      A_t = delta(M_t)              # boundary message only
  2. BULK_ONLY      A_t = delta(C_t)              # entrained context only
  3. CODE_PLUS_BULK A_t = delta(M_t, C_t)         # the operational unit
  4. CODE_PLUS_SHUFFLED  A_t = delta(M_t, C_shuf) # codeword intact, sync broken

C_t is a discretization of the leading slow-mode coordinate (quartile of
the dominant oscillator phase). Shuffling permutes C_t across samples
within each batch, preserving the marginal distribution of C and the
distribution of (M, A) but destroying the (X_t, C_t, F_{t+Δ}) synchrony.

Predictions (operative claim, narrower than originally drafted)
---------------------------------------------------------------
- CODE_PLUS_BULK and CODE_ONLY may be statistically indistinguishable in this
  regime: when channel capacity is large enough for the encoder to extract
  slow-mode information directly from the substrate, an explicit bulk channel
  is one route among others (latent-context route of the Latent
  Anticipation proposition). In the observed runs CODE_ONLY is in fact
  slightly higher than CODE_PLUS_BULK.
- CODE_PLUS_BULK > BULK_ONLY: a coarse four-class context is too coarse to
  address eight response programs.
- CODE_PLUS_SHUFFLED < CODE_PLUS_BULK at evaluation: shuffling the bulk
  destroys the synchronization the trained system has learned to use, while
  CODE_ONLY (which never relied on bulk) is unaffected. This is the supported
  diagnostic claim; it does not require bulk to dominate code-only.

Outputs
-------
- figures/ablation_code_vs_bulk.{pdf,png}
- figures/ablation_code_vs_bulk.txt
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

from first_code_complex_sim import Config, World


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


# Use 4 quartile classes for C_t (leading oscillator phase).
N_CONTEXT_CLASSES = 4


# Base config closely matches the medium preset of first_code_complex_sim.py.
BASE_CONFIG = dict(
    latent_dim=10,
    n_actions=8,
    n_codewords=8,
    pop_size=160,
    generations=400,
    samples_per_gen=768,
    eval_samples=12000,
    mutation_std=0.045,
    bias_mutation_std=0.018,
    decoder_flip_prob=0.08,
    elite_frac=0.45,
    slow_noise=0.14,
    substrate_noise=0.28,
    future_delta=1.1,
    run_drift=False,
    run_random_search=False,
)


def build_config(tag: str, seed: int, n_dim: int) -> Config:
    cfg_kwargs = dict(BASE_CONFIG)
    cfg_kwargs["n_dim"] = n_dim
    cfg_kwargs["tag"] = tag
    cfg_kwargs["seed"] = seed
    return Config(**cfg_kwargs)


def context_from_z(z_now: np.ndarray) -> np.ndarray:
    """Discretize the leading oscillator phase into N_CONTEXT_CLASSES classes."""
    phi1 = np.arctan2(z_now[:, 1], z_now[:, 0]) % (2 * np.pi)
    return (phi1 // (2 * np.pi / N_CONTEXT_CLASSES)).astype(np.int64)


def encode(theta: np.ndarray, bias: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-individual argmax codeword. theta: (P, K, N), bias: (P, K)."""
    logits = np.einsum("pkn,sn->psk", theta, x, optimize=True)
    logits += bias[:, None, :]
    return np.argmax(logits, axis=2).astype(np.int16)


# ---------------------------------------------------------------------------
# Architecture-specific evaluation. Each returns per-individual normalized
# fitness of shape (P,).


def normalize(scores: np.ndarray, chosen_scores: np.ndarray) -> np.ndarray:
    oracle = np.max(scores, axis=1)
    mean = scores.mean(axis=1)
    denom = np.maximum(oracle - mean, 1e-6)
    return ((chosen_scores - mean[None, :]) / denom[None, :]).mean(axis=1)


def eval_code_only(theta, bias, decoder, x, c_t, scores) -> np.ndarray:
    """decoder shape: (P, K)."""
    msgs = encode(theta, bias, x)  # (P, S)
    actions = np.take_along_axis(decoder, msgs, axis=1)  # (P, S)
    chosen = np.take_along_axis(scores.T[None, :, :], actions[:, None, :], axis=1)[:, 0, :]
    return normalize(scores, chosen)


def eval_bulk_only(decoder, c_t, scores) -> np.ndarray:
    """decoder shape: (P, L)."""
    actions = decoder[:, c_t]  # (P, S)
    chosen = np.take_along_axis(scores.T[None, :, :], actions[:, None, :], axis=1)[:, 0, :]
    return normalize(scores, chosen)


def eval_code_plus_bulk(theta, bias, decoder, x, c_t, scores) -> np.ndarray:
    """decoder shape: (P, K, L)."""
    msgs = encode(theta, bias, x)  # (P, S)
    actions = decoder[
        np.arange(decoder.shape[0])[:, None],
        msgs,
        c_t[None, :],
    ]
    chosen = np.take_along_axis(scores.T[None, :, :], actions[:, None, :], axis=1)[:, 0, :]
    return normalize(scores, chosen)


def eval_code_plus_shuffled(theta, bias, decoder, x, c_t, scores, rng) -> np.ndarray:
    """Same decoder shape as code+bulk, but evaluate with shuffled c."""
    perm = rng.permutation(c_t.shape[0])
    c_shuf = c_t[perm]
    return eval_code_plus_bulk(theta, bias, decoder, x, c_shuf, scores)


# ---------------------------------------------------------------------------
# Population init / mutation per architecture.


def init_population(rng: np.random.Generator, cfg: Config, arch: str):
    p, k, n = cfg.pop_size, cfg.n_codewords, cfg.n_dim
    L = N_CONTEXT_CLASSES
    if arch == "code_only":
        theta = rng.normal(0, 0.25 / np.sqrt(n), size=(p, k, n)).astype(np.float32)
        bias = rng.normal(0, 0.02, size=(p, k)).astype(np.float32)
        decoder = rng.integers(0, cfg.n_actions, size=(p, k), dtype=np.int16)
        return {"theta": theta, "bias": bias, "decoder": decoder}
    if arch == "bulk_only":
        decoder = rng.integers(0, cfg.n_actions, size=(p, L), dtype=np.int16)
        return {"decoder": decoder}
    if arch in ("code_plus_bulk", "code_plus_shuffled"):
        theta = rng.normal(0, 0.25 / np.sqrt(n), size=(p, k, n)).astype(np.float32)
        bias = rng.normal(0, 0.02, size=(p, k)).astype(np.float32)
        decoder = rng.integers(0, cfg.n_actions, size=(p, k, L), dtype=np.int16)
        return {"theta": theta, "bias": bias, "decoder": decoder}
    raise ValueError(arch)


def reproduce_pop(
    rng: np.random.Generator, cfg: Config, arch: str,
    pop: dict, fitness: np.ndarray,
) -> dict:
    p = cfg.pop_size
    n_survive = max(2, int(round(cfg.elite_frac * p)))
    order = np.argsort(fitness)[::-1][:n_survive]

    new = {}
    has_theta = "theta" in pop
    if has_theta:
        new["theta"] = pop["theta"].copy()
        new["bias"] = pop["bias"].copy()
    new["decoder"] = pop["decoder"].copy()

    for i in range(p):
        if i < n_survive:
            src = order[i]
            if has_theta:
                new["theta"][i] = pop["theta"][src]
                new["bias"][i] = pop["bias"][src]
            new["decoder"][i] = pop["decoder"][src]
        else:
            parent_pos = rng.integers(n_survive)
            src = order[parent_pos]
            if has_theta:
                new["theta"][i] = pop["theta"][src] + rng.normal(0, cfg.mutation_std, size=pop["theta"][src].shape)
                new["bias"][i] = pop["bias"][src] + rng.normal(0, cfg.bias_mutation_std, size=pop["bias"][src].shape)
            new["decoder"][i] = pop["decoder"][src]
            if rng.uniform() < cfg.decoder_flip_prob:
                idx = tuple(rng.integers(s) for s in new["decoder"][i].shape)
                new["decoder"][i][idx] = rng.integers(cfg.n_actions)
    return new


# ---------------------------------------------------------------------------


def evolve(world: World, cfg: Config, arch: str, seed_offset: int) -> dict:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    pop = init_population(rng, cfg, arch)
    history = []

    for gen in range(cfg.generations):
        x, scores, z_now = world.sample_batch(rng, cfg.samples_per_gen)
        c_t = context_from_z(z_now)
        if arch == "code_only":
            fit = eval_code_only(pop["theta"], pop["bias"], pop["decoder"], x, c_t, scores)
        elif arch == "bulk_only":
            fit = eval_bulk_only(pop["decoder"], c_t, scores)
        elif arch == "code_plus_bulk":
            fit = eval_code_plus_bulk(pop["theta"], pop["bias"], pop["decoder"], x, c_t, scores)
        elif arch == "code_plus_shuffled":
            fit = eval_code_plus_shuffled(pop["theta"], pop["bias"], pop["decoder"], x, c_t, scores, rng)
        else:
            raise ValueError(arch)
        history.append((float(fit.max()), float(fit.mean())))
        pop = reproduce_pop(rng, cfg, arch, pop, fit)

    # Held-out evaluation. For shuffled architecture we additionally evaluate
    # both with and without shuffling to compare meaning preservation vs
    # synchronization.
    x_eval, scores_eval, z_eval = world.sample_batch(rng, cfg.eval_samples)
    c_eval = context_from_z(z_eval)
    if arch == "code_only":
        fit_eval = eval_code_only(pop["theta"], pop["bias"], pop["decoder"], x_eval, c_eval, scores_eval)
        sync_intact = float(np.max(fit_eval))
        sync_broken = sync_intact  # no bulk to break
    elif arch == "bulk_only":
        fit_eval = eval_bulk_only(pop["decoder"], c_eval, scores_eval)
        sync_intact = float(np.max(fit_eval))
        # shuffle bulk-only too (this just becomes random)
        c_shuf = c_eval[rng.permutation(c_eval.shape[0])]
        sync_broken = float(np.max(eval_bulk_only(pop["decoder"], c_shuf, scores_eval)))
    elif arch == "code_plus_bulk":
        fit_eval = eval_code_plus_bulk(pop["theta"], pop["bias"], pop["decoder"], x_eval, c_eval, scores_eval)
        sync_intact = float(np.max(fit_eval))
        c_shuf = c_eval[rng.permutation(c_eval.shape[0])]
        fit_shuf = eval_code_plus_bulk(pop["theta"], pop["bias"], pop["decoder"], x_eval, c_shuf, scores_eval)
        sync_broken = float(np.max(fit_shuf))
    elif arch == "code_plus_shuffled":
        # Trained with shuffle; evaluate both ways
        fit_eval = eval_code_plus_bulk(pop["theta"], pop["bias"], pop["decoder"], x_eval, c_eval, scores_eval)
        sync_intact = float(np.max(fit_eval))
        c_shuf = c_eval[rng.permutation(c_eval.shape[0])]
        fit_shuf = eval_code_plus_bulk(pop["theta"], pop["bias"], pop["decoder"], x_eval, c_shuf, scores_eval)
        sync_broken = float(np.max(fit_shuf))

    return {
        "arch": arch,
        "history": np.array(history),
        "final_intact": sync_intact,
        "final_shuffled": sync_broken,
    }


def t95(n: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(n - 1, 1.96)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-dim", type=int, default=128)
    parser.add_argument("--seeds", default="321,322,323,324,325", type=str)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    architectures = ["code_only", "bulk_only", "code_plus_bulk", "code_plus_shuffled"]

    t0 = time.time()
    results = {arch: [] for arch in architectures}
    histories = {arch: [] for arch in architectures}
    final_shuf = {arch: [] for arch in architectures}

    for seed in seeds:
        for ai, arch in enumerate(architectures):
            cfg = build_config(tag=f"abl_{arch}_seed{seed}", seed=seed, n_dim=args.n_dim)
            if args.quick:
                cfg = Config(**{**cfg.__dict__, "generations": 100, "samples_per_gen": 256, "eval_samples": 2048})
            world = World(cfg)
            t_run = time.time()
            r = evolve(world, cfg, arch, seed_offset=1000 * (ai + 1))
            results[arch].append(r["final_intact"])
            final_shuf[arch].append(r["final_shuffled"])
            histories[arch].append(r["history"])
            elapsed = time.time() - t0
            print(
                f"seed={seed} arch={arch:<22s} intact={r['final_intact']:+.3f}  "
                f"shuffled={r['final_shuffled']:+.3f}  "
                f"({time.time() - t_run:.1f}s, total {elapsed:.0f}s)",
                flush=True,
            )

    runtime = time.time() - t0

    # Aggregate
    agg = {}
    for arch in architectures:
        intact = np.array(results[arch])
        shuf = np.array(final_shuf[arch])
        n = len(intact)
        agg[arch] = {
            "intact_mean": float(intact.mean()),
            "intact_ci": float(t95(n) * intact.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "shuf_mean": float(shuf.mean()),
            "shuf_ci": float(t95(n) * shuf.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        }

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    palette = {
        "code_only": "#52616b",
        "bulk_only": "#9b3d2f",
        "code_plus_bulk": "#1f7770",
        "code_plus_shuffled": "#b65a34",
    }
    labels = {
        "code_only": "code only",
        "bulk_only": "bulk only",
        "code_plus_bulk": "code + bulk",
        "code_plus_shuffled": "code + bulk (trained shuffled)",
    }

    # Panel (a): bar chart of final intact viability
    ax = axes[0]
    x = np.arange(len(architectures))
    means = np.array([agg[a]["intact_mean"] for a in architectures])
    cis = np.array([agg[a]["intact_ci"] for a in architectures])
    bars = ax.bar(x, means, yerr=cis, capsize=4, color=[palette[a] for a in architectures], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[a] for a in architectures], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("normalized viability\n(0 = mean action, 1 = oracle)")
    ax.set_title(f"(a) Architecture comparison (mean +/- 95% CI, {len(seeds)} seeds)", fontsize=10)
    ax.axhline(0, color="#27313a", lw=1, ls=":", alpha=0.6)
    ax.grid(True, axis="y", ls=":", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    # Panel (b): paired intact-vs-shuffled-eval bar for code+bulk and code-only baselines
    ax = axes[1]
    paired_archs = ["code_only", "code_plus_bulk"]
    width = 0.32
    xs = np.arange(len(paired_archs))
    intact_means = np.array([agg[a]["intact_mean"] for a in paired_archs])
    intact_cis = np.array([agg[a]["intact_ci"] for a in paired_archs])
    shuf_means = np.array([agg[a]["shuf_mean"] for a in paired_archs])
    shuf_cis = np.array([agg[a]["shuf_ci"] for a in paired_archs])
    ax.bar(xs - width/2, intact_means, width=width, yerr=intact_cis, capsize=3,
           color=[palette[a] for a in paired_archs], label="evaluated synchronized")
    ax.bar(xs + width/2, shuf_means, width=width, yerr=shuf_cis, capsize=3,
           color=[palette[a] for a in paired_archs], alpha=0.4, hatch="//", label="evaluated with bulk shuffled")
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[a] for a in paired_archs], fontsize=10)
    ax.set_ylabel("normalized viability")
    ax.set_title("(b) Synchronization vs codeword identity", fontsize=10)
    ax.axhline(0, color="#27313a", lw=1, ls=":", alpha=0.6)
    ax.grid(True, axis="y", ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for i, arch in enumerate(paired_archs):
        ax.text(i - width/2, intact_means[i] + 0.02, f"{intact_means[i]:.2f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, shuf_means[i] + 0.02, f"{shuf_means[i]:.2f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout(pad=1.0)
    fig.savefig(FIG_DIR / "ablation_code_vs_bulk.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ablation_code_vs_bulk.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # ---- Summary file ----
    lines = [
        "# Code-vs-bulk ablation",
        f"# n_dim={args.n_dim}  seeds={seeds}  quick={args.quick}",
        f"# runtime_seconds={runtime:.1f}",
        f"# base_config={json.dumps(BASE_CONFIG)}",
        "",
        f"{'arch':<24} {'intact_mean':>12} {'intact_ci':>12} {'shuf_mean':>12} {'shuf_ci':>12}",
    ]
    for arch in architectures:
        a = agg[arch]
        lines.append(
            f"{arch:<24} {a['intact_mean']:>12.4f} {a['intact_ci']:>12.4f} "
            f"{a['shuf_mean']:>12.4f} {a['shuf_ci']:>12.4f}"
        )
    lines.append("")
    lines.append("# Per-seed intact viability")
    for arch in architectures:
        lines.append(f"{arch}: {[f'{v:.3f}' for v in results[arch]]}")
    (FIG_DIR / "ablation_code_vs_bulk.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
