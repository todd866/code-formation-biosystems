#!/usr/bin/env python3
"""Richer first-code emergence simulation.

This is a heavier companion to ``emergence_demo.py``. It avoids the clean
"signal in the first two coordinates" construction by generating a structured
high-dimensional chemical substrate from a hidden low-dimensional slow
trajectory, then selecting finite bottleneck protocols that coordinate a
downstream response with future viability.

The evolved organism/protocol sees only X in R^N. It does not see the hidden
trajectory coordinates or the oracle future action scores. It may transmit only
one of K finite boundary messages, and a decoder maps each message to one of R
downstream response programs. Selection acts on the viability score of the
chosen response.

Outputs:
  figures/first_code_complex_<tag>.png
  figures/first_code_complex_<tag>.txt

Example quick smoke:
  python3 code/first_code_complex_sim.py --preset smoke

Example overnight-ish run on an Apple Silicon laptop:
  python3 code/first_code_complex_sim.py --preset overnight --tag overnight
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Config:
    tag: str
    seed: int
    n_dim: int
    latent_dim: int
    n_actions: int
    n_codewords: int
    pop_size: int
    generations: int
    samples_per_gen: int
    eval_samples: int
    mutation_std: float
    bias_mutation_std: float
    decoder_flip_prob: float
    elite_frac: float
    slow_noise: float
    substrate_noise: float
    future_delta: float
    run_drift: bool
    run_random_search: bool


def config_from_args(args: argparse.Namespace) -> Config:
    presets = {
        "smoke": dict(
            n_dim=48,
            latent_dim=8,
            n_actions=6,
            n_codewords=6,
            pop_size=64,
            generations=80,
            samples_per_gen=512,
            eval_samples=4096,
            mutation_std=0.06,
            bias_mutation_std=0.025,
            decoder_flip_prob=0.10,
            elite_frac=0.5,
            slow_noise=0.16,
            substrate_noise=0.25,
            future_delta=0.9,
            run_drift=True,
            run_random_search=True,
        ),
        "medium": dict(
            n_dim=128,
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
        ),
        "overnight": dict(
            n_dim=256,
            latent_dim=12,
            n_actions=8,
            n_codewords=8,
            pop_size=256,
            generations=1200,
            samples_per_gen=1024,
            eval_samples=40000,
            mutation_std=0.035,
            bias_mutation_std=0.014,
            decoder_flip_prob=0.06,
            elite_frac=0.4,
            slow_noise=0.13,
            substrate_noise=0.30,
            future_delta=1.15,
            run_drift=True,
            run_random_search=True,
        ),
    }
    if args.preset not in presets:
        raise ValueError(f"unknown preset {args.preset!r}")
    cfg = presets[args.preset] | {
        "tag": args.tag or args.preset,
        "seed": args.seed,
    }

    # Allow explicit overrides without requiring every parameter on the CLI.
    for key in [
        "n_dim",
        "latent_dim",
        "n_actions",
        "n_codewords",
        "pop_size",
        "generations",
        "samples_per_gen",
        "eval_samples",
    ]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return Config(**cfg)


class World:
    """Fixed random chemical world used for all conditions in a run."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        n, l, r = cfg.n_dim, cfg.latent_dim, cfg.n_actions

        # Random structured embedding from hidden slow trajectory to nominal
        # chemical coordinates. QR gives a random active subspace; additional
        # nonlinear terms make the raw substrate nonlinearly warped.
        q, _ = np.linalg.qr(rng.normal(size=(n, l)))
        self.embed_linear = q.astype(np.float32)
        self.embed_quad = rng.normal(0, 0.22 / np.sqrt(l), size=(l, n)).astype(np.float32)
        self.embed_inter = rng.normal(0, 0.18 / np.sqrt(l), size=(l, n)).astype(np.float32)

        # Future viability landscape. Each downstream response program has a
        # linear preference, quadratic preference, and pairwise interaction over
        # hidden trajectory coordinates. These random response landscapes are the
        # "chemistry" that selection can exploit but the encoder cannot observe.
        self.action_linear = rng.normal(0, 1.0 / np.sqrt(l), size=(r, l)).astype(np.float32)
        self.action_quad = rng.normal(0, 0.45 / np.sqrt(l), size=(r, l)).astype(np.float32)
        n_pairs = min(12, l * (l - 1) // 2)
        pairs = []
        all_pairs = [(i, j) for i in range(l) for j in range(i + 1, l)]
        rng.shuffle(all_pairs)
        pairs = all_pairs[:n_pairs]
        self.pairs = np.array(pairs, dtype=np.int32)
        self.action_inter = rng.normal(0, 0.50 / np.sqrt(max(1, n_pairs)), size=(r, n_pairs)).astype(np.float32)

        # Per-coordinate scaling makes some nominal dimensions chemically quiet
        # and others loud, so the nominal dimension exceeds effective dimension.
        self.chemical_scales = rng.lognormal(mean=-0.15, sigma=0.55, size=n).astype(np.float32)

    def sample_latent(self, rng: np.random.Generator, n_samples: int, future: bool = False) -> np.ndarray:
        """Sample hidden slow trajectory coordinates.

        The first four coordinates are two oscillator phases. Remaining latent
        coordinates mix slow developmental/environmental variables and noise.
        If future=True, the oscillator and developmental coordinates are
        advanced by future_delta before being embedded or scored.
        """
        cfg = self.cfg
        l = cfg.latent_dim
        phi1 = rng.uniform(0, 2 * np.pi, n_samples)
        phi2 = rng.uniform(0, 2 * np.pi, n_samples)
        stage = rng.uniform(-1.0, 1.0, n_samples)
        if future:
            phi1 = (phi1 + cfg.future_delta) % (2 * np.pi)
            phi2 = (phi2 + 0.43 * cfg.future_delta) % (2 * np.pi)
            stage = np.clip(stage + 0.22 * np.sin(cfg.future_delta), -1.0, 1.0)

        z = np.zeros((n_samples, l), dtype=np.float32)
        z[:, 0] = np.cos(phi1)
        z[:, 1] = np.sin(phi1)
        if l > 2:
            z[:, 2] = np.cos(phi2)
        if l > 3:
            z[:, 3] = np.sin(phi2)
        if l > 4:
            z[:, 4] = stage
        if l > 5:
            z[:, 5] = stage * stage - 0.33
        if l > 6:
            z[:, 6:] = rng.normal(0, cfg.slow_noise, size=(n_samples, l - 6))
        return z

    def sample_batch(self, rng: np.random.Generator, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return substrate X, true future viability scores, and latent Z."""
        cfg = self.cfg
        z_now = self.sample_latent(rng, n_samples, future=False)
        z_future = self.sample_latent_from_now(z_now)

        linear = z_now @ self.embed_linear.T
        quad = (z_now * z_now) @ self.embed_quad
        inter = np.sin(z_now) @ self.embed_inter
        x = np.tanh(linear + quad + inter)
        x += rng.normal(0, cfg.substrate_noise, size=x.shape).astype(np.float32)
        x *= self.chemical_scales
        x = x.astype(np.float32)

        scores = self.future_scores(z_future)
        return x, scores, z_now

    def sample_latent_from_now(self, z_now: np.ndarray) -> np.ndarray:
        """Advance sampled latent coordinates deterministically plus small drift."""
        cfg = self.cfg
        z = np.array(z_now, copy=True)
        # Recover oscillator phase from cos/sin pairs and advance.
        phi1 = np.arctan2(z[:, 1], z[:, 0]) + cfg.future_delta
        z[:, 0] = np.cos(phi1)
        z[:, 1] = np.sin(phi1)
        if cfg.latent_dim > 3:
            phi2 = np.arctan2(z[:, 3], z[:, 2]) + 0.43 * cfg.future_delta
            z[:, 2] = np.cos(phi2)
            z[:, 3] = np.sin(phi2)
        if cfg.latent_dim > 4:
            z[:, 4] = np.clip(z[:, 4] + 0.22 * np.sin(cfg.future_delta), -1.0, 1.0)
        if cfg.latent_dim > 5:
            z[:, 5] = z[:, 4] * z[:, 4] - 0.33
        return z.astype(np.float32)

    def future_scores(self, z_future: np.ndarray) -> np.ndarray:
        scores = z_future @ self.action_linear.T
        scores += (z_future * z_future) @ self.action_quad.T
        if len(self.pairs) > 0:
            inter_features = z_future[:, self.pairs[:, 0]] * z_future[:, self.pairs[:, 1]]
            scores += inter_features @ self.action_inter.T
        # Center per sample; exp-like growth is unnecessary and less stable.
        scores = scores - scores.mean(axis=1, keepdims=True)
        return scores.astype(np.float32)


def random_population(rng: np.random.Generator, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = rng.normal(0, 0.25 / np.sqrt(cfg.n_dim), size=(cfg.pop_size, cfg.n_codewords, cfg.n_dim)).astype(np.float32)
    bias = rng.normal(0, 0.02, size=(cfg.pop_size, cfg.n_codewords)).astype(np.float32)
    decoder = rng.integers(0, cfg.n_actions, size=(cfg.pop_size, cfg.n_codewords), dtype=np.int16)
    return theta, bias, decoder


def evaluate_population(
    theta: np.ndarray,
    bias: np.ndarray,
    decoder: np.ndarray,
    x: np.ndarray,
    scores: np.ndarray,
    chunk: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized fitness, oracle-action accuracy, and message entropy."""
    p = theta.shape[0]
    oracle_action = np.argmax(scores, axis=1).astype(np.int16)
    oracle_score = np.max(scores, axis=1)
    mean_score = scores.mean(axis=1)
    denom = np.maximum(oracle_score - mean_score, 1e-6)

    fitness = np.empty(p, dtype=np.float32)
    accuracy = np.empty(p, dtype=np.float32)
    entropy = np.empty(p, dtype=np.float32)
    k = theta.shape[1]

    for start in range(0, p, chunk):
        end = min(p, start + chunk)
        logits = np.einsum("pkn,sn->psk", theta[start:end], x, optimize=True)
        logits += bias[start:end, None, :]
        messages = np.argmax(logits, axis=2).astype(np.int16)  # (p_chunk, samples)
        actions = np.take_along_axis(decoder[start:end], messages, axis=1)
        chosen_scores = np.take_along_axis(scores.T[None, :, :], actions[:, None, :], axis=1)[:, 0, :]
        normalized = (chosen_scores - mean_score[None, :]) / denom[None, :]
        fitness[start:end] = normalized.mean(axis=1)
        accuracy[start:end] = (actions == oracle_action[None, :]).mean(axis=1)
        for i, msg in enumerate(messages):
            counts = np.bincount(msg, minlength=k).astype(np.float32)
            probs = counts / max(1.0, counts.sum())
            nz = probs > 0
            entropy[start + i] = -np.sum(probs[nz] * np.log2(probs[nz])) / np.log2(k)
    return fitness, accuracy, entropy


def reproduce(
    rng: np.random.Generator,
    cfg: Config,
    theta: np.ndarray,
    bias: np.ndarray,
    decoder: np.ndarray,
    fitness: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = cfg.pop_size
    n_survive = max(2, int(round(cfg.elite_frac * p)))
    if mode == "selection":
        survivor_idx = np.argsort(fitness)[::-1][:n_survive]
    elif mode == "drift":
        survivor_idx = rng.choice(p, size=n_survive, replace=False)
    elif mode == "random_search":
        return random_population(rng, cfg)
    else:
        raise ValueError(mode)

    new_theta = np.empty_like(theta)
    new_bias = np.empty_like(bias)
    new_decoder = np.empty_like(decoder)
    new_theta[:n_survive] = theta[survivor_idx]
    new_bias[:n_survive] = bias[survivor_idx]
    new_decoder[:n_survive] = decoder[survivor_idx]

    for i in range(n_survive, p):
        parent_pos = rng.integers(n_survive)
        new_theta[i] = new_theta[parent_pos] + rng.normal(0, cfg.mutation_std, size=new_theta[parent_pos].shape)
        new_bias[i] = new_bias[parent_pos] + rng.normal(0, cfg.bias_mutation_std, size=new_bias[parent_pos].shape)
        new_decoder[i] = new_decoder[parent_pos]
        if rng.uniform() < cfg.decoder_flip_prob:
            j = rng.integers(cfg.n_codewords)
            new_decoder[i, j] = rng.integers(cfg.n_actions)
        if rng.uniform() < 0.04:
            a, b = rng.choice(cfg.n_codewords, size=2, replace=False)
            new_theta[i, [a, b]] = new_theta[i, [b, a]]
            new_bias[i, [a, b]] = new_bias[i, [b, a]]
            new_decoder[i, [a, b]] = new_decoder[i, [b, a]]
    return new_theta, new_bias, new_decoder


def run_condition(cfg: Config, world: World, mode: str, seed_offset: int) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    theta, bias, decoder = random_population(rng, cfg)
    best_fit = np.zeros(cfg.generations, dtype=np.float32)
    mean_fit = np.zeros(cfg.generations, dtype=np.float32)
    best_acc = np.zeros(cfg.generations, dtype=np.float32)
    mean_entropy = np.zeros(cfg.generations, dtype=np.float32)

    for gen in range(cfg.generations):
        x, scores, _ = world.sample_batch(rng, cfg.samples_per_gen)
        fit, acc, ent = evaluate_population(theta, bias, decoder, x, scores)
        best_fit[gen] = fit.max()
        mean_fit[gen] = fit.mean()
        best_acc[gen] = acc[np.argmax(fit)]
        mean_entropy[gen] = ent.mean()
        theta, bias, decoder = reproduce(rng, cfg, theta, bias, decoder, fit, mode)

    x_eval, scores_eval, z_eval = world.sample_batch(rng, cfg.eval_samples)
    fit_eval, acc_eval, ent_eval = evaluate_population(theta, bias, decoder, x_eval, scores_eval)
    best_idx = int(np.argmax(fit_eval))
    return {
        "best_fit": best_fit,
        "mean_fit": mean_fit,
        "best_acc": best_acc,
        "mean_entropy": mean_entropy,
        "eval_fit": fit_eval,
        "eval_acc": acc_eval,
        "eval_entropy": ent_eval,
        "best_individual": (theta[best_idx].copy(), bias[best_idx].copy(), decoder[best_idx].copy()),
        "eval_batch": (x_eval, scores_eval, z_eval),
    }


def best_messages(theta: np.ndarray, bias: np.ndarray, x: np.ndarray) -> np.ndarray:
    logits = x @ theta.T + bias[None, :]
    return np.argmax(logits, axis=1)


def plot_results(cfg: Config, results: dict[str, dict[str, np.ndarray | tuple[np.ndarray, ...]]]) -> Path:
    fig = plt.figure(figsize=(12.0, 9.0))
    gs = fig.add_gridspec(2, 2)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_latent = fig.add_subplot(gs[1, 0])
    ax_conf = fig.add_subplot(gs[1, 1])

    colors = {
        "selection": "#1f7770",
        "drift": "#b65a34",
        "random_search": "#52616b",
    }
    labels = {
        "selection": "selection",
        "drift": "no selection / drift",
        "random_search": "random-population baseline",
    }
    gens = np.arange(cfg.generations)
    for mode, res in results.items():
        c = colors[mode]
        ax_curve.plot(gens, res["best_fit"], color=c, lw=2, label=f"best ({labels[mode]})")
        ax_curve.plot(gens, res["mean_fit"], color=c, lw=1, ls="--", alpha=0.55, label=f"mean ({labels[mode]})")
        ax_acc.plot(gens, res["best_acc"], color=c, lw=2, label=labels[mode])

    ax_curve.axhline(0, color="#27313a", lw=1, ls=":", alpha=0.8)
    ax_curve.set_title("(a) Viability score under finite bottleneck", fontsize=10)
    ax_curve.set_xlabel("generation")
    ax_curve.set_ylabel("normalized viability\n(0 = mean action, 1 = oracle action)")
    ax_curve.set_xlim(0, cfg.generations - 1)
    ax_curve.legend(frameon=False, fontsize=7, loc="lower right")
    ax_curve.grid(True, ls=":", alpha=0.3)

    ax_acc.axhline(1.0 / cfg.n_actions, color="#27313a", lw=1, ls=":", alpha=0.8)
    ax_acc.text(cfg.generations - 1, 1.0 / cfg.n_actions + 0.015, "chance", ha="right", va="bottom", fontsize=8)
    ax_acc.set_title("(b) Oracle-action agreement", fontsize=10)
    ax_acc.set_xlabel("generation")
    ax_acc.set_ylabel("best-of-population action accuracy")
    ax_acc.set_xlim(0, cfg.generations - 1)
    ax_acc.set_ylim(0, 1.02)
    ax_acc.legend(frameon=False, fontsize=8, loc="lower right")
    ax_acc.grid(True, ls=":", alpha=0.3)

    # Visualize selected codewords over the first two latent principal
    # coordinates. The response landscape depends on multiple slow coordinates,
    # so the raw first oscillator plane can hide the learned partition.
    sel = results["selection"]
    theta, bias, decoder = sel["best_individual"]
    x_eval, scores_eval, z_eval = sel["eval_batch"]
    n_plot = min(7000, x_eval.shape[0])
    msg = best_messages(theta, bias, x_eval[:n_plot])
    palette = plt.cm.tab10(np.linspace(0, 1, max(10, cfg.n_codewords)))
    z_plot = z_eval[:n_plot]
    z_centered = z_plot - z_plot.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
    latent_pc = z_centered @ vt[:2].T
    for k in range(cfg.n_codewords):
        mask = msg == k
        ax_latent.scatter(latent_pc[mask, 0], latent_pc[mask, 1], s=4, alpha=0.45, color=palette[k])
    ax_latent.set_title("(c) Selected finite messages over hidden latent state", fontsize=10)
    ax_latent.set_xlabel("latent PC1")
    ax_latent.set_ylabel("latent PC2")

    # Confusion matrix: message vs oracle action for selected best individual.
    oracle = np.argmax(scores_eval, axis=1)
    msg_all = best_messages(theta, bias, x_eval)
    conf = np.zeros((cfg.n_codewords, cfg.n_actions), dtype=np.float32)
    for m, a in zip(msg_all, oracle):
        conf[m, a] += 1
    conf = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    im = ax_conf.imshow(conf, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_conf.set_title("(d) Selected message fibers vs oracle action", fontsize=10)
    ax_conf.set_xlabel("oracle future response")
    ax_conf.set_ylabel("boundary message")
    ax_conf.set_xticks(range(cfg.n_actions))
    ax_conf.set_yticks(range(cfg.n_codewords))
    fig.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04, label="within-message fraction")

    fig.tight_layout(pad=1.0)
    out = FIG_DIR / f"first_code_complex_{cfg.tag}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary(cfg: Config, results: dict[str, dict[str, np.ndarray | tuple[np.ndarray, ...]]], runtime: float) -> Path:
    lines = [
        "# Complex first-code emergence simulation",
        f"tag={cfg.tag}",
        f"seed={cfg.seed}",
        f"n_dim={cfg.n_dim}",
        f"latent_dim={cfg.latent_dim}",
        f"n_actions={cfg.n_actions}",
        f"n_codewords={cfg.n_codewords}",
        f"pop_size={cfg.pop_size}",
        f"generations={cfg.generations}",
        f"samples_per_gen={cfg.samples_per_gen}",
        f"eval_samples={cfg.eval_samples}",
        f"mutation_std={cfg.mutation_std}",
        f"decoder_flip_prob={cfg.decoder_flip_prob}",
        f"runtime_seconds={runtime:.1f}",
        "",
        "condition final_best_norm_viability final_best_action_accuracy final_best_message_entropy",
    ]
    for mode, res in results.items():
        fit = res["eval_fit"]
        acc = res["eval_acc"]
        ent = res["eval_entropy"]
        best_idx = int(np.argmax(fit))
        lines.append(f"{mode} {fit[best_idx]:.4f} {acc[best_idx]:.4f} {ent[best_idx]:.4f}")
    out = FIG_DIR / f"first_code_complex_{cfg.tag}.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "medium", "overnight"], default="smoke")
    parser.add_argument("--tag", default=None, help="output tag; defaults to preset name")
    parser.add_argument("--seed", type=int, default=321)
    for name in [
        "n_dim",
        "latent_dim",
        "n_actions",
        "n_codewords",
        "pop_size",
        "generations",
        "samples_per_gen",
        "eval_samples",
    ]:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    t0 = time.time()
    world = World(cfg)

    conditions = ["selection"]
    if cfg.run_drift:
        conditions.append("drift")
    if cfg.run_random_search:
        conditions.append("random_search")

    results = {}
    for i, mode in enumerate(conditions):
        print(f"running {mode}...")
        results[mode] = run_condition(cfg, world, mode, seed_offset=1000 * (i + 1))
        fit = results[mode]["eval_fit"]
        acc = results[mode]["eval_acc"]
        best_idx = int(np.argmax(fit))
        print(f"  final best normalized viability={fit[best_idx]:.3f}, action accuracy={acc[best_idx]:.3f}")

    runtime = time.time() - t0
    fig_path = plot_results(cfg, results)
    summary_path = write_summary(cfg, results, runtime)
    print(f"wrote {fig_path}")
    print(f"wrote {summary_path}")
    print(f"runtime_seconds={runtime:.1f}")


if __name__ == "__main__":
    main()
