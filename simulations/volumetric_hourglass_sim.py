"""Volumetric / hourglass code-formation simulator (PyTorch + MPS).

Trains an L-layer hourglass channel architecture over a high-dimensional
substrate with slow-mode-conditioned future task. Uses gradient descent
through straight-through argmax (VQ-VAE-style) so the credit signal can
propagate through the whole chain — pure evolution can't bridge >3-4
discrete layers.

Connection to the paper: this is a numerical instantiation of the
alternating-improvement dynamics of Prop 4 (Lloyd--Bayes), where the
encoder-step and decoder-step are interleaved by simultaneous SGD on a
shared end-to-end objective. Discrete codewords on the forward pass;
soft gradient flow on the backward pass.

Outputs per-layer per-iteration: D_eff, codeword usage entropy, MI(M;F),
MI(M;phase_k for each slow mode), per-class fiber loss spread.

Designed to run unattended for ~5h on Apple M5 (MPS).
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------ configuration ------------------------------ #


@dataclass
class Config:
    # latent / substrate
    n_phase_modes: int = 4
    n_dev_modes: int = 4
    n_noise_modes: int = 4
    phase_taus: tuple = (5.0, 20.0, 60.0, 200.0)
    delta: float = 10.0
    n_bulk: int = 2048
    embed_hidden: int = 96

    # task
    r_task: int = 8
    task_phase_mode: int = 3   # longest-tau phase decides F

    # hourglass: single discrete waist, continuous radial profile around it.
    # width_profile gives the per-layer continuous representation dimension N_r.
    # Encoder = chain of Linear+Tanh from N_bulk down through this profile;
    # then a single discrete bottleneck of K codewords at the waist; then
    # decoder = chain of Linear+Tanh expanding back; then R-way classifier.
    width_profile: tuple = (1024, 512, 256, 128, 64, 32, 16, 32, 64, 128, 256, 512)
    waist_k: int = 16
    # Gumbel-Softmax temperature schedule for the waist:
    tau_init: float = 1.0
    tau_min: float = 0.5
    anneal_frac: float = 0.4

    # training
    n_seeds: int = 5
    n_iters: int = 18000
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0

    # logging / runtime
    log_every: int = 500
    measure_every: int = 250
    eval_batch: int = 2000
    seed: int = 20260429
    run_tag: str = "volumetric_hourglass_v1"
    device: str = "mps"


# ------------------------------ substrate model ---------------------------- #


class GenerativeModel:
    """Latent slow-mode + nonlinear bulk embedding (NumPy on CPU then transferred).

    The substrate weights are fixed for the run so all seeds see the
    same generative process. Each batch returns (X0, F, phase_class).
    """

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.lift_dim = 2 * cfg.n_phase_modes + cfg.n_dev_modes + cfg.n_noise_modes
        self.W1 = rng.standard_normal(
            (cfg.embed_hidden, self.lift_dim)
        ).astype(np.float32) / np.sqrt(self.lift_dim)
        self.q_dim = self.lift_dim * (self.lift_dim + 1) // 2
        self.W2 = rng.standard_normal(
            (cfg.n_bulk, cfg.embed_hidden)
        ).astype(np.float32) / np.sqrt(cfg.embed_hidden)
        self.W3 = rng.standard_normal(
            (cfg.n_bulk, self.q_dim)
        ).astype(np.float32) / np.sqrt(self.q_dim)
        i, j = np.triu_indices(self.lift_dim)
        self.qi = i.astype(np.int32)
        self.qj = j.astype(np.int32)
        self.task_mode = cfg.task_phase_mode

    def sample(self, rng: np.random.Generator, S: int):
        cfg = self.cfg
        phases_now = rng.uniform(0.0, 2 * np.pi, (S, cfg.n_phase_modes)).astype(np.float32)
        taus = np.asarray(cfg.phase_taus, dtype=np.float32)
        sigma = np.sqrt(cfg.delta / taus)
        drift = rng.standard_normal((S, cfg.n_phase_modes)).astype(np.float32) * sigma
        phases_future = phases_now + drift
        dev = rng.uniform(-1.0, 1.0, (S, cfg.n_dev_modes)).astype(np.float32)
        noise = rng.standard_normal((S, cfg.n_noise_modes)).astype(np.float32)
        cs = np.cos(phases_now); sn = np.sin(phases_now)
        z = np.concatenate([cs, sn, dev, noise], axis=1).astype(np.float32)
        h1 = np.tanh(z @ self.W1.T)
        q = z[:, self.qi] * z[:, self.qj]
        X0 = np.tanh(h1 @ self.W2.T + q @ self.W3.T).astype(np.float32)
        f_phase = phases_future[:, self.task_mode] % (2 * np.pi)
        F_target = (f_phase * cfg.r_task / (2 * np.pi)).astype(np.int64)
        F_target = np.clip(F_target, 0, cfg.r_task - 1)
        phase_class_now = (
            (phases_now % (2 * np.pi)) * cfg.r_task / (2 * np.pi)
        ).astype(np.int64)
        phase_class_now = np.clip(phase_class_now, 0, cfg.r_task - 1)
        return X0, F_target, phase_class_now


# ------------------------------ model -------------------------------------- #


class VQLayer(nn.Module):
    """Gumbel-Softmax discrete bottleneck.

    Encoder produces K logits. With straight-through Gumbel-Softmax:
    forward = one-hot of argmax(logits + Gumbel noise), backward = soft
    Gumbel-Softmax gradient at temperature tau. Decoder is an embedding
    table: one-hot index selects a codeword vector in embed_dim.
    Temperature annealed externally via set_tau().
    """

    def __init__(self, n_in: int, K: int, embed_dim: int):
        super().__init__()
        self.K = K
        self.embed_dim = embed_dim
        self.encoder = nn.Linear(n_in, K)
        self.decoder = nn.Embedding(K, embed_dim)
        nn.init.normal_(self.decoder.weight, std=0.5)
        # tau is set per forward via the model's annealing schedule
        self.register_buffer("tau", torch.tensor(1.0))

    def set_tau(self, tau: float):
        self.tau.fill_(tau)

    def forward(self, x: torch.Tensor):
        logits = self.encoder(x)                                         # (S, K)
        tau = float(self.tau.clamp_min(1e-3).item())
        if self.training:
            # Use torch.nn.functional.gumbel_softmax — well-tested
            soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            soft = F.softmax(logits / tau, dim=-1)
        m = soft.argmax(dim=-1)                                          # (S,)
        hard = F.one_hot(m, num_classes=self.K).to(soft.dtype)
        ste = soft + (hard - soft).detach()                              # (S, K)
        # decoded codeword (forward = hard pick, backward via soft)
        z_q = ste @ self.decoder.weight                                  # (S, D)
        # auxiliary diversity loss (discourage codebook collapse): KL(uniform || P(m))
        # encourages encoder to spread mass across codewords
        avg_p = soft.mean(dim=0)                                         # (K,)
        diversity = (avg_p * (avg_p.clamp_min(1e-9).log()
                              + math.log(self.K))).sum()
        return z_q, m, diversity, torch.zeros((), device=z_q.device, dtype=z_q.dtype)


class HourglassModel(nn.Module):
    """Hourglass: continuous Linear+Tanh chain down to waist, single discrete
    bottleneck (Gumbel-Softmax K-way), continuous Linear+Tanh chain back up,
    final R-way classifier head.

    The volumetric story: D_eff(r) is measured at every continuous layer.
    Bulk has high D_eff. Encoder layers compress smoothly. Waist's discrete
    code sits inside the K-codeword embedding. Decoder re-expands.

    cfg.width_profile gives encoder widths bulk -> waist (length L_e+1) then
    decoder widths waist -> output (length L_d+1). The waist is at the
    minimum-width index.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        widths = list(cfg.width_profile)
        # find waist (minimum width) — split point
        waist_idx = int(np.argmin(widths))
        self.waist_idx = waist_idx
        self.waist_dim = widths[waist_idx]
        self.K = cfg.waist_k
        # encoder: bulk -> ... -> waist_dim
        enc_dims = [cfg.n_bulk] + widths[: waist_idx + 1]
        self.encoder = nn.ModuleList()
        for i in range(len(enc_dims) - 1):
            self.encoder.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
        # discrete bottleneck: waist_dim -> K logits, with Gumbel STE + lookup
        self.waist_logits = nn.Linear(self.waist_dim, self.K)
        self.codebook = nn.Embedding(self.K, self.waist_dim)
        nn.init.normal_(self.codebook.weight, std=0.5)
        # decoder: waist_dim -> ... -> last_width -> R-way classifier
        dec_dims = widths[waist_idx:] + [cfg.r_task]
        self.decoder = nn.ModuleList()
        for i in range(len(dec_dims) - 1):
            self.decoder.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
        # tau buffer (for Gumbel)
        self.register_buffer("tau", torch.tensor(1.0))

    def set_tau(self, tau: float):
        self.tau.fill_(tau)

    def forward(self, x: torch.Tensor):
        """Returns final_logits, codes_list, states_list, aux_loss.

        codes_list: length 1, just the waist codeword index per sample.
        states_list: continuous representations at every depth (encoder
                     outputs + waist + decoder outputs).
        """
        states = [x]
        h = x
        # encoder
        for lin in self.encoder:
            h = torch.tanh(lin(h))
            states.append(h)
        # discrete waist
        logits = self.waist_logits(h)
        tau = float(self.tau.clamp_min(1e-3).item())
        if self.training:
            soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            soft = F.softmax(logits / tau, dim=-1)
        m = soft.argmax(dim=-1)
        hard = F.one_hot(m, num_classes=self.K).to(soft.dtype)
        ste = soft + (hard - soft).detach()                                     # (S, K)
        z_waist = ste @ self.codebook.weight                                    # (S, waist_dim)
        # the waist quantized state replaces h going into decoder
        states.append(z_waist)
        h = z_waist
        # decoder layers (last linear is the classifier; no tanh on it)
        for i, lin in enumerate(self.decoder):
            h = lin(h)
            if i < len(self.decoder) - 1:
                h = torch.tanh(h)
                states.append(h)
        final_logits = h
        codes_list = [m]
        aux = torch.zeros((), device=x.device, dtype=x.dtype)
        return final_logits, codes_list, states, aux


# ------------------------------ measurements ------------------------------- #


def participation_ratio(X: np.ndarray) -> float:
    if X.ndim != 2 or X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(Xc.astype(np.float64), compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    lam = s ** 2
    denom = float((lam ** 2).sum())
    if denom == 0:
        return 0.0
    return float((lam.sum() ** 2) / denom)


def usage_entropy(codes: np.ndarray, K: int) -> float:
    counts = np.bincount(codes, minlength=K).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    h = -float((p * np.log2(p)).sum())
    return h / math.log2(K) if K > 1 else 0.0


def mi_discrete(codes: np.ndarray, labels: np.ndarray, K: int, R: int) -> float:
    S = codes.shape[0]
    joint = np.zeros((K, R), dtype=np.float64)
    np.add.at(joint, (codes, labels), 1.0)
    joint /= S
    p_m = joint.sum(axis=1, keepdims=True)
    p_l = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    out = np.where(mask, joint * np.log2(joint / (p_m * p_l + 1e-30) + 1e-30), 0.0)
    return float(out.sum())


def measure(model: HourglassModel, gm: GenerativeModel, rng: np.random.Generator,
            cfg: Config, device: torch.device) -> dict:
    """Compute radial profile measurements for the volumetric architecture.

    The volumetric story has one discrete code (at the waist) and many
    continuous layers. Measurements:
      - acc on future task
      - D_eff(depth) at every continuous layer + at the waist
      - waist codeword usage entropy and MI(M_waist; F), MI(M_waist; phase_k)
    """
    model.train(False)
    with torch.no_grad():
        X0_np, F_np, pc_np = gm.sample(rng, cfg.eval_batch)
        x = torch.from_numpy(X0_np).to(device)
        F_t = torch.from_numpy(F_np).to(device)
        final_logits, codes, states, _aux = model(x)
        acc = (final_logits.argmax(dim=-1) == F_t).float().mean().item()
        m_waist = codes[0].detach().cpu().numpy()
        states_np = [s.detach().cpu().numpy() for s in states]
    n_depths = len(states_np)
    d_eff = np.array(
        [participation_ratio(s) for s in states_np], dtype=np.float32
    )
    K = model.K
    H_waist = usage_entropy(m_waist, K)
    mi_F_waist = mi_discrete(m_waist, F_np, K, cfg.r_task)
    mi_phase_waist = np.array(
        [mi_discrete(m_waist, pc_np[:, k], K, cfg.r_task)
         for k in range(cfg.n_phase_modes)],
        dtype=np.float32,
    )
    model.train(True)
    return {
        "acc": float(acc),
        "d_eff": d_eff,                    # (n_depths,)
        "H_waist": float(H_waist),
        "mi_F_waist": float(mi_F_waist),
        "mi_phase_waist": mi_phase_waist,  # (n_phase_modes,)
        "n_depths": n_depths,
    }


# ------------------------------ training ----------------------------------- #


def train_seed(cfg: Config, gm: GenerativeModel, seed: int, run_dir: Path,
               log) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(cfg.device)
    model = HourglassModel(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # number of continuous depths in the model
    # = 1 (bulk) + len(encoder) + 1 (waist) + (len(decoder) - 1)
    L_e = len(model.encoder)
    L_d = len(model.decoder)
    n_depths = 1 + L_e + 1 + (L_d - 1)
    n_meas = cfg.n_iters // cfg.measure_every + 1
    history = {
        "iter": np.zeros(n_meas, dtype=np.int32),
        "loss": np.zeros(n_meas, dtype=np.float32),
        "acc": np.zeros(n_meas, dtype=np.float32),
        "d_eff": np.zeros((n_meas, n_depths), dtype=np.float32),
        "H_waist": np.zeros(n_meas, dtype=np.float32),
        "mi_F_waist": np.zeros(n_meas, dtype=np.float32),
        "mi_phase_waist": np.zeros((n_meas, cfg.n_phase_modes), dtype=np.float32),
    }

    measure_idx = 0
    t_seed_start = time.time()
    running_loss = 0.0
    running_acc = 0.0
    running_n = 0

    anneal_steps = max(1, int(cfg.anneal_frac * cfg.n_iters))
    for it in range(cfg.n_iters):
        # tau anneal: linear decay from tau_init to tau_min over first
        # anneal_frac of training, then constant.
        if it < anneal_steps:
            frac = it / anneal_steps
            tau = cfg.tau_init + (cfg.tau_min - cfg.tau_init) * frac
        else:
            tau = cfg.tau_min
        model.set_tau(tau)

        X0_np, F_np, _ = gm.sample(rng, cfg.batch_size)
        x = torch.from_numpy(X0_np).to(device)
        F_t = torch.from_numpy(F_np).to(device)
        final_logits, codes, states, aux = model(x)
        ce = F.cross_entropy(final_logits, F_t)
        loss = ce + aux
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        with torch.no_grad():
            acc = (final_logits.argmax(dim=-1) == F_t).float().mean().item()
        running_loss += float(loss.item())
        running_acc += acc
        running_n += 1

        if (it % cfg.measure_every == 0) or (it == cfg.n_iters - 1):
            m = measure(model, gm, rng, cfg, device)
            history["iter"][measure_idx] = it
            history["loss"][measure_idx] = running_loss / max(1, running_n)
            history["acc"][measure_idx] = m["acc"]
            history["d_eff"][measure_idx] = m["d_eff"]
            history["H_waist"][measure_idx] = m["H_waist"]
            history["mi_F_waist"][measure_idx] = m["mi_F_waist"]
            history["mi_phase_waist"][measure_idx] = m["mi_phase_waist"]
            measure_idx += 1

        if (it % cfg.log_every == 0) or (it == cfg.n_iters - 1):
            now = time.time()
            elapsed = now - t_seed_start
            iter_per_s = (it + 1) / max(1e-6, elapsed)
            eta = (cfg.n_iters - it - 1) / max(1e-6, iter_per_s)
            log(
                f"  seed {seed}  it {it:6d}/{cfg.n_iters}  "
                f"loss={running_loss/max(1,running_n):.3f}  "
                f"acc={running_acc/max(1,running_n):.3f}  "
                f"meas_acc={history['acc'][max(0,measure_idx-1)]:.3f}  "
                f"H_waist={history['H_waist'][max(0,measure_idx-1)]:.3f}  "
                f"MI_F_waist={history['mi_F_waist'][max(0,measure_idx-1)]:.3f}  "
                f"D_waist={history['d_eff'][max(0,measure_idx-1), model.waist_idx + 1]:.2f}  "
                f"{iter_per_s:.1f}it/s  ETA={eta/60:.1f}m"
            )
            running_loss = 0.0; running_acc = 0.0; running_n = 0

    for k in history:
        history[k] = history[k][:measure_idx]
    state_dict = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.savez(seed_dir / "history.npz", **history)
    np.savez(seed_dir / "model_state.npz", **state_dict)
    log(f"  seed {seed} done. final acc={history['acc'][-1]:.3f}, "
        f"wall={(time.time()-t_seed_start)/60:.1f}m")
    return history


def run(cfg: Config, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    log_f = open(log_path, "w", buffering=1)

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    cfg_dict = asdict(cfg)
    cfg_dict["width_profile"] = list(cfg.width_profile)
    cfg_dict["phase_taus"] = list(cfg.phase_taus)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    log(f"run_dir = {run_dir}")
    log(f"cfg = {cfg_dict}")
    log(f"torch={torch.__version__} device={cfg.device}")

    rng_gm = np.random.default_rng(cfg.seed)
    gm = GenerativeModel(cfg, rng_gm)
    log(f"width_profile={list(cfg.width_profile)}, K_waist={cfg.waist_k}, "
        f"n_seeds={cfg.n_seeds}, n_iters={cfg.n_iters}, batch={cfg.batch_size}")

    t0 = time.time()
    all_histories = []
    for s in range(cfg.n_seeds):
        seed = cfg.seed + 1000 * s
        log(f"=== seed {s+1}/{cfg.n_seeds} (torch_seed={seed}) ===")
        h = train_seed(cfg, gm, seed, run_dir, log)
        all_histories.append(h)
        agg = {}
        for key in ["iter", "loss", "acc", "d_eff",
                    "H_waist", "mi_F_waist", "mi_phase_waist"]:
            stacked = np.stack([hh[key] for hh in all_histories], axis=0)
            agg[key] = stacked
        np.savez(run_dir / "aggregate_history.npz", **agg)

    log(f"=== ALL DONE. total wall = {(time.time()-t0)/3600:.2f}h ===")
    log_f.close()


# ------------------------------ entry ------------------------------------- #


def smoke():
    cfg = Config(
        n_bulk=128, embed_hidden=16,
        width_profile=(64, 32, 16, 32, 64),
        waist_k=8,
        n_seeds=1, n_iters=600,
        batch_size=512, log_every=100, measure_every=50,
        eval_batch=1024, run_tag="smoke", lr=1e-3,
    )
    run_dir = Path(__file__).resolve().parent.parent / "runs" / "smoke_torch"
    if run_dir.exists():
        for p in sorted(run_dir.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
    run(cfg, run_dir)


# Sweep design: tests Prop-6 cardinality floor at waist + volumetric structure.
# R=8 is the task alphabet, so the cardinality floor is K_waist >= 8.
# L=7 channels: 6 VQ layers + 1 classifier head.
# Sweep: vary the discrete-waist alphabet K, holding the continuous
# encoder/decoder widths (the radial profile) fixed. Tests Prop-6
# cardinality floor (R=8) at the waist while keeping the volumetric
# profile identical across conditions.
SWEEP_WIDTHS = (1024, 512, 256, 128, 64, 32, 16, 32, 64, 128, 256, 512)
SWEEP_CONDITIONS = [
    # name, waist_k, n_seeds, n_iters
    ("waist_K_4_below_floor",   4,  6, 15000),
    ("waist_K_8_at_floor",      8,  6, 15000),
    ("waist_K_16_2x_floor",    16,  6, 15000),
    ("waist_K_32_4x_floor",    32,  6, 15000),
    ("waist_K_64_8x_floor",    64,  6, 15000),
    ("waist_K_256_huge",      256,  4, 15000),
    ("waist_K_2048_eff_continuous", 2048, 4, 15000),
]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = Path(__file__).resolve().parent.parent / "runs"
        sweep_dir = base / f"{ts}_volumetric_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        with open(sweep_dir / "manifest.json", "w") as f:
            json.dump(
                {
                    "width_profile": list(SWEEP_WIDTHS),
                    "conditions": [
                        {"name": n, "waist_k": k,
                         "n_seeds": s, "n_iters": it}
                        for (n, k, s, it) in SWEEP_CONDITIONS
                    ],
                },
                f, indent=2,
            )
        for (name, waist_k, n_seeds, n_iters) in SWEEP_CONDITIONS:
            cfg = Config(
                width_profile=SWEEP_WIDTHS,
                waist_k=waist_k,
                n_seeds=n_seeds,
                n_iters=n_iters,
                batch_size=2048,
                log_every=500,
                measure_every=200,
                run_tag=name,
            )
            cfg_dir = sweep_dir / name
            run(cfg, cfg_dir)
        print(f"sweep complete: {sweep_dir}")
        return

    cfg = Config()
    if len(sys.argv) > 1 and sys.argv[1] == "big":
        cfg.n_seeds = 8
        cfg.n_iters = 35000
        cfg.batch_size = 2048
        cfg.run_tag = "volumetric_hourglass_big"

    ts = time.strftime("%Y%m%d_%H%M%S")
    base = Path(__file__).resolve().parent.parent / "runs"
    run_dir = base / f"{ts}_{cfg.run_tag}"
    run(cfg, run_dir)


if __name__ == "__main__":
    main()
