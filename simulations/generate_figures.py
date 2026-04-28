#!/usr/bin/env python3
"""Generate figures for the boundary-code manuscript.

Figure 1: schematic of boundary-code formation with slow-mode conditioning.
Figure 2: one-bit slow-mode code example, with a Delta-sweep panel showing
that the loss-minimizing partition tracks the future-relevant trajectory
coordinate continuously rather than coincidentally rotating to a special axis.
All randomness is deterministic via fixed seeds.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def add_box(ax, xy, wh, text, face, edge="#27313a", fontsize=9):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def add_arrow(ax, start, end, color="#27313a", width=1.4, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


# NOTE: Figure 1 (boundary-code schematic) is now produced by
# fig1_schematic.py. The previous block-diagram version that lived here
# has been removed to avoid silently overwriting the new schematic when
# this script is re-run.


def binary_mutual_information(messages, target):
    messages = np.asarray(messages).astype(int)
    target = np.asarray(target).astype(int)
    mi = 0.0
    for m in (0, 1):
        for f in (0, 1):
            p = np.mean((messages == m) & (target == f))
            if p == 0:
                continue
            pm = np.mean(messages == m)
            pf = np.mean(target == f)
            mi += p * np.log2(p / (pm * pf))
    return mi


def best_one_bit_partition(x, future, angles):
    best_acc = -np.inf
    best_angle = None
    best_msg = None
    for theta in angles:
        projection = np.cos(theta) * x[:, 0] + np.sin(theta) * x[:, 1]
        msg = (projection > 0).astype(int)
        acc = np.mean(msg == future)
        if acc < 0.5:
            msg = 1 - msg
            acc = 1 - acc
        if acc > best_acc:
            best_acc = acc
            best_angle = theta
            best_msg = msg
    return best_angle, best_acc, best_msg


def simulate_substrate(rng, n, noise):
    phi = rng.uniform(0, 2 * np.pi, n)
    x = np.column_stack((np.cos(phi), np.sin(phi))) + rng.normal(0, noise, size=(n, 2))
    return phi, x


def figure_phase_partition():
    rng = np.random.default_rng(11)
    n = 120_000
    noise = 0.42
    delta = np.pi / 2

    phi, x = simulate_substrate(rng, n, noise)
    future = (np.cos(phi + delta) > 0).astype(int)

    current_msg = (x[:, 0] > 0).astype(int)
    random_msg = rng.integers(0, 2, size=n)
    oracle_msg = future.copy()

    angles = np.linspace(0, 2 * np.pi, 721)
    best_angle, best_acc, best_msg = best_one_bit_partition(x, future, angles)

    labels = ["random", "present\naxis", "future-\nrelevant", "oracle"]
    msgs = [random_msg, current_msg, best_msg, oracle_msg]
    accs = [np.mean(m == future) for m in msgs]
    mis = [binary_mutual_information(m, future) for m in msgs]

    # ---- Delta sweep ---------------------------------------------------------
    sweep_rng = np.random.default_rng(29)
    sweep_phi, sweep_x = simulate_substrate(sweep_rng, 60_000, noise)
    deltas = np.linspace(0.0, np.pi, 25)
    sweep_best_acc = np.zeros_like(deltas)
    sweep_best_angle = np.zeros_like(deltas)
    sweep_present_acc = np.zeros_like(deltas)
    for i, d in enumerate(deltas):
        f = (np.cos(sweep_phi + d) > 0).astype(int)
        ang, acc, _ = best_one_bit_partition(sweep_x, f, angles)
        sweep_best_acc[i] = acc
        sweep_best_angle[i] = ang
        present = (sweep_x[:, 0] > 0).astype(int)
        a_p = np.mean(present == f)
        if a_p < 0.5:
            a_p = 1 - a_p
        sweep_present_acc[i] = a_p

    # ---- Composite figure ----------------------------------------------------
    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.95, 1.15])
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_sweep = fig.add_subplot(gs[0, 2])

    # Panel (a): scatter at delta=pi/2 with cuts.
    # Colours encode the future class F_{t+Delta} (explained in caption,
    # not in plot legend or title -- avoids redundancy).
    ax = ax_scatter
    idx = rng.choice(n, size=3200, replace=False)
    colors = np.where(future[idx] == 1, "#1f7770", "#b65a34")
    ax.scatter(x[idx, 0], x[idx, 1], c=colors, s=5, alpha=0.38, linewidths=0)
    lim = 2.15
    t = np.linspace(-lim, lim, 100)
    ax.plot(np.zeros_like(t), t, color="#52616b", lw=1.5, ls="--", label="present-axis cut")
    direction = best_angle + np.pi / 2
    ax.plot(t * np.cos(direction), t * np.sin(direction), color="#101820", lw=2.0,
            label="future-relevant cut")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"substrate coordinate $X_1$")
    ax.set_ylabel(r"substrate coordinate $X_2$")
    ax.set_title(r"(a) Substrate at $\Delta=\pi/2$", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    # Panel (b): bar chart at delta=pi/2
    ax = ax_bars
    xs = np.arange(len(labels))
    bars = ax.bar(xs, accs, color=["#9aa3aa", "#52616b", "#1f7770", "#b65a34"], width=0.72)
    ax.set_ylim(0.45, 1.07)
    ax.set_ylabel("future-class accuracy")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("(b) One-bit code performance", fontsize=10)
    for bar, acc, mi in zip(bars, accs, mis):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            f"{acc:.2f}\n{mi:.2f} bits",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(0.5, color="#27313a", lw=1, ls=":", alpha=0.7)
    ax.text(3.15, 0.505, "chance", ha="right", va="bottom", fontsize=8, color="#27313a")

    # Panel (c): Delta sweep -- two curves plus chance baseline.
    # Angular-error recovery is reported numerically in
    # figures/phase_partition_results.txt rather than as a main-figure
    # element, to keep the figure focused on the pedagogical claim:
    # the present-axis cut fails at nonzero horizon, the future-relevant
    # cut stays high.
    ax = ax_sweep
    ax.plot(
        deltas,
        sweep_best_acc,
        color="#1f7770",
        lw=2.0,
        marker="o",
        ms=4,
        label="future-relevant 1-bit code",
    )
    ax.plot(
        deltas,
        sweep_present_acc,
        color="#52616b",
        lw=1.5,
        marker="s",
        ms=4,
        ls="--",
        label="present-axis code",
    )
    ax.axhline(0.5, color="#27313a", lw=1, ls=":", alpha=0.7, label="chance")
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0.45, 1.05)
    ax.set_xlabel(r"future horizon $\Delta$ (radians)")
    ax.set_ylabel("future-class accuracy")
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title(r"(c) Sweep over $\Delta$", fontsize=10)

    fig.tight_layout(pad=1.0)
    fig.savefig(FIG_DIR / "fig2_phase_partition_demo.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_phase_partition_demo.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    sweep_lines = "\n".join(
        f"delta={d:.4f}\tbest_acc={a:.4f}\tbest_angle={ang:.4f}\tpresent_acc={p:.4f}"
        for d, a, ang, p in zip(deltas, sweep_best_acc, sweep_best_angle, sweep_present_acc)
    )

    result = (
        f"noise={noise:.2f}\n"
        f"delta_radians={delta:.6f}\n"
        f"best_angle_radians={best_angle:.6f}\n"
        f"random_accuracy={accs[0]:.4f}\n"
        f"present_axis_accuracy={accs[1]:.4f}\n"
        f"best_phase_partition_accuracy={accs[2]:.4f}\n"
        f"phase_oracle_accuracy={accs[3]:.4f}\n"
        f"random_mi_bits={mis[0]:.4f}\n"
        f"present_axis_mi_bits={mis[1]:.4f}\n"
        f"best_phase_partition_mi_bits={mis[2]:.4f}\n"
        f"phase_oracle_mi_bits={mis[3]:.4f}\n"
        "\n"
        "# Delta sweep (n=60000 per delta, noise=0.42)\n"
        f"{sweep_lines}\n"
    )
    (FIG_DIR / "phase_partition_results.txt").write_text(result)


def main():
    figure_phase_partition()


if __name__ == "__main__":
    main()
