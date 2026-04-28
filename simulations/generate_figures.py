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


def figure_boundary_code():
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rng = np.random.default_rng(3)

    # High-dimensional substrate represented by many microstates.
    substrate = Circle((0.16, 0.52), 0.17, facecolor="#d9edf2", edgecolor="#1f5663", lw=1.4)
    ax.add_patch(substrate)
    pts = rng.normal(size=(135, 2))
    pts = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-8)
    r = np.sqrt(rng.uniform(0, 1, size=(135, 1))) * 0.145
    pts = np.array([0.16, 0.52]) + pts * r
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color="#1f5663", alpha=0.75, linewidths=0)
    ax.text(
        0.16,
        0.28,
        r"high-dimensional" "\n" r"substrate $X$",
        ha="center",
        va="center",
        fontsize=9,
    )

    add_box(
        ax,
        (0.36, 0.39),
        (0.10, 0.26),
        "finite\nboundary\nchannel",
        "#f4dfcf",
        fontsize=8.5,
    )
    add_box(
        ax,
        (0.55, 0.34),
        (0.14, 0.36),
        "fiber\npartition\n" + r"$M=\pi(X)$" + "\n" + r"$\{m_1,m_2,m_3\}$",
        "#e6e2f5",
        fontsize=8.5,
    )
    add_box(
        ax,
        (0.77, 0.39),
        (0.13, 0.26),
        "decoder\nresponse\n" + r"$\delta(M)$",
        "#dcebd7",
        fontsize=8.5,
    )

    add_arrow(ax, (0.33, 0.52), (0.36, 0.52))
    add_arrow(ax, (0.46, 0.52), (0.55, 0.52))
    add_arrow(ax, (0.69, 0.52), (0.77, 0.52))

    # Slow mode shown as a long trajectory feeding the partition and decoder.
    xs = np.linspace(0.08, 0.91, 300)
    ys = 0.84 + 0.035 * np.sin(2 * np.pi * 2.2 * xs)
    ax.plot(xs, ys, color="#9b3d2f", lw=2)
    ax.text(
        0.50,
        0.93,
        r"slow-mode trajectory coordinate $\phi$",
        ha="center",
        va="center",
        fontsize=9,
    )
    add_arrow(ax, (0.52, 0.82), (0.60, 0.70), color="#9b3d2f", width=1.1, rad=-0.2)
    add_arrow(ax, (0.77, 0.82), (0.82, 0.65), color="#9b3d2f", width=1.1, rad=-0.2)

    ax.text(
        0.50,
        0.12,
        "capacity forces projection; viability loss selects reusable action-relevant fibers",
        ha="center",
        va="center",
        fontsize=9,
        color="#27313a",
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "fig1_boundary_code_schematic.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_boundary_code_schematic.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


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

    labels = ["random", "present\naxis", "best\n1-bit", "oracle"]
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

    # Panel (a): scatter at delta=pi/2 with cuts
    ax = ax_scatter
    idx = rng.choice(n, size=3200, replace=False)
    colors = np.where(future[idx] == 1, "#1f7770", "#b65a34")
    ax.scatter(x[idx, 0], x[idx, 1], c=colors, s=5, alpha=0.38, linewidths=0)
    lim = 2.15
    t = np.linspace(-lim, lim, 100)
    ax.plot(np.zeros_like(t), t, color="#52616b", lw=1.5, ls="--", label="present-axis cut")
    direction = best_angle + np.pi / 2
    ax.plot(t * np.cos(direction), t * np.sin(direction), color="#101820", lw=2.0, label="best 1-bit cut")
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

    # Panel (c): Delta sweep
    ax = ax_sweep
    ax.plot(
        deltas,
        sweep_best_acc,
        color="#1f7770",
        lw=2.0,
        marker="o",
        ms=4,
        label="best 1-bit",
    )
    ax.plot(
        deltas,
        sweep_present_acc,
        color="#52616b",
        lw=1.5,
        marker="s",
        ms=4,
        ls="--",
        label="present axis",
    )
    ax.axhline(0.5, color="#27313a", lw=1, ls=":", alpha=0.7)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0.45, 1.05)
    ax.set_xlabel(r"future horizon $\Delta$ (radians)")
    ax.set_ylabel("future-class accuracy")
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title(r"(c) Sweep over $\Delta$", fontsize=10)

    # Overlay the angular error of the recovered projection-vector angle
    # relative to the theoretical optimum -Delta. The partition
    # (cos t x1 + sin t x2 > 0) is invariant under t -> t + pi (label
    # flip), so the error is wrapped into (-pi/2, pi/2]. Near zero across
    # the sweep means the search recovers the theory.
    expected = (-deltas) % np.pi
    raw_diff = sweep_best_angle - expected
    angular_error = ((raw_diff + np.pi / 2) % np.pi) - np.pi / 2
    ax2 = ax.twinx()
    ax2.plot(
        deltas,
        angular_error,
        color="#b65a34",
        lw=1.4,
        marker="^",
        ms=3.5,
        alpha=0.85,
        label="angular error",
    )
    ax2.axhline(0, color="#b65a34", lw=0.8, ls=":", alpha=0.5)
    ax2.set_ylabel(
        "angular error from optimum (radians)", color="#b65a34"
    )
    ax2.tick_params(axis="y", colors="#b65a34")
    ax2.set_ylim(-np.pi / 8, np.pi / 8)
    ax2.set_yticks([-np.pi / 8, -np.pi / 16, 0, np.pi / 16, np.pi / 8])
    ax2.set_yticklabels(
        [r"$-\pi/8$", r"$-\pi/16$", "0", r"$\pi/16$", r"$\pi/8$"]
    )

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
    figure_boundary_code()
    figure_phase_partition()


if __name__ == "__main__":
    main()
