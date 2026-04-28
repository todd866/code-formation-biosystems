#!/usr/bin/env python3
"""Figure 1 schematic: forced codeword separation under finite reliable capacity.

Renders a two-panel schematic illustrating Proposition 6 (Forced nontrivial code).
Same substrate geometry in both panels; the only visual change is the boundary
capacity and the resulting partition.

  (a) Aliasing: S1 and S2 share a reliable message m. The decoder must choose
      one action delta(m) = a, which is gamma-bad on at least one of S1, S2,
      so regret >= gamma p.
  (b) Distinct reliable messages: S1 -> m1 -> a1 and S2 -> m2 -> a2. The
      decoder can match each regime to its near-optimal action; regret < gamma p
      becomes possible.

Outputs:
  figures/fig1_boundary_code_schematic.pdf
  figures/fig1_boundary_code_schematic.png
"""

from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

# Colour palette
COLOR_S1 = "#d04a4a"  # red
COLOR_S2 = "#2e6fb5"  # blue
COLOR_GRAY = "#9aa4ad"
COLOR_FUNNEL = "#f7e0b6"
COLOR_FUNNEL_EDGE = "#c0892b"
COLOR_BAD = "#7a1a1a"
COLOR_GOOD = "#1a5e2c"


def precompute_substrate_points(seed: int = 7):
    """Compute deterministic dot positions so both panels are identical."""
    rng = np.random.default_rng(seed)
    n_gray = 70
    angles = rng.uniform(0, 2 * np.pi, n_gray)
    radii = np.sqrt(rng.uniform(0, 1, n_gray))
    n_s1 = 18
    n_s2 = 18
    s1_dx = rng.uniform(-1, 1, n_s1)
    s1_dy = rng.uniform(-1, 1, n_s1)
    s2_dx = rng.uniform(-1, 1, n_s2)
    s2_dy = rng.uniform(-1, 1, n_s2)
    return dict(
        gray_angles=angles,
        gray_radii=radii,
        s1_dx=s1_dx,
        s1_dy=s1_dy,
        s2_dx=s2_dx,
        s2_dy=s2_dy,
    )


SUB_POINTS = precompute_substrate_points()


def draw_substrate(ax, x_center=1.7, y_center=2.5, radius_x=1.35, radius_y=1.55):
    """Draw the substrate ellipse with gray points, two coloured regions, and
    optimal-action arrows. Same geometry every call."""
    substrate = mpatches.Ellipse(
        (x_center, y_center),
        2 * radius_x,
        2 * radius_y,
        facecolor="#eef2f7",
        edgecolor="#34495e",
        lw=1.4,
        zorder=0,
    )
    ax.add_patch(substrate)

    # Gray "other" points
    gx = x_center + 0.92 * radius_x * SUB_POINTS["gray_radii"] * np.cos(SUB_POINTS["gray_angles"])
    gy = y_center + 0.92 * radius_y * SUB_POINTS["gray_radii"] * np.sin(SUB_POINTS["gray_angles"])
    ax.scatter(gx, gy, c=COLOR_GRAY, s=8, alpha=0.45, zorder=1, edgecolors="none")

    # S1 region (red, lower-right)
    s1_c = (x_center + 0.40, y_center - 0.75)
    s1_patch = mpatches.Ellipse(
        s1_c, 0.95, 0.55, facecolor=COLOR_S1, alpha=0.22,
        edgecolor=COLOR_S1, lw=1.2, zorder=2,
    )
    ax.add_patch(s1_patch)
    ax.scatter(
        s1_c[0] + 0.42 * SUB_POINTS["s1_dx"],
        s1_c[1] + 0.24 * SUB_POINTS["s1_dy"],
        c=COLOR_S1, s=12, zorder=3, edgecolors="none",
    )
    ax.text(s1_c[0], s1_c[1], "$S_1$", fontsize=11, ha="center", va="center",
            fontweight="bold", color="#5a0f0f", zorder=4)
    # Optimal action a1*: east arrow
    ax.annotate(
        "", xy=(s1_c[0] + 0.85, s1_c[1] - 0.55),
        xytext=(s1_c[0] + 0.15, s1_c[1] - 0.55),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_S1, lw=1.6, mutation_scale=12),
    )
    ax.text(s1_c[0] + 0.5, s1_c[1] - 0.78, "$a_1^{\\star}$", fontsize=9,
            ha="center", color=COLOR_S1)

    # S2 region (blue, upper-left)
    s2_c = (x_center - 0.45, y_center + 0.75)
    s2_patch = mpatches.Ellipse(
        s2_c, 0.85, 0.55, facecolor=COLOR_S2, alpha=0.22,
        edgecolor=COLOR_S2, lw=1.2, zorder=2,
    )
    ax.add_patch(s2_patch)
    ax.scatter(
        s2_c[0] + 0.38 * SUB_POINTS["s2_dx"],
        s2_c[1] + 0.24 * SUB_POINTS["s2_dy"],
        c=COLOR_S2, s=12, zorder=3, edgecolors="none",
    )
    ax.text(s2_c[0], s2_c[1], "$S_2$", fontsize=11, ha="center", va="center",
            fontweight="bold", color="#0d2c52", zorder=4)
    # Optimal action a2*: north arrow, placed ABOVE the substrate to avoid overlap
    ax.annotate(
        "", xy=(s2_c[0] - 0.95, y_center + radius_y + 0.35),
        xytext=(s2_c[0] - 0.95, y_center + radius_y - 0.15),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_S2, lw=1.6, mutation_scale=12),
    )
    ax.text(s2_c[0] - 0.55, y_center + radius_y + 0.10, "$a_2^{\\star}$", fontsize=9,
            va="center", color=COLOR_S2)

    # Substrate label
    ax.text(x_center, y_center - radius_y - 0.45,
            "high-D substrate $X$", fontsize=10, ha="center", color="#34495e")

    return s1_c, s2_c


def draw_funnel(ax, x_left, x_right, y_top, y_bottom, neck_width):
    """Draw a labelled funnel/bottleneck.  Neck width encodes capacity."""
    y_mid = (y_top + y_bottom) / 2
    poly = mpatches.Polygon(
        [
            (x_left, y_top),
            (x_right, y_top),
            (x_right - 0.18, y_mid + neck_width / 2),
            (x_right - 0.18, y_mid - neck_width / 2),
            (x_right, y_bottom),
            (x_left, y_bottom),
            (x_left + 0.18, y_mid - neck_width / 2),
            (x_left + 0.18, y_mid + neck_width / 2),
        ],
        facecolor=COLOR_FUNNEL,
        edgecolor=COLOR_FUNNEL_EDGE,
        lw=1.4,
        zorder=0,
    )
    ax.add_patch(poly)


def draw_puck(ax, x, y, label, color, alpha=0.5):
    puck = mpatches.Circle(
        (x, y), 0.30, facecolor=color, alpha=alpha,
        edgecolor=color, lw=1.6, zorder=2,
    )
    ax.add_patch(puck)
    ax.text(x, y, label, fontsize=11, ha="center", va="center",
            fontweight="bold", color="#1c1c1c", zorder=3)


def panel_aliasing(ax):
    """Panel (a): S1 and S2 alias to a single reliable message m."""
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")

    s1_c, s2_c = draw_substrate(ax)

    # Funnel with very narrow neck (K=1) -- arrows pass THROUGH it
    draw_funnel(ax, x_left=3.7, x_right=5.0, y_top=3.6, y_bottom=1.4, neck_width=0.25)
    ax.text(4.35, 3.95, "boundary  ($K=1$)", fontsize=10, ha="center", color="#7a5012")

    # Arrows from S1, S2 converge through the funnel neck (single codeword)
    neck_x = 4.35
    neck_y = 2.5
    ax.annotate(
        "", xy=(neck_x, neck_y), xytext=(s1_c[0] + 0.55, s1_c[1]),
        arrowprops=dict(arrowstyle="-", color=COLOR_S1, lw=1.5,
                        connectionstyle="arc3,rad=0.10"),
    )
    ax.annotate(
        "", xy=(neck_x, neck_y), xytext=(s2_c[0] + 0.55, s2_c[1]),
        arrowprops=dict(arrowstyle="-", color=COLOR_S2, lw=1.5,
                        connectionstyle="arc3,rad=-0.10"),
    )
    # Continuation from funnel exit to single codeword puck
    draw_puck(ax, 6.5, 2.5, "$m$", "#7f8c8d", alpha=0.5)
    ax.annotate(
        "", xy=(6.20, 2.5), xytext=(neck_x + 0.20, 2.5),
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6, mutation_scale=12),
    )

    # Decoder: label ABOVE arrow, arrow itself east-pointing
    ax.text(8.95, 2.85, "$\\delta(m) = a$", fontsize=11, ha="center", va="bottom",
            color="#3a3a3a")
    ax.annotate(
        "", xy=(9.4, 2.5), xytext=(8.5, 2.5),
        arrowprops=dict(arrowstyle="-|>", color="#3a3a3a", lw=1.8, mutation_scale=14),
    )

    # Regret bracket on the right
    bx = 9.7
    ax.plot([bx, bx + 0.18, bx + 0.18, bx], [1.6, 1.6, 3.4, 3.4], color=COLOR_BAD, lw=1.4)
    ax.text(bx + 0.30, 2.5,
            "regret\n$\\geq \\gamma p$\non $S_1$ or $S_2$",
            fontsize=9.5, ha="left", va="center", color=COLOR_BAD)

    # Panel title
    ax.text(5.5, 4.7,
            "(a) Aliasing: $S_1, S_2$ share a reliable message",
            fontsize=11.5, ha="center", fontweight="bold", color=COLOR_BAD)


def panel_split(ax):
    """Panel (b): S1 -> m1, S2 -> m2 (distinct reliable messages)."""
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")

    s1_c, s2_c = draw_substrate(ax)

    # Funnel with wider neck (K >= 2) -- arrows pass THROUGH at distinct heights
    draw_funnel(ax, x_left=3.7, x_right=5.0, y_top=3.6, y_bottom=1.4, neck_width=1.4)
    ax.text(4.35, 3.95, "boundary  ($K \\geq 2$)", fontsize=10, ha="center", color="#7a5012")

    # Two distinct codewords; arrows pass through the funnel and continue to them
    draw_puck(ax, 6.5, 3.3, "$m_2$", COLOR_S2, alpha=0.45)
    draw_puck(ax, 6.5, 1.7, "$m_1$", COLOR_S1, alpha=0.45)

    # Arrows from substrate -> funnel exit (distinct y) -> codeword
    ax.annotate(
        "", xy=(6.20, 1.7), xytext=(s1_c[0] + 0.55, s1_c[1]),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_S1, lw=1.5, mutation_scale=12,
                        connectionstyle="arc3,rad=0.06"),
    )
    ax.annotate(
        "", xy=(6.20, 3.3), xytext=(s2_c[0] + 0.55, s2_c[1]),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_S2, lw=1.5, mutation_scale=12,
                        connectionstyle="arc3,rad=-0.06"),
    )

    # Decoder labels ABOVE their arrows (no strike-through)
    ax.text(8.95, 3.65, "$\\delta(m_2) = a_2$", fontsize=11, ha="center", va="bottom", color=COLOR_S2)
    ax.annotate(
        "", xy=(9.4, 3.3), xytext=(8.5, 3.3),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_S2, lw=1.8, mutation_scale=14),
    )
    ax.text(8.95, 2.05, "$\\delta(m_1) = a_1$", fontsize=11, ha="center", va="bottom", color=COLOR_S1)
    ax.annotate(
        "", xy=(9.4, 1.7), xytext=(8.5, 1.7),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_S1, lw=1.8, mutation_scale=14),
    )

    # Regret bracket
    bx = 9.7
    ax.plot([bx, bx + 0.18, bx + 0.18, bx], [1.0, 1.0, 4.0, 4.0], color=COLOR_GOOD, lw=1.4)
    ax.text(bx + 0.30, 2.5,
            "regret\n$< \\gamma p$\npossible",
            fontsize=9.5, ha="left", va="center", color=COLOR_GOOD)

    # Panel title
    ax.text(5.5, 4.7,
            "(b) Distinct reliable messages: $K \\geq r$",
            fontsize=11.5, ha="center", fontweight="bold", color=COLOR_GOOD)


def main():
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), gridspec_kw=dict(hspace=0.10))
    panel_aliasing(axes[0])
    panel_split(axes[1])

    out_pdf = FIG_DIR / "fig1_boundary_code_schematic.pdf"
    out_png = FIG_DIR / "fig1_boundary_code_schematic.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
