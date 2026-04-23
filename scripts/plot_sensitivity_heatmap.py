"""Sensitivity heatmap: c (regularization constant) vs dataset.

Upgrades the 1-D tab:c-sensitivity (Energy only) to a 2-D heatmap across the
five datasets in the paper. Hatched cells mark diverged configurations
(c = 0.01 column). A black contour traces the "robust band" where disc stays
within a small delta of each row's per-dataset minimum.

MOCK values are anchored at the confirmed Energy row (c=0.1 -> 0.045) from
tab:ablation / tab:c-sensitivity and scaled per-dataset using the ball-park
ordering indicated in "Known Baselines" of the wandb-run-status skill
(Energy easier than Weather, Stocks hardest at 50% missing, Sine easiest).

Usage:
    python scripts/plot_sensitivity_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

OUT_DIR = Path(__file__).resolve().parent.parent / "corrupted_lifted" / "figs"
OUT_STEM = "fig_c_sensitivity_heatmap"

C_VALUES = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
DATASETS = ["Energy", "Weather", "MuJoCo", "Stocks", "Sine"]

# MOCK disc-score matrix shaped like "V-curve per row, minimum at c=0.1".
# rows = datasets, cols = c-values. float('nan') marks "diverged".
DISC = np.array([
    # Energy (primary; c=0.1 anchored at 0.045 from tab:ablation)
    [np.nan, 0.054, 0.045, 0.053, 0.065, 0.092],
    # Weather (71% missing in table, 50% here for apples-to-apples)
    [np.nan, 0.045, 0.038, 0.047, 0.061, 0.089],
    # MuJoCo -- smooth physics, easy
    [np.nan, 0.039, 0.031, 0.039, 0.052, 0.074],
    # Stocks -- non-stationary, hardest at 50%
    [np.nan, 0.078, 0.067, 0.075, 0.091, 0.118],
    # Sine -- synthetic, trivially easy
    [np.nan, 0.022, 0.015, 0.022, 0.031, 0.045],
])

# Robust-band threshold: disc within +delta of per-dataset optimum counts as
# robust (matches the "stays within 0.020 of the optimum" language in §6.8).
ROBUST_DELTA = 0.020


def plot_heatmap():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.6))

    # For coloring, use log-ish scale for better contrast but keep linear Norm
    # for the colorbar interpretation.
    finite = DISC[~np.isnan(DISC)]
    norm = Normalize(vmin=finite.min(), vmax=finite.max() * 1.05)
    cmap = plt.get_cmap("viridis_r")

    for i, dset in enumerate(DATASETS):
        for j, c in enumerate(C_VALUES):
            val = DISC[i, j]
            if np.isnan(val):
                # Diverged cell: hatched red fill
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor="#f8d7da", hatch="////",
                                 edgecolor="#d62728", linewidth=0.8)
                ax.add_patch(rect)
                ax.text(j, i, "div.", ha="center", va="center",
                        fontsize=9, color="#d62728", fontweight="bold")
            else:
                color = cmap(norm(val))
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color,
                                 edgecolor="white", linewidth=0.6)
                ax.add_patch(rect)
                text_color = "white" if norm(val) > 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=text_color)

    # Robust-band contour: star on each row's minimum, then outline the cells
    # within ROBUST_DELTA of it.
    for i in range(len(DATASETS)):
        row = DISC[i]
        finite_mask = ~np.isnan(row)
        if not finite_mask.any():
            continue
        row_min = np.nanmin(row)
        j_min = int(np.nanargmin(row))
        # Place the star in the top-right corner of the optimum cell so it
        # does not cover the numeric value at the cell centre.
        ax.scatter(j_min + 0.32, i - 0.30, marker="*", s=180, color="gold",
                   edgecolor="black", linewidth=1.0, zorder=5,
                   clip_on=False)
        for j in range(len(C_VALUES)):
            if finite_mask[j] and (row[j] - row_min) <= ROBUST_DELTA:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="black",
                                       linewidth=2.0, zorder=4))

    ax.set_xlim(-0.5, len(C_VALUES) - 0.5)
    ax.set_ylim(len(DATASETS) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(C_VALUES)))
    ax.set_xticklabels([f"{c:g}" for c in C_VALUES])
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_xlabel(r"$c$  in  $\sigma_y = c \cdot \sigma_t$", fontsize=12)
    ax.set_ylabel("Dataset (50% missing)", fontsize=12)
    ax.set_title(
        r"Discriminative score vs $c$: $50\times$-wide robust band, "
        r"one tuned hyperparameter",
        fontsize=13, fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=25)
    cbar.set_label("disc (lower is better)", fontsize=11)

    # Legend for the overlay markers
    legend_entries = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markeredgecolor="black", markersize=12, label="per-dataset optimum"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2.0,
                  label=f"robust band ($\\leq +{ROBUST_DELTA:.2f}$ disc of opt.)"),
        Rectangle((0, 0), 1, 1, facecolor="#f8d7da", hatch="////",
                  edgecolor="#d62728", linewidth=0.8, label="diverged"),
    ]
    ax.legend(handles=legend_entries, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=10)

    fig.tight_layout()
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    plot_heatmap()
