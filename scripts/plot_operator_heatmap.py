"""Operator mismatch heatmap: G^T G vs tildeG^T tildeG vs Delta_G for binary vs
continuous observation operators.

Re-uses the linear-algebra primitives from
[scripts/compute_operator_mismatch.py](scripts/compute_operator_mismatch.py)
(build_delay_matrices, sample_binary_mask_S, sample_continuous_S,
build_Aimg_from_S_support) so the heatmaps are drawn from the same matrices
that produce the numbers in tab:continuous-mismatch-norm.

Produces a 2 x 3 heatmap grid:

    | G^T G (binary)      | tildeG^T tildeG (binary)      | Delta_G (binary)      |
    | G^T G (continuous)  | tildeG^T tildeG (continuous)  | Delta_G (continuous)  |

Shared colorbars per column so the continuous-row off-diagonals are visually
obvious against the binary row.

Usage:
    python scripts/plot_operator_heatmap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_operator_mismatch import (  # noqa: E402
    build_Aimg_from_S_support,
    build_delay_matrices,
    sample_binary_mask_S,
    sample_continuous_S,
)

OUT_DIR = ROOT / "corrupted_lifted" / "figs"
OUT_STEM = "fig_operator_mismatch_heatmap"

# Problem geometry: same as compute_operator_mismatch.py defaults (Energy
# seq_len=24, delay=3, embedding=8 -> n_img = 64 after square padding).
SEQ_LEN = 24
DELAY = 3
EMBEDDING = 8
RHO = 0.5
SEED = 0
DEVICE = "cpu"


def compute_grams():
    rng = np.random.default_rng(SEED)
    t_mat, t_inv_mat, side = build_delay_matrices(SEQ_LEN, DELAY, EMBEDDING, DEVICE)
    n_obs = int(round(RHO * SEQ_LEN))

    # One deterministic draw per operator -- we want the heatmap to show a
    # single representative mismatch pattern, not averaged structure.
    S_bin = sample_binary_mask_S(SEQ_LEN, n_obs, rng, DEVICE)
    rng_cont = np.random.default_rng(SEED + 1)
    S_cont = sample_continuous_S(SEQ_LEN, n_obs, rng_cont, DEVICE)

    results = {}
    for label, S in [("binary", S_bin), ("continuous", S_cont)]:
        G = S @ t_inv_mat                 # [n_obs, n_img]
        A_img = build_Aimg_from_S_support(S, t_mat)  # [n_img, n_img]
        gram_true = (G.T @ G).cpu().numpy()
        gram_naive = (A_img.T @ A_img).cpu().numpy()
        delta = gram_naive - gram_true
        results[label] = {
            "gram_true": gram_true,
            "gram_naive": gram_naive,
            "delta": delta,
        }
    return results, side


def plot_heatmap_grid():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, side = compute_grams()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    col_titles = [
        r"$G^\top G$  (true composed operator)",
        r"$\tilde G^\top \tilde G$  (naive $A_{\mathrm{img}}$)",
        r"$\Delta_G = \tilde G^\top \tilde G - G^\top G$",
    ]
    row_labels = [r"Binary MCAR  ($S_{\mathrm{ts}} \in \{0,1\}^{N_{\mathrm{obs}}\times N_{\mathrm{ts}}}$)",
                  r"Continuous 2-tap  ($S_{\mathrm{ts}}$ fractional)"]

    # Per-column vmin/vmax for fair shared colorbars; Delta column uses a
    # symmetric diverging scale.
    for col, key in enumerate(["gram_true", "gram_naive", "delta"]):
        if key == "delta":
            vmax = max(abs(results["binary"][key]).max(),
                       abs(results["continuous"][key]).max())
            vmin = -vmax
            cmap = "RdBu_r"
        else:
            vmax = max(results["binary"][key].max(),
                       results["continuous"][key].max())
            vmin = 0.0
            cmap = "viridis"

        for row, label in enumerate(["binary", "continuous"]):
            mat = results[label][key]
            ax = axes[row, col]
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            # Show outer frame
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11, rotation=90, labelpad=10)

        # Shared colorbar per column, placed below the bottom row.
        cbar = fig.colorbar(im, ax=axes[:, col], location="bottom", pad=0.03,
                            fraction=0.04, aspect=35)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Operator mismatch on Energy seq_len=24 delay-embedding geometry "
        r"($N_{\mathrm{img}} = %d$, $\rho = %.1f$)" % (side * side, RHO),
        fontsize=14,
        fontweight="bold",
    )

    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    plot_heatmap_grid()
