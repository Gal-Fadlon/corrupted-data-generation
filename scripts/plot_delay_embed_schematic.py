"""Delay-embedding + dual-space schematic (companion to fig:method).

Explains the lift and the operator mismatch C1/M1 visually for readers not
familiar with ImagenTime, in three panels:

  (left)   raw irregular time series: observed dots / red x for missing
  (middle) the delay-embedding image, with the sliding-window footprint of a
           single TS point highlighted across its d image pixels
  (right)  one row of each operator side by side -- A_img (double-counts d
           pixels) vs G = S_ts . T^{-1} (one row of the true composed
           operator) -- to make the ~28x dimension claim concrete

Uses the actual DelayEmbedder mapping from models.img_transformations so the
panel geometry matches the code.

Usage:
    python scripts/plot_delay_embed_schematic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_operator_mismatch import build_delay_matrices  # noqa: E402

OUT_DIR = ROOT / "corrupted_lifted" / "figs"
OUT_STEM = "fig_delay_embed_schematic"

# Geometry matching the Energy-dataset setup in §7 (seq_len=24, d~4): small
# enough to be legible but large enough that a single TS point's footprint
# spans several image pixels, which is the whole point of panel (b).
SEQ_LEN = 24
DELAY = 2
EMBEDDING = 10
SEED = 5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Real time series
    t = np.linspace(0, 1, SEQ_LEN)
    y = 0.8 * np.sin(2 * np.pi * 1.8 * t) + 0.3 * np.cos(2 * np.pi * 3 * t)
    y = y + 0.08 * rng.standard_normal(SEQ_LEN)

    # Missing pattern: randomly drop ~40% of points
    obs_mask = rng.random(SEQ_LEN) > 0.4
    obs_idx = np.where(obs_mask)[0]
    miss_idx = np.where(~obs_mask)[0]

    # Delay-embedding mapping: which image pixels receive signal[t]
    t_mat, t_inv_mat, side = build_delay_matrices(SEQ_LEN, DELAY, EMBEDDING, "cpu")

    # Build the delay-embedding image by lifting the full (imputed) signal;
    # for the schematic we use the real y even at missing positions so the
    # image has interpretable structure.  Real MMPS would see zeros there.
    img = (t_mat.numpy() @ y).reshape(side, side)

    # Pick one TS index (observed, for clarity) whose footprint we highlight.
    highlight_ts = int(obs_idx[len(obs_idx) // 2])
    # footprint_pixels = list of linear indices p in {0..side*side-1} where
    # T[p, highlight_ts] = 1
    mask_col = t_mat[:, highlight_ts].numpy()
    footprint_linear = np.where(mask_col > 0)[0]
    footprint_rc = [(p // side, p % side) for p in footprint_linear]

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.3], wspace=0.35)

    # -- Panel (a): raw irregular TS --
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(np.arange(SEQ_LEN), y, color="#cccccc", lw=1.0, zorder=1)
    ax_a.scatter(obs_idx, y[obs_idx], color="#2ca02c", s=50, zorder=3,
                 label=r"observed $y_{\mathrm{ts}}$")
    ax_a.scatter(miss_idx, y[miss_idx], color="#d62728",
                 marker="x", s=70, linewidths=2.0, zorder=3, label="missing")
    # Mark the highlighted time index
    ax_a.axvline(highlight_ts, color="#1f77b4", ls="--", lw=1.3, alpha=0.7)
    ax_a.annotate(f"time index $t = {highlight_ts}$",
                  xy=(highlight_ts, y[highlight_ts]),
                  xytext=(highlight_ts + 1.5, y[highlight_ts] + 0.7),
                  color="#1f77b4", fontsize=10,
                  arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))
    ax_a.set_title("(a) Raw irregular time series", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("time index")
    ax_a.set_ylabel(r"$x_{\mathrm{ts}}$")
    ax_a.set_xticks(np.arange(0, SEQ_LEN, 2))
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc="upper right", fontsize=9)

    # -- Panel (b): delay-embedding image with highlighted footprint --
    ax_b = fig.add_subplot(gs[0, 1])
    im = ax_b.imshow(img, cmap="viridis", aspect="equal", origin="upper")
    for (r, c) in footprint_rc:
        ax_b.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                 edgecolor="#ffd400", linewidth=2.5))
    ax_b.set_title(
        f"(b) Delay-embedding image $x_{{\\mathrm{{img}}}} = \\mathcal{{T}}(x_{{\\mathrm{{ts}}}})$\n"
        f"one TS point at $t={highlight_ts}$ hits "
        f"$d = {len(footprint_rc)}$ image pixels",
        fontsize=11, fontweight="bold",
    )
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)

    # -- Panel (c): one row of A_img vs one row of G side by side --
    # A_img row for ts-index highlight_ts: the d image pixels get a 1 each.
    # G row for observation k (pointing to ts = highlight_ts): the d image
    # pixels share 1/d weight each (T^{-1} averages).
    n_img = side * side
    row_Aimg = np.zeros(n_img)
    row_Aimg[footprint_linear] = 1.0
    row_G = np.zeros(n_img)
    row_G[footprint_linear] = 1.0 / len(footprint_linear)

    ax_c = fig.add_subplot(gs[0, 2])
    x_range = np.arange(n_img)
    width = 0.38
    ax_c.bar(x_range - width / 2, row_Aimg, width=width, color="#d62728",
             label=r"$A_{\mathrm{img}}$ row (naive MMPS)",
             edgecolor="black", linewidth=0.3)
    ax_c.bar(x_range + width / 2, row_G, width=width, color="#2ca02c",
             label=r"$G = S_{\mathrm{ts}} \cdot \mathcal{T}^{-1}$ row (ours)",
             edgecolor="black", linewidth=0.3)
    ax_c.axhline(0, color="black", lw=0.5)
    ax_c.set_xlim(-0.5, n_img - 0.5)
    ax_c.set_xlabel("image-pixel index $p$ (length $N_{\\mathrm{img}}$)")
    ax_c.set_ylabel("row weight")
    ax_c.set_title(
        r"(c) One row of $A_{\mathrm{img}}$ vs $G$"
        "\n"
        r"$A_{\mathrm{img}}$ over-counts $d$ pixels as independent"
        " ; $G$ averages by $1/d$",
        fontsize=11, fontweight="bold",
    )
    # Visually highlight non-zero entries' locations
    for p in footprint_linear:
        ax_c.axvspan(p - 0.45, p + 0.45, color="#ffd400", alpha=0.18, zorder=0)
    ax_c.legend(loc="upper right", fontsize=9)
    ax_c.grid(True, alpha=0.25, axis="y")

    # Sub-caption / dimension-claim note under panel (c)
    ax_c.text(
        0.02, -0.28,
        f"CG dimension: naive = $N_{{\\mathrm{{img}}}}$ = {n_img}   "
        f"vs   ours = $N_{{\\mathrm{{obs}}}}$ = {len(obs_idx)}    "
        f"$\\Rightarrow$ $\\sim {n_img // max(len(obs_idx), 1)}\\times$ reduction",
        transform=ax_c.transAxes, fontsize=10, color="#333333",
    )

    fig.suptitle(
        "Dual-space setting: why $A_{\\mathrm{img}} \\neq G = S_{\\mathrm{ts}} \\circ \\mathcal{T}^{-1}$",
        fontsize=14, fontweight="bold", y=1.02,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
