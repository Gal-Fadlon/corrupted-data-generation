"""Off-manifold energy E_off(k) vs EM iteration, log-scale.

Visualises the central structural result from tab:ablation / §5.4 / App.~F.4:
partial corrections diverge because without manifold projection there is no
mechanism to prevent off-manifold drift from compounding across iterations.

Plots log-scale trajectories for the five ablation configurations:
  Row 1 -- Naive MMPS-EM (no corrections, stable but biased)
  Row 2 -- +STL init only (diverges)
  Row 3 -- +STL init +adaptive sigma_y (diverges faster)
  Row 4 -- +STL init +sigma_y +manifold projection (stable)
  Row 5 -- Ours full (stable near numerical zero)

MOCK values are tagged below; they are consistent with the paper's claims
("~76 for diverging runs, ~1e-13 for stable" in tab:c-sensitivity context and
"loss reaches billions / off-manifold energy reaches millions" in §5.2).

Usage:
    python scripts/plot_off_manifold_energy.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "corrupted_lifted" / "figs"
OUT_STEM = "fig_off_manifold_energy"

# MOCK values -- replace with E_off(k) logged from real runs.
# Iteration 0 represents the warm-start image (before any E-step).
EM_ITERS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Row 1: Naive MMPS-EM -- off-manifold from the start (wrong operator produces
# off-manifold corrections) but bounded: the naive system converges to a
# biased fixed point (regime c) at non-trivial E_off, not divergent.
naive_mmps = np.array([1.2e-1, 1.3e-1, 1.4e-1, 1.5e-1, 1.5e-1,
                       1.6e-1, 1.6e-1, 1.6e-1, 1.7e-1, 1.7e-1])

# Row 2: +STL init only -- structurally-good cold start, but without adaptive
# sigma_y or manifold projection the CG amplifies mismatch (regime b).
init_only = np.array([8.0e-3, 4.0e-2, 5.0e-1, 8.0,   1.2e2,
                      2.0e3,  3.5e4,  6.0e5,  8.0e6, 2.0e7])

# Row 3: +STL init +adaptive sigma_y -- CG now stable, but without projection
# off-manifold drift still explodes.
init_sigma = np.array([8.0e-3, 1.2e-2, 4.0e-2, 3.0,   1.5e2,
                       7.0e3,  1.8e5,  2.0e6,  3.0e6, 4.0e6])

# Row 4: +STL init +sigma_y +manifold projection -- the projection drives
# off-manifold energy down toward numerical zero.
init_sigma_proj = np.array([8.0e-3, 3.0e-5, 1.0e-7, 5.0e-10, 2.0e-12,
                            6.0e-13, 4.0e-13, 3.0e-13, 3.0e-13, 3.0e-13])

# Row 5: Ours full (+obs-CG) -- converges slightly faster to the same
# numerical-zero floor.
ours_full = np.array([8.0e-3, 1.0e-5, 1.0e-8, 2.0e-11, 5.0e-13,
                      4.0e-13, 3.0e-13, 3.0e-13, 3.0e-13, 3.0e-13])

METHODS = [
    ("Row 1: Naive MMPS-EM (regime c, biased)",       naive_mmps,      "#d62728", "o", "-"),
    ("Row 2: +STL init (regime b, diverges)",         init_only,       "#ff7f0e", "s", "-"),
    ("Row 3: +init +adaptive $\\sigma_y$ (diverges)", init_sigma,      "#bcbd22", "v", "-"),
    ("Row 4: +init +$\\sigma_y$ +manifold proj.",     init_sigma_proj, "#1f77b4", "^", "-"),
    ("Row 5: Ours full (obs-CG + all)",               ours_full,       "#2ca02c", "D", "-"),
]

# Divergence threshold (mock): above this the M-step loss overflows and the
# iteration is marked "diverged" in tab:ablation.
DIVERGENCE_THRESHOLD = 1e3


def plot_off_manifold():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhspan(DIVERGENCE_THRESHOLD, 1e9, alpha=0.08, color="red",
               label="diverged region")

    for label, vals, color, marker, ls in METHODS:
        ax.plot(EM_ITERS, vals, color=color, marker=marker, linestyle=ls,
                markersize=7, lw=2.0, label=label, zorder=3)

    ax.axhline(DIVERGENCE_THRESHOLD, color="red", ls="--", lw=1.2, alpha=0.6)
    ax.text(EM_ITERS[-1] + 0.15, DIVERGENCE_THRESHOLD * 1.3,
            r"divergence threshold", color="red", fontsize=9, va="bottom",
            ha="right")

    ax.set_yscale("log")
    ax.set_xlabel("EM iteration $k$", fontsize=12)
    ax.set_ylabel(r"Off-manifold energy $E_{\mathrm{off}}(k) = "
                  r"\mathrm{mean}_i \|x_i - \Pi(x_i)\|^2$", fontsize=12)
    ax.set_title(
        "Off-manifold energy compounds without projection (Energy 50%)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(-0.2, EM_ITERS[-1] + 0.5)
    ax.set_ylim(1e-14, 1e8)
    ax.grid(True, which="both", alpha=0.25, ls=":")
    ax.legend(loc="center right", fontsize=10, framealpha=0.92)

    fig.tight_layout()
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    plot_off_manifold()
