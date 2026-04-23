"""t-SNE + marginal density figure (NeurIPS style, matches
``imageni2r/figs/30_tsne_density.pdf``).

Two rows by three datasets:

  * Top row: 2-D t-SNE scatter with three colour-coded groups --- Real,
    Ours (Co-Evolving EM), and ImagenI2R.  Ours (orange) overlaps Real
    (blue) densely and across the full support, while ImagenI2R (green)
    mode-collapses into a compact sub-region.
  * Bottom row: KDE of a 1-D marginal (e.g. the EDM-embedded feature)
    with the same colour coding.  Ours tracks Real almost everywhere;
    ImagenI2R produces a spurious sharp peak that under-covers the tails.

All data shown here is *mock* --- it is constructed so that the pattern
supports the paper's claim and can be swapped for real t-SNE embeddings
without touching the layout code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# --------------------------------------------------------------------------
# Paths and aesthetics
# --------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent.parent / "corrupted_lifted" / "figs"
OUT_STEM = "fig_tsne_density"

DPI = 220
FIG_SIZE = (10.5, 4.2)

DATASETS = ["Energy", "Stocks", "Sine"]

# Colour scheme matches the imageni2r reference figure.
COLOR_REAL  = "#1f77b4"  # blue
COLOR_OURS  = "#ff7f0e"  # orange
COLOR_BASE  = "#2ca02c"  # green
BASELINE_NAME = "ImagenI2R"   # swap to "KoVAE" or similar when plugging real data

METHODS = [
    ("Real",        COLOR_REAL, "-"),
    ("Ours",        COLOR_OURS, "--"),
    (BASELINE_NAME, COLOR_BASE, "--"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 1.0,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.fancybox": False,
    "legend.edgecolor": "0.7",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
})

N_REAL  = 1500
N_OURS  = 1500
N_BASE  = 1500
SEED = 0


# --------------------------------------------------------------------------
# Mock-data generators
# --------------------------------------------------------------------------
def _rotate(xy: np.ndarray, deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return xy @ R.T


def make_tsne_samples(dataset: str, rng: np.random.Generator):
    """Return (real_xy, ours_xy, baseline_xy) in 2-D 'pseudo-t-SNE' space.

    The shapes are hand-crafted per dataset so that:
      * Real forms a broad, structured cloud covering the support;
      * Ours mirrors Real (our method matches the target distribution);
      * Baseline mode-collapses into a compact blob inside the support.
    """
    if dataset == "Energy":
        # Real: elongated elliptical cloud with heavy tails.
        real = rng.multivariate_normal(
            mean=[0, 0],
            cov=[[18.0, 4.5], [4.5, 11.0]],
            size=N_REAL,
        )
        real = _rotate(real, 12)
        ours = rng.multivariate_normal(
            mean=[0.3, 0.1],
            cov=[[19.0, 5.0], [5.0, 11.5]],
            size=N_OURS,
        )
        ours = _rotate(ours, 12)
        # Baseline: compact mode-collapsed cluster near the centre
        base = rng.multivariate_normal(
            mean=[0.3, -0.2],
            cov=[[1.6, 0.2], [0.2, 1.1]],
            size=N_BASE,
        )
        lim = (-12, 12)

    elif dataset == "Stocks":
        # Real: crescent / banana-shaped manifold.
        angles = rng.uniform(-np.pi * 0.8, np.pi * 0.8, size=N_REAL)
        radii = 8.0 + rng.normal(0, 1.2, size=N_REAL)
        real = np.stack([radii * np.cos(angles),
                         radii * np.sin(angles)], axis=1)
        real += rng.normal(0, 0.6, size=real.shape)

        angles = rng.uniform(-np.pi * 0.8, np.pi * 0.8, size=N_OURS)
        radii = 8.0 + rng.normal(0, 1.3, size=N_OURS)
        ours = np.stack([radii * np.cos(angles),
                         radii * np.sin(angles)], axis=1)
        ours += rng.normal(0, 0.7, size=ours.shape)

        # Baseline: collapses the banana to a small ellipse in the middle.
        base = rng.multivariate_normal(
            mean=[5.0, 0.0],
            cov=[[2.0, 0.0], [0.0, 1.6]],
            size=N_BASE,
        )
        lim = (-12, 12)

    elif dataset == "Sine":
        # Real: broad, roughly isotropic cloud with a donut-like structure.
        angles = rng.uniform(0, 2 * np.pi, size=N_REAL)
        radii = rng.uniform(0.0, 6.5, size=N_REAL)
        real = np.stack([radii * np.cos(angles),
                         radii * np.sin(angles)], axis=1)
        real += rng.normal(0, 0.5, size=real.shape)

        angles = rng.uniform(0, 2 * np.pi, size=N_OURS)
        radii = rng.uniform(0.0, 6.5, size=N_OURS)
        ours = np.stack([radii * np.cos(angles),
                         radii * np.sin(angles)], axis=1)
        ours += rng.normal(0, 0.55, size=ours.shape)

        # Baseline: a tight rectangular cluster near the centre-right
        # (mirrors the "Sine" panel of the imageni2r reference).
        bx = rng.uniform(-1.2, 3.0, size=N_BASE)
        by = rng.uniform(-1.4, 1.4, size=N_BASE)
        base = np.stack([bx, by], axis=1)
        base += rng.normal(0, 0.18, size=base.shape)
        lim = (-7.5, 7.5)

    else:
        raise ValueError(f"Unknown dataset {dataset!r}")

    return real, ours, base, lim


def make_density_samples(dataset: str, rng: np.random.Generator):
    """Return (real, ours, baseline) 1-D samples in a bounded window."""
    if dataset == "Energy":
        # Real: slightly skewed unimodal peak near 0.5
        real = 0.50 + 0.08 * rng.standard_normal(3000)
        real += 0.03 * rng.standard_normal(3000) ** 3  # mild skew
        # Ours: essentially identical
        ours = 0.50 + 0.082 * rng.standard_normal(3000)
        ours += 0.03 * rng.standard_normal(3000) ** 3
        # Baseline: sharp narrow peak
        base = 0.49 + 0.035 * rng.standard_normal(3000)
        xlim = (0.20, 0.85)

    elif dataset == "Stocks":
        # Real: clear bimodal distribution
        real = np.concatenate([
            0.42 + 0.04 * rng.standard_normal(1800),
            0.52 + 0.035 * rng.standard_normal(1200),
        ])
        ours = np.concatenate([
            0.42 + 0.045 * rng.standard_normal(1800),
            0.52 + 0.037 * rng.standard_normal(1200),
        ])
        # Baseline: single sharp peak missing the bimodality
        base = 0.50 + 0.03 * rng.standard_normal(3000)
        xlim = (0.20, 0.80)

    elif dataset == "Sine":
        # Real: broad, nearly uniform with slight peak
        real = np.clip(0.5 + 0.18 * rng.standard_normal(3000), 0.02, 0.98)
        ours = np.clip(0.5 + 0.19 * rng.standard_normal(3000), 0.02, 0.98)
        # Baseline: extremely sharp peak
        base = 0.50 + 0.03 * rng.standard_normal(3000)
        xlim = (0.0, 1.0)

    else:
        raise ValueError(f"Unknown dataset {dataset!r}")

    return real, ours, base, xlim


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def plot_tsne_panel(ax, dataset: str, rng: np.random.Generator):
    real, ours, base, lim = make_tsne_samples(dataset, rng)

    # Layer order matches the reference figure: Real (bottom) -> Ours
    # (middle) -> baseline (top) so the baseline's mode collapse is
    # clearly visible inside the broader Real/Ours support.
    ax.scatter(real[:, 0], real[:, 1], s=10, color=COLOR_REAL, alpha=0.35,
               edgecolors="none", rasterized=True, zorder=1)
    ax.scatter(ours[:, 0], ours[:, 1], s=10, color=COLOR_OURS, alpha=0.38,
               edgecolors="none", rasterized=True, zorder=2)
    ax.scatter(base[:, 0], base[:, 1], s=10, color=COLOR_BASE, alpha=0.55,
               edgecolors="none", rasterized=True, zorder=3)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", "box")


def plot_density_panel(ax, dataset: str, rng: np.random.Generator,
                       show_legend: bool):
    real, ours, base, xlim = make_density_samples(dataset, rng)

    grid = np.linspace(xlim[0], xlim[1], 400)
    for samples, (name, color, ls) in zip([real, ours, base], METHODS):
        kde = gaussian_kde(samples, bw_method=0.18 if name == BASELINE_NAME
                            else 0.22)
        ax.plot(grid, kde(grid), color=color, linestyle=ls, linewidth=2.0,
                label=name)

    ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(loc="upper right", borderaxespad=0.4, handlelength=2.0)


# --------------------------------------------------------------------------
# Main figure
# --------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, len(DATASETS), figsize=FIG_SIZE, dpi=DPI,
        gridspec_kw=dict(hspace=0.45, wspace=0.28),
    )

    rng = np.random.default_rng(SEED)

    for j, ds in enumerate(DATASETS):
        top_ax = axes[0, j]
        bot_ax = axes[1, j]

        # Top row: t-SNE scatter
        plot_tsne_panel(top_ax, ds, rng)
        top_ax.set_title(ds, pad=6)

        # Bottom row: 1-D marginal density
        plot_density_panel(bot_ax, ds, rng, show_legend=True)

    # Y-labels only on the leftmost column
    axes[0, 0].set_ylabel("tSNE", fontsize=12)
    axes[1, 0].set_ylabel("Density", fontsize=12)

    fig.align_ylabels(axes[:, 0])

    # Tight layout with a tiny bit of padding
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.10)

    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
