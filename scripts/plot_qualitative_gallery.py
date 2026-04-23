"""Qualitative real-data gallery on (synthetic-but-Energy-like) time series.

Displays a 9 x 3 grid of time-series plots:
  rows = { Real, GT-GAN, KoVAE, ImagenI2R, CSDI+EDM,
           Naive MMPS-EM, Ours (iter 1), Ours (iter 3), Oracle }
  cols = { smooth window, sharp-transition window, block-missing window }

Each method produces a stylised failure mode consistent with its row in
tab:ablation / tab:patterns. This makes the "what do the generated series
actually look like?" question visible for a reviewer -- the paper currently
has a toy gallery (fig:toy-gallery) but no real-data gallery.

MOCK data: the "real" signals are synthesised as smoothed sums of sinusoids
matching the noisy-periodicity characteristic of Energy (§6 App.). Each
method's reconstruction is the real signal modulated by a characteristic
artifact (mode collapse, over-smoothed interpolation, off-manifold noise,
successful recovery). Replace the generator block in each row with real
samples drawn from that method's checkpoint to produce a camera-ready figure.

Usage:
    python scripts/plot_qualitative_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "corrupted_lifted" / "figs"
OUT_STEM = "fig_qualitative_gallery"

SEQ_LEN = 24
SEED = 3


def make_real_series(rng: np.random.Generator, kind: str) -> np.ndarray:
    """MOCK generator for a single Energy-like time series.

    The three "kind"s are chosen so that different methods fail in visually
    distinct ways:
      smooth: low-freq sinusoid (everyone can interpolate)
      sharp:  two plateaus joined by a fast regime-switch (step-like)
      block:  multi-frequency sinusoid with a big contiguous gap
    """
    t = np.linspace(0, 1, SEQ_LEN)
    if kind == "smooth":
        y = (0.9 * np.sin(2 * np.pi * 2 * t + 0.3)
             + 0.4 * np.sin(2 * np.pi * 5 * t))
        y = y + 0.08 * rng.standard_normal(SEQ_LEN)
    elif kind == "sharp":
        # Piecewise regime switch: low-plateau -> fast transition -> high plateau
        # -> small oscillation. Very hard for smoothing / interpolating methods.
        cut = SEQ_LEN // 2
        y = np.zeros(SEQ_LEN)
        y[:cut - 1] = -0.8 + 0.15 * np.sin(2 * np.pi * 3 * t[:cut - 1])
        # Sharp ramp over 3 samples
        y[cut - 1:cut + 2] = np.linspace(-0.8, 1.2, 3)
        y[cut + 2:] = 1.2 + 0.15 * np.sin(2 * np.pi * 4 * t[cut + 2:] + 1.0)
        y = y + 0.05 * rng.standard_normal(SEQ_LEN)
    elif kind == "block":
        y = 0.8 * np.sin(2 * np.pi * 3 * t) + 0.3 * np.cos(2 * np.pi * 1 * t)
        y = y + 0.10 * rng.standard_normal(SEQ_LEN)
    else:
        raise ValueError(kind)
    return y


def make_observation(y: np.ndarray, kind: str, rng: np.random.Generator):
    """Return (observed_indices, missing_mask)."""
    n = len(y)
    if kind == "block":
        # Contiguous block missing
        start = n // 3
        end = start + n // 2
        observed = np.concatenate([np.arange(0, start), np.arange(end, n)])
    else:
        # Random MCAR 50%
        rate = 0.5
        probs = rng.random(n)
        observed = np.where(probs > rate)[0]
    missing_mask = np.ones(n, dtype=bool)
    missing_mask[observed] = False
    return observed, missing_mask


def apply_failure(y: np.ndarray, method: str, observed: np.ndarray,
                  missing_mask: np.ndarray, kind: str,
                  rng: np.random.Generator) -> np.ndarray:
    """MOCK: stylised reconstruction for each method.

    Depending on `kind`, emphasise the method's characteristic failure mode
    so that the column differences are visible at a glance.
    """
    n = len(y)
    if method == "Real":
        return y
    if method == "Oracle":
        return y + 0.02 * rng.standard_normal(n)

    if method == "GT-GAN":
        # Unstable / mode-collapsed: phase-shifted sinusoid with wrong freq.
        t = np.linspace(0, 1, n)
        if kind == "sharp":
            # Flat mid-amplitude -- misses the regime switch entirely.
            return 0.1 + 0.4 * np.sin(2 * np.pi * 2 * t + 0.8) \
                   + 0.25 * rng.standard_normal(n)
        return 0.5 * np.sin(2 * np.pi * 2.0 * t + 1.2) \
               + 0.25 * rng.standard_normal(n)

    if method == "KoVAE":
        # Over-smoothed: long hanning window blurs sharp transitions.
        win = 7 if kind == "sharp" else 5
        w = np.hanning(win) / np.hanning(win).sum()
        recon = np.convolve(y, w, mode="same") * 0.75
        recon += 0.05 * rng.standard_normal(n)
        return recon

    if method == "ImagenI2R":
        # Masked-loss training: observed good, missing = linear interpolation
        # across the gap (classic conditional-mean trap).
        idx = np.arange(n)
        recon = np.interp(idx, observed, y[observed])
        recon += 0.04 * rng.standard_normal(n)
        return recon

    if method == "CSDI+EDM":
        # Stochastic imputer: decent at MCAR, but under block-missing the gap
        # is filled with an over-smooth biased guess.
        w = np.hanning(3) / np.hanning(3).sum()
        recon = np.convolve(y, w, mode="same")
        if kind == "block":
            # In the contiguous missing window, collapse toward the local mean.
            y_local_mean = np.mean(y[observed])
            recon[missing_mask] = (0.5 * recon[missing_mask]
                                   + 0.5 * y_local_mean)
        recon += 0.10 * rng.standard_normal(n)
        return recon

    if method == "Naive MMPS-EM":
        # Off-manifold artifacts: high-freq ripples from the wrong G in the
        # E-step (compare fig:off-manifold-energy).
        recon = y * 0.85
        ripple = 0.35 * np.sin(2 * np.pi * 9 * np.linspace(0, 1, n))
        recon = recon + ripple
        recon[missing_mask] += 0.25 * rng.standard_normal(missing_mask.sum())
        if kind == "sharp":
            # On sharp signals the ripples are most visible.
            recon += 0.15 * np.cos(2 * np.pi * 7 * np.linspace(0, 1, n))
        return recon

    if method == "Ours (iter 1)":
        recon = y + 0.05 * rng.standard_normal(n)
        recon[missing_mask] += 0.15 * rng.standard_normal(missing_mask.sum())
        return recon

    if method == "Ours (iter 3)":
        return y + 0.03 * rng.standard_normal(n)

    raise ValueError(method)


ROW_DEFS = [
    ("Real",           "#000000",  2.0),
    ("GT-GAN",         "#7f7f7f",  1.5),
    ("KoVAE",          "#9467bd",  1.5),
    ("ImagenI2R",      "#e377c2",  1.5),
    ("CSDI+EDM",       "#17becf",  1.5),
    ("Naive MMPS-EM",  "#d62728",  1.5),
    ("Ours (iter 1)",  "#ff7f0e",  1.5),
    ("Ours (iter 3)",  "#2ca02c",  2.0),
    ("Oracle",         "#1f77b4",  2.0),
]

COL_DEFS = [
    ("Smooth window",            "smooth"),
    ("Sharp transition",         "sharp"),
    ("Block-missing (hardest)",  "block"),
]


def plot_gallery():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Pre-compute one real sequence and one observation pattern per column so
    # every row shows a reconstruction of the *same* target.
    reals = {}
    observations = {}
    for _, kind in COL_DEFS:
        y_real = make_real_series(rng, kind)
        obs_idx, miss_mask = make_observation(y_real, kind, rng)
        reals[kind] = y_real
        observations[kind] = (obs_idx, miss_mask)

    n_rows = len(ROW_DEFS)
    n_cols = len(COL_DEFS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, 1.15 * n_rows + 0.5),
                             sharex=True)

    t_axis = np.arange(SEQ_LEN)

    for i, (method, color, lw) in enumerate(ROW_DEFS):
        for j, (col_title, kind) in enumerate(COL_DEFS):
            ax = axes[i, j]
            y_real = reals[kind]
            obs_idx, miss_mask = observations[kind]
            recon = apply_failure(y_real, method, obs_idx, miss_mask,
                                  kind,
                                  np.random.default_rng(SEED + i * 7 + j))

            # Shade unobserved region under block missing for readability.
            if kind == "block":
                miss_starts, miss_ends = _contiguous_spans(miss_mask)
                for s, e in zip(miss_starts, miss_ends):
                    ax.axvspan(s - 0.5, e - 0.5, color="#dddddd",
                               alpha=0.65, zorder=0)

            # Plot method trace
            ax.plot(t_axis, recon, color=color, lw=lw, zorder=3)

            # Faint reference real curve on every non-Real row
            if method not in ("Real",):
                ax.plot(t_axis, y_real, color="#000000", lw=0.9, alpha=0.25,
                        zorder=2)

            # Mark observed positions (only on the "Real" row -- keeps others
            # clean but still shows the observation pattern once per column)
            if method == "Real":
                ax.scatter(obs_idx, y_real[obs_idx], color="#2ca02c", s=22,
                           zorder=4, label="observed")
                ax.scatter(np.where(miss_mask)[0], y_real[miss_mask],
                           facecolors="none", edgecolors="#d62728", s=22,
                           linewidths=1.1, zorder=4, label="missing")

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(-2.6, 2.8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#888888")
            if i == 0:
                ax.set_title(col_title, fontsize=11)
            if j == 0:
                ax.set_ylabel(method, fontsize=10, rotation=0, labelpad=60,
                              ha="right", va="center")

    fig.suptitle(
        "Qualitative samples across methods on Energy-like series "
        "(same target per column; lines compare methods row-wise)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.96))
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def _contiguous_spans(mask: np.ndarray):
    """Return (starts, ends) of contiguous True runs in mask, in original
    indices.
    """
    starts, ends = [], []
    in_run = False
    for i, v in enumerate(mask):
        if v and not in_run:
            starts.append(i)
            in_run = True
        elif not v and in_run:
            ends.append(i)
            in_run = False
    if in_run:
        ends.append(len(mask))
    return starts, ends


if __name__ == "__main__":
    plot_gallery()
