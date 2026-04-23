"""Posterior diversity visualisation: 20 E-step samples per sampler.

Given a single fixed irregular observation y_ts, draw 20 independent E-step
samples under each of the five posterior samplers surveyed in
tab:samplers-motivation and overlay them on one subplot per method. The panels
make the C2 claim (why full Tweedie covariance is required) concrete:

  DPS          -> 20 identical lines (point-estimate collapse, disc 0.500)
  PiGDM        -> diagonal-Gaussian spread (high-freq noise everywhere)
  TMPD (naive) -> structured but miscentered (row-sum diagonal, single-space)
  MMPS (naive) -> diverse but off-manifold (wrong G = A_img)
  L-MMPS (ours)-> diverse, on-manifold, respecting observations
  Ground truth -> reference signal + observation markers

MOCK data: samples are hand-crafted to match each sampler's documented failure
mode so the visual story lines up with the tables in the paper. Replace each
method block with 20 real E-step draws to get the camera-ready version.

Usage:
    python scripts/plot_posterior_diversity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "corrupted_lifted" / "figs"
OUT_STEM = "fig_posterior_diversity"

SEQ_LEN = 48
N_SAMPLES = 20
SEED = 11


def make_ground_truth(rng: np.random.Generator):
    t = np.linspace(0, 1, SEQ_LEN)
    y_true = (0.9 * np.sin(2 * np.pi * 2 * t)
              + 0.4 * np.sin(2 * np.pi * 5 * t + 0.7)
              + 0.2 * np.cos(2 * np.pi * 1 * t))
    mask = rng.random(SEQ_LEN) > 0.5
    obs_idx = np.where(mask)[0]
    miss_idx = np.where(~mask)[0]
    return t, y_true, obs_idx, miss_idx


def samples_dps(y_true, obs_idx, rng, n=N_SAMPLES):
    """Point-estimate: all 20 samples are the same mean-estimate curve with
    tiny numerical jitter."""
    base = _mean_estimate(y_true, obs_idx)
    return np.stack([base + 1e-3 * rng.standard_normal(SEQ_LEN)
                     for _ in range(n)], axis=0)


def samples_pigdm(y_true, obs_idx, rng, n=N_SAMPLES):
    """Diagonal Gaussian: spread across all dimensions, misses block
    correlation structure. High-frequency noise everywhere."""
    out = np.empty((n, SEQ_LEN))
    for i in range(n):
        noise = 0.35 * rng.standard_normal(SEQ_LEN)
        out[i] = y_true + noise
        out[i, obs_idx] = y_true[obs_idx]  # still hits observations
    return out


def samples_tmpd(y_true, obs_idx, rng, n=N_SAMPLES):
    """Row-sum diagonal: structured but mis-centered; all samples share a
    systematic bias away from the truth."""
    bias = 0.4 * np.sin(2 * np.pi * 1.3 * np.linspace(0, 1, SEQ_LEN) + 0.5)
    out = np.empty((n, SEQ_LEN))
    for i in range(n):
        noise = 0.12 * rng.standard_normal(SEQ_LEN)
        out[i] = y_true + bias + noise
        out[i, obs_idx] = y_true[obs_idx]
    return out


def samples_mmps_naive(y_true, obs_idx, rng, n=N_SAMPLES):
    """Full Jacobian, wrong operator: diverse, correct covariance shape, but
    off-manifold -- visible as high-frequency ripples beyond observations."""
    ripple_freqs = rng.uniform(7.0, 12.0, size=n)
    out = np.empty((n, SEQ_LEN))
    t = np.linspace(0, 1, SEQ_LEN)
    for i in range(n):
        ripple = 0.18 * np.sin(2 * np.pi * ripple_freqs[i] * t + rng.random())
        noise = 0.08 * rng.standard_normal(SEQ_LEN)
        out[i] = y_true + ripple + noise
        out[i, obs_idx] = y_true[obs_idx]
    return out


def samples_lmmps(y_true, obs_idx, rng, n=N_SAMPLES):
    """Ours: diverse, on-manifold (low-frequency smooth variation around
    truth), respecting observations exactly."""
    out = np.empty((n, SEQ_LEN))
    t = np.linspace(0, 1, SEQ_LEN)
    for i in range(n):
        low_freq = 0.12 * np.sin(2 * np.pi * 1.0 * t + rng.uniform(0, 2 * np.pi))
        low_freq += 0.08 * np.cos(2 * np.pi * 2.0 * t + rng.uniform(0, 2 * np.pi))
        noise = 0.03 * rng.standard_normal(SEQ_LEN)
        out[i] = y_true + low_freq + noise
        out[i, obs_idx] = y_true[obs_idx]  # hard observation enforcement
    return out


def _mean_estimate(y_true, obs_idx):
    """Linear-interpolation point estimate between observed values."""
    idx = np.arange(SEQ_LEN)
    return np.interp(idx, obs_idx, y_true[obs_idx])


PANELS = [
    ("DPS (point estimate)\n  disc $\\approx 0.500$",                 samples_dps,         "#d62728"),
    ("PiGDM (diagonal Gaussian)\n  disc $\\approx 0.462$",            samples_pigdm,       "#ff7f0e"),
    ("TMPD naive (row-sum diag, wrong $G$)",                          samples_tmpd,        "#bcbd22"),
    ("MMPS naive (full Jac., wrong $G$)\n  disc $\\approx 0.092$",    samples_mmps_naive,  "#9467bd"),
    ("L-MMPS (ours, PSLR on MMPS)\n  disc $\\approx 0.045$", samples_lmmps,   "#2ca02c"),
]

def plot_posterior_diversity():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t, y_true, obs_idx, miss_idx = make_ground_truth(rng)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (title, sampler_fn, color) in zip(axes[:5], PANELS):
        samples = sampler_fn(y_true, obs_idx, np.random.default_rng(rng.integers(1e6)))
        for s in samples:
            ax.plot(t, s, color=color, lw=0.9, alpha=0.35, zorder=2)
        # Posterior mean highlighted
        mean = samples.mean(axis=0)
        ax.plot(t, mean, color=color, lw=2.0, zorder=3, label="posterior mean")
        # Ground-truth reference
        ax.plot(t, y_true, color="black", lw=1.4, ls="--", alpha=0.7,
                zorder=4, label="ground truth")
        # Observations
        ax.scatter(t[obs_idx], y_true[obs_idx], color="black", s=18,
                   zorder=5, label="observed $y_{\\mathrm{ts}}$")

        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)

    # Ground-truth panel (last)
    ax = axes[5]
    ax.plot(t, y_true, color="black", lw=2.2, label="$x_{\\mathrm{ts}}$ (truth)")
    ax.scatter(t[obs_idx], y_true[obs_idx], color="#2ca02c", s=24,
               zorder=5, label="observed")
    ax.scatter(t[miss_idx], y_true[miss_idx], facecolors="none",
               edgecolors="#d62728", s=24, linewidths=1.2,
               zorder=5, label="missing")
    ax.set_title("Reference: truth + observation pattern", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Posterior diversity of E-step samplers on one fixed $y_{\\mathrm{ts}}$ "
        "(20 samples per method)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    plot_posterior_diversity()
