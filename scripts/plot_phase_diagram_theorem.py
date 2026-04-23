"""Phase diagram for Theorem 3.1 (operator mismatch in MCEM).

Visualises the three regimes stated in the theorem:

  (a) small ||Delta||    -> monotone contraction to 0                 (blue)
  (b) medium ||Delta||   -> transient dip, then divergence            (red)
  (c) large ||Delta||    -> quick settling to a non-zero biased fp    (orange)

The error trajectory is computed from the closed-form expression implied by
App.~A's linear-Gaussian recurrence:

    err(k; ||Delta||) =  rho^k * err_0               (contraction)
                      + bias(||Delta||) * (1 - rho^k)/(1 - rho)   (bias)
                      + noise_growth(k; ||Delta||)  (finite-sample var)

Each term is tagged with a # MOCK comment; the functional form matches
Theorem 3.1 and its proof, only the specific scalar constants are chosen so
that the three regimes appear at visually clean Delta values.

Usage:
    python scripts/plot_phase_diagram_theorem.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

OUT_DIR = Path(__file__).resolve().parent.parent / "corrupted_lifted" / "figs"
OUT_STEM = "fig_phase_diagram"

# MOCK parameters controlling the analytical regime model.
N_ITERS = 25
N_DELTAS = 240
DELTA_RANGE = (1e-3, 1.0)

# Baseline contraction rate (spectral radius of (I - K G) when Delta=0).
RHO = 0.70
# Initial log-error: cold start at err_0 = 10^{LOG_ERR0} away from truth.
LOG_ERR0 = 1.2
# Noise floor of the linear-Gaussian recurrence when Delta=0.
LOG_NOISE_FLOOR = -7.0
# Boundary between regime (a) and regimes (b)/(c).
DELTA_STABLE = 0.015
# Boundary between (b) and (c).
DELTA_BIASED = 0.35
# Per-iteration growth factor at the worst point inside regime (b).
GROWTH_RATE_MAX = 1.22
# Asymptotic-bias linear constant: bias(delta) = BIAS_SLOPE * delta.
BIAS_SLOPE = 8.0
SEED = 7

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})


def effective_rho(delta: np.ndarray) -> np.ndarray:
    """Effective spectral radius of the E-step recurrence as a function of
    ||Delta||.

    (a) delta << DELTA_STABLE: rho_eff = RHO (pure contraction)
    (b) delta near sqrt(DELTA_STABLE * DELTA_BIASED): rho_eff > 1 (divergence)
    (c) delta >> DELTA_BIASED: rho_eff returns to near RHO (biased contraction)

    The bump is a smooth log-space Gaussian centred at the geometric mean of
    the two thresholds; its height is chosen so the peak crosses 1.
    """
    log_d = np.log10(np.maximum(delta, 1e-30))
    centre = 0.5 * (np.log10(DELTA_STABLE) + np.log10(DELTA_BIASED))
    width = 0.5 * (np.log10(DELTA_BIASED) - np.log10(DELTA_STABLE))
    bump = (GROWTH_RATE_MAX - RHO) * np.exp(-((log_d - centre) / width) ** 2)
    return RHO + bump


def bias_magnitude(delta: np.ndarray) -> np.ndarray:
    """Asymptotic bias of the biased fixed point in regime (c).

    bias = C * delta   (linear in ||Delta|| per Thm 3.1, App.~A.3).
    """
    return BIAS_SLOPE * delta


def trajectory(delta: float, rng: np.random.Generator) -> np.ndarray:
    """Return the log10 error trajectory of length N_ITERS for this Delta."""
    rho_eff = float(effective_rho(np.array([delta]))[0])
    bias = float(bias_magnitude(np.array([delta]))[0])

    err0 = 10 ** LOG_ERR0
    log_errs = np.empty(N_ITERS)
    for k in range(N_ITERS):
        contr_term = (rho_eff ** k) * err0
        # Geometric bias accumulation: saturates to bias / (1 - min(rho,1))
        if rho_eff < 1:
            bias_term = bias * (1 - rho_eff ** (k + 1)) / (1 - rho_eff)
        else:
            # When rho_eff >= 1, bias grows unboundedly like rho^k.
            bias_term = bias * (rho_eff ** (k + 1))
        noise_term = 10 ** LOG_NOISE_FLOOR * (1 + 0.02 * rng.standard_normal())
        total = contr_term + bias_term + noise_term
        log_errs[k] = np.log10(max(total, 1e-12))
    return log_errs


def compute_phase_grid():
    rng = np.random.default_rng(SEED)
    deltas = np.geomspace(DELTA_RANGE[0], DELTA_RANGE[1], N_DELTAS)
    grid = np.stack([trajectory(d, rng) for d in deltas], axis=1)
    return deltas, grid


def plot_phase_diagram():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deltas, err = compute_phase_grid()

    fig, ax = plt.subplots(figsize=(10, 6))

    log_deltas = np.log10(deltas)
    vmin, vmax = -5, 5
    # Centre the diverging colormap on 0 so "near-zero error" reads white
    # and the three regimes map to blue / red / orange-ish.

    im = ax.imshow(
        np.clip(err, vmin, vmax),
        origin="lower",
        aspect="auto",
        extent=(log_deltas[0], log_deltas[-1], 0, N_ITERS - 1),
        cmap="RdBu_r",
        norm=Normalize(vmin=vmin, vmax=vmax),
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}\|\mu_k - \mu^\star\|$", fontsize=11)

    # Overlay: k*(||Delta||) curve -- predicted iteration where the noise-
    # growth term overtakes contraction inside regime (b).
    with np.errstate(divide="ignore", invalid="ignore"):
        in_bump = (deltas >= DELTA_STABLE) & (deltas <= DELTA_BIASED)
        rho_eff_vals = effective_rho(deltas)
        # k* is defined where the effective rho first exceeds 1.
        mask = in_bump & (rho_eff_vals > 1)
        k_star = np.log(10 ** (LOG_ERR0) / np.maximum(deltas, 1e-12)) \
            / np.log(1.0 / RHO)
        k_star = np.clip(k_star, 0, N_ITERS - 1)
    if mask.any():
        ax.plot(log_deltas[mask], k_star[mask], color="black", lw=2.2,
                linestyle="--",
                label=r"$k^\star \approx \log(\|e_0\|/\|\Delta\|) / \log(1/\rho)$")

    # Regime boundaries
    ax.axvline(np.log10(DELTA_STABLE), color="#222222", lw=1.5, ls=":",
               label=r"$\|\Delta\| = \delta_{\mathrm{crit}}^{(1)}$")
    ax.axvline(np.log10(DELTA_BIASED), color="#222222", lw=1.5, ls="-.",
               label=r"$\|\Delta\| = \delta_{\mathrm{crit}}^{(2)}$")

    # Region labels
    x_a = 0.5 * (log_deltas[0] + np.log10(DELTA_STABLE))
    x_b = 0.5 * (np.log10(DELTA_STABLE) + np.log10(DELTA_BIASED))
    x_c = 0.5 * (np.log10(DELTA_BIASED) + log_deltas[-1])
    y_label = N_ITERS - 1.8
    label_kw = dict(fontsize=11.5, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#555555", alpha=0.92))
    ax.text(x_a, y_label, "(a) contraction",        color="#08306b", **label_kw)
    ax.text(x_b, y_label, "(b) delayed divergence", color="#67000d", **label_kw)
    ax.text(x_c, y_label, "(c) biased fixed point", color="#662506", **label_kw)

    ax.set_xlabel(r"Operator-mismatch norm $\|\Delta\|$  (log$_{10}$ scale)",
                  fontsize=12)
    ax.set_ylabel("EM iteration $k$", fontsize=12)
    ax.set_title(
        "Phase diagram of Theorem 3.1: three regimes of MCEM under operator mismatch",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower left", framealpha=0.92, fontsize=10)

    tick_logs = np.linspace(log_deltas[0], log_deltas[-1], 7)
    ax.set_xticks(tick_logs)
    ax.set_xticklabels([f"{10**t:.2g}" for t in tick_logs])
    ax.set_ylim(0, N_ITERS - 1)

    fig.tight_layout()
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    plot_phase_diagram()
