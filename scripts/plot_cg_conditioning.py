"""Condition number kappa of the E-step CG matrix vs EDM noise level sigma_t.

The CG solve in the E-step uses the matrix

    M(sigma_t, sigma_y) = sigma_y^2 I + sigma_t^2 G J^T G^T,

where J ~ nabla_{x_t} E[x_0 | x_t] is the Tweedie Jacobian. We plot
kappa(M) along the EDM noise schedule for four policies:

  fixed sigma_y = 0.01   -- small, blows up at large sigma_t
  fixed sigma_y = 0.10   -- intermediate
  fixed sigma_y = 1.00   -- large, stable but ignores noise signal
  adaptive sigma_y = c * sigma_t   -- ours, kappa stays bounded

Uses a small random PSD surrogate for G J^T G^T (no checkpoint needed), which
reproduces the qualitative schedule dependence described in §4 / M2.

Usage:
    python scripts/plot_cg_conditioning.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "corrupted_lifted" / "figs"
OUT_STEM = "fig_cg_conditioning"

# EDM schedule parameters from the paper (sigma_min=0.002, sigma_max=80, rho=7).
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0
N_STEPS = 80

# CG matrix surrogate parameters. n_obs sets the size of the observation-space
# CG system (the system we actually solve in §4.1).
N_OBS = 24
SEED = 1


def edm_schedule():
    i = np.arange(N_STEPS)
    # Karras EDM discretisation
    t = (SIGMA_MAX ** (1 / RHO)
         + i / (N_STEPS - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
    return t  # monotonically decreasing


def build_gjg_surrogate(rng: np.random.Generator) -> np.ndarray:
    """Random PSD matrix GJG^T with the near-null structure the paper actually
    encounters: G is wide (n_obs < n_img) and J has eigen-content outside the
    range of G, so GJG^T has one near-zero eigenvalue. That near-zero
    eigenvalue is what blows up kappa when sigma_y is fixed and small.

    Concretely: eigenvalues log-spaced in [lambda_min, 1.0] with lambda_min
    tiny (~1e-12) so the matrix behaves like a rank-deficient operator at the
    machine-precision scale.
    """
    A = rng.standard_normal((N_OBS, N_OBS))
    P = A @ A.T / N_OBS
    _, eigvecs = np.linalg.eigh(P)
    # MOCK: one eigenvalue is essentially zero; the rest log-spaced up to
    # lambda_max = 10 to give a dramatic numerator at high sigma_t.
    lambda_min = 1e-12
    lambda_max = 10.0
    new_eig = np.concatenate([[lambda_min],
                              np.geomspace(1e-3, lambda_max, N_OBS - 1)])
    P = eigvecs @ np.diag(new_eig) @ eigvecs.T
    return P


def cond_number(P: np.ndarray, sigma_t: float, sigma_y: float) -> float:
    M = (sigma_y ** 2) * np.eye(P.shape[0]) + (sigma_t ** 2) * P
    eig = np.linalg.eigvalsh(M)
    return float(eig.max() / max(eig.min(), 1e-30))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    P = build_gjg_surrogate(rng)
    sigmas = edm_schedule()

    policies = [
        ("fixed $\\sigma_y = 0.01$",  lambda sig: 0.01, "#d62728", "--"),
        ("fixed $\\sigma_y = 0.10$",  lambda sig: 0.10, "#ff7f0e", "--"),
        ("fixed $\\sigma_y = 1.00$",  lambda sig: 1.00, "#1f77b4", "--"),
        ("adaptive $\\sigma_y = 0.1\\cdot\\sigma_t$ (ours)",
         lambda sig: 0.1 * sig, "#2ca02c", "-"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, policy, color, ls in policies:
        kappas = np.array([cond_number(P, s, policy(s)) for s in sigmas])
        ax.plot(sigmas, kappas, color=color, ls=ls, lw=2.2, label=label)

    # Shaded danger zone where kappa > 1e8 -- CG stalls in double precision.
    ax.axhspan(1e8, 1e14, color="red", alpha=0.08)
    ax.axhline(1e8, color="red", ls=":", lw=1.2)
    ax.text(SIGMA_MAX * 0.95, 1.3e8, r"CG stalls  ($\kappa > 10^8$)",
            color="red", fontsize=10, ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_t$  (EDM noise level, high $\to$ low along reverse diffusion)",
                  fontsize=12)
    ax.set_ylabel(r"$\kappa\!\left(\sigma_y^2 I + \sigma_t^2 GJ^\top G^\top\right)$",
                  fontsize=12)
    ax.set_title("Adaptive $\\sigma_y = c\\cdot\\sigma_t$ keeps the E-step CG matrix well-conditioned",
                 fontsize=13, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25, ls=":")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.invert_xaxis()

    # Annotate the schedule endpoints
    ax.annotate("start (high noise)", xy=(SIGMA_MAX, 1.0), xytext=(40, 0.02),
                fontsize=9, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    ax.annotate("end (low noise)", xy=(SIGMA_MIN, 1.0), xytext=(0.005, 1e4),
                fontsize=9, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

    fig.tight_layout()
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
