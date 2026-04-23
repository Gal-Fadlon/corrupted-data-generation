"""Generate figs/fig_regime_c_continuous.pdf for App. E.5.

Overlays the binary-mask no-obs-CG trajectory (regime b --- initial descent then
divergence at iter 9) with the pilot measurements of the continuous-time
no-obs-CG run (three measured points at iters 0, 3, 5).  No interpolation is
drawn between the continuous points; the full continuous trajectory is
populated by the scheduled 21-run matrix (App. E.4, tab:continuous-trajectory).

Data sources:
  * binary:     the "Baseline (no obs-CG, regime b)" row of tab:convergence
                (main.tex, Energy 50%), iterations {0,1,2,3,4,5,7,9}.  Real
                measurements; iters 6 and 8 are not in the source table.
  * continuous: the pilot Energy seq24 rho=0.5 no-obs-CG run, with real
                measurements at iters {0, 3, 5} = {0.44, 0.47, 0.49}.

Design choices:
  * No fabricated continuous points: only the three measured values are drawn.
  * No connecting line on the continuous points: avoids implying a trajectory
    that was not measured.
  * A neutral dashed band shows the theorem-predicted envelope (monotone climb
    from ~0.44 toward 0.5) as a qualitative prediction, clearly labeled.
  * Binary trajectory is drawn in full as in tab:convergence.

Usage:
    python scripts/plot_regime_c_continuous.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --- Binary no-obs-CG (regime b) --- real numbers from tab:convergence ---
BIN_ITERS  = np.array([0, 1, 2, 3, 4, 5, 7, 9], dtype=float)
BIN_SCORES = np.array([0.387, 0.185, 0.068, 0.049, 0.049, 0.057, 0.055, 0.456])


# --- Continuous no-obs-CG (regime c) --- three measured pilot points ---
CONT_MEASURED_ITERS  = np.array([0, 3, 5], dtype=float)
CONT_MEASURED_SCORES = np.array([0.440, 0.470, 0.490])


def make_plot(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.5))

    # Random-chance line
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.text(9.3, 0.502, "random chance (0.5)", fontsize=8, ha="right",
            va="bottom", color="black", alpha=0.65)

    # --- Binary regime-(b): full measured trajectory ---
    ax.plot(BIN_ITERS, BIN_SCORES, "-o", color="#1f77b4",
            linewidth=2.0, markersize=6,
            label=r"Binary MCAR, no-obs-CG  (regime b, measured)")
    ax.annotate(r"divergence between iter 7 and iter 9",
                xy=(9, 0.456), xytext=(5.2, 0.35),
                fontsize=8.5, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0))
    # Shade the initial-convergence window
    ax.axvspan(0, 5, color="#1f77b4", alpha=0.05)
    ax.text(2.5, 0.28, "initial convergence\n(regime b descent)",
            fontsize=8, ha="center", va="center", color="#1f77b4", alpha=0.75)

    # --- Continuous regime-(c): only the three measured pilot points ---
    ax.plot(CONT_MEASURED_ITERS, CONT_MEASURED_SCORES, "o",
            color="#d62728", markersize=8.5,
            markeredgecolor="black", markeredgewidth=0.8,
            label=r"Continuous 2-tap, no-obs-CG  (pilot, 3 measured points)")
    # Annotate each measured value
    for (it, sc) in zip(CONT_MEASURED_ITERS, CONT_MEASURED_SCORES):
        ax.annotate(f"{sc:.2f}", xy=(it, sc), xytext=(it + 0.15, sc + 0.013),
                    fontsize=8, color="#d62728")

    # --- Theorem-1 prediction band (not data) ---
    # Qualitative envelope: monotone climb from the iter-0 measured value toward 0.5
    # over the scheduled matrix iterations.  Drawn as a dashed grey arrow.
    ax.annotate("", xy=(9, 0.500), xytext=(0, 0.440),
                arrowprops=dict(arrowstyle="->", color="grey",
                                linestyle="--", lw=1.2, alpha=0.7))
    ax.text(4.5, 0.515, "Thm. 1 prediction (regime c): monotone climb to 0.5",
            fontsize=8.5, color="grey", alpha=0.85, ha="center")

    # Axis cosmetics
    ax.set_xlabel("EM iteration $k$", fontsize=11)
    ax.set_ylabel(r"discriminative score $\downarrow$", fontsize=11)
    ax.set_title("Regime jump under no-obs-CG: binary (b) vs. continuous (c)",
                 fontsize=11)
    ax.set_xlim(-0.4, 10.0)
    ax.set_ylim(0.0, 0.56)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 7, 9])
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    leg = ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95)
    leg.get_frame().set_edgecolor("gray")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), format="png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default="corrupted_lifted/figs/fig_regime_c_continuous.pdf")
    args = ap.parse_args()
    make_plot(Path(args.out))


if __name__ == "__main__":
    main()
