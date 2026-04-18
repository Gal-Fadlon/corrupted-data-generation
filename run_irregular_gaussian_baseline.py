#!/usr/bin/env python3
"""
Gaussian-noise baseline on top of run_irregular.py (SOTA-style unconditional diffusion + TST).

Goal (time series):
  - Train only on **noisy** windows (additive Gaussian in TS domain after MinMax, same as
    utils_data.real_data_loading + add_gaussian_noise).
  - **No missing data**: missing_rate is forced to 0 (no NaN masking).
  - Evaluate **discriminative score** (and optional memorization) by comparing **generated**
    series to **clean** windows from the same loader (test_loader uses ori_data — see gen_dataloader).

Usage (single noise level):
  python run_irregular_gaussian_baseline.py --config ./configs/seq_len_24/energy.yaml \\
      --gaussian_noise_level 0.10 --wandb true --tags gaussian_baseline energy

Usage (default 5-level grid, easy → hard):
  python run_irregular_gaussian_baseline.py --config ./configs/seq_len_24/energy.yaml \\
      --run_noise_grid --wandb true --tags gaussian_baseline_grid energy

Custom grid:
  python run_irregular_gaussian_baseline.py --config ... --run_noise_grid \\
      --noise_grid 0.03 0.06 0.10 0.14 0.20 --wandb true
"""

import argparse
import copy
import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from run_irregular import main as irregular_main

# Default grid: noise std on MinMax-normalized series in [0, 1] (easy → hard).
DEFAULT_NOISE_GRID = (0.03, 0.06, 0.09, 0.12, 0.18)


def _parse_leading_grid_flags() -> Tuple[argparse.Namespace, List[str]]:
    """Parse --run_noise_grid / --noise_grid without conflicting with parse_args_irregular()."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--run_noise_grid",
        action="store_true",
        help="Run one full training job per σ in --noise_grid (separate W&B runs).",
    )
    pre.add_argument(
        "--noise_grid",
        type=float,
        nargs="+",
        default=None,
        help="Gaussian noise stds for training corruption (MinMax units). Default: 5-step easy→hard.",
    )
    known, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, remaining


def _force_gaussian_baseline_constraints(args, sigma: Optional[float]) -> None:
    """No masking; training noise only via gaussian_noise_level."""
    if getattr(args, "missing_rate", 0.0) != 0.0:
        logging.warning(
            "[gaussian_baseline] missing_rate=%s → forcing 0 (no missing values; noise-only corruption).",
            args.missing_rate,
        )
        args.missing_rate = 0.0
    if sigma is not None:
        args.gaussian_noise_level = float(sigma)


def validate_ts_gaussian_pipeline(args) -> None:
    """
    End-to-end checks (same code path as training data):

    1. ``real_data_loading`` loads MinMax-normalized ``ori_data`` (T × D), adds
       ``add_gaussian_noise(ori_data, sigma)`` **before** sliding windows — noise is in
       **time series domain** (per timestep, per feature).
    2. With ``missing_rate == 0``, no rows are replaced by NaN; training windows have no
       missing values (only an extra index column on the noisy side).
    3. Clean windows ``data`` and noisy windows ``irregular_data`` use the **same**
       shuffle index, so we can verify ``noisy_ts - clean`` has empirical std ≈ ``sigma``.

    See ``utils_data.real_data_loading`` and ``gen_dataloader`` (train = noisy,
    test = clean ``ori_data`` windows). Evaluation in ``run_irregular`` compares
    generated TS to **clean** ``test_loader`` batches.
    """
    from utils.utils_data import (
        REAL_DATASETS_FOR_GAUSSIAN_BASELINE,
        build_gaussian_baseline_corruption_pairs,
    )

    if args.dataset not in REAL_DATASETS_FOR_GAUSSIAN_BASELINE:
        logging.info(
            "[gaussian_baseline/validate] skip detailed TS check for dataset=%s (not CSV/real_data_loading path)",
            args.dataset,
        )
        return

    if getattr(args, "mix_missing_rates", False):
        raise RuntimeError(
            "gaussian_baseline: mix_missing_rates must be False (use standard real_data_loading)."
        )
    mr = float(getattr(args, "missing_rate", 0.0))
    if mr != 0.0:
        raise RuntimeError(f"gaussian_baseline: missing_rate must be 0, got {mr}")

    sigma = float(args.gaussian_noise_level)
    if sigma <= 0:
        raise RuntimeError(f"gaussian_baseline: gaussian_noise_level must be > 0, got {sigma}")

    clean, noisy_ts = build_gaussian_baseline_corruption_pairs(args, seed=args.seed)
    n = min(3000, clean.shape[0])
    if n < 2:
        raise RuntimeError("Not enough windows to validate (need len >= 2).")

    clean = clean[:n].astype(np.float64)
    noisy_ts = noisy_ts[:n].astype(np.float64)

    if clean.shape != noisy_ts.shape:
        raise RuntimeError(f"Shape mismatch clean {clean.shape} vs noisy_ts {noisy_ts.shape}")

    if np.isnan(noisy_ts).any():
        raise RuntimeError("NaNs in noisy training windows with missing_rate=0 (unexpected).")

    diff = noisy_ts - clean
    emp_std = float(np.std(diff))
    # Finite-sample tolerance; additive N(0, sigma^2) per entry
    tol = max(0.12 * sigma, 0.015)
    if abs(emp_std - sigma) > tol:
        logging.warning(
            "[gaussian_baseline/validate] empirical std(noisy-clean)=%.6f vs sigma=%.6f "
            "(tol=%.6f). Check noise_timestep override or data path.",
            emp_std,
            sigma,
            tol,
        )
    else:
        logging.info(
            "[gaussian_baseline/validate] TS-domain noise OK: std(noisy-clean)=%.6f ≈ sigma=%.6f",
            emp_std,
            sigma,
        )


def _bump_log_dir(args, sigma: float) -> None:
    """Avoid checkpoint/log collisions across grid runs."""
    root = getattr(args, "log_dir", "./logs") or "./logs"
    sub = f"gaussian_baseline/sigma_{sigma:.4f}".replace(".", "p")
    args.log_dir = os.path.join(root, sub)


def main() -> None:
    grid_flags, _ = _parse_leading_grid_flags()

    from utils.utils_args import parse_args_irregular

    args = parse_args_irregular()
    import torch

    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if getattr(args, "pure_diffem", False) or getattr(args, "use_diffem", False):
        print(
            "[gaussian_baseline] DiffEM redirect is disabled here; use run_diffem*.py for EM.",
            file=sys.stderr,
        )
        sys.exit(2)

    if grid_flags.run_noise_grid:
        grid = tuple(grid_flags.noise_grid) if grid_flags.noise_grid else DEFAULT_NOISE_GRID
        if len(grid) < 1:
            print("[gaussian_baseline] --noise_grid must list at least one σ.", file=sys.stderr)
            sys.exit(1)
        logging.info("[gaussian_baseline] Running noise grid (%d levels): %s", len(grid), grid)
        base_args = copy.deepcopy(args)
        for sigma in grid:
            run_args = copy.deepcopy(base_args)
            _force_gaussian_baseline_constraints(run_args, sigma)
            _bump_log_dir(run_args, sigma)
            tag_sigma = f"gauss_sigma_{sigma:.4f}".replace(".", "p")
            run_args.tags = list(run_args.tags or []) + [
                "gaussian_noise_baseline",
                "no_missing",
                tag_sigma,
            ]
            logging.info(
                "[gaussian_baseline] Starting run with gaussian_noise_level=%s (eval vs clean test set)",
                run_args.gaussian_noise_level,
            )
            validate_ts_gaussian_pipeline(run_args)
            irregular_main(run_args)
        return

    _force_gaussian_baseline_constraints(args, None)
    if getattr(args, "gaussian_noise_level", 0) is None or float(args.gaussian_noise_level) <= 0:
        print(
            "[gaussian_baseline] Set --gaussian_noise_level > 0 (noise std on MinMax data), "
            "or use --run_noise_grid.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.tags = list(args.tags or []) + ["gaussian_noise_baseline", "no_missing"]
    logging.info(
        "[gaussian_baseline] gaussian_noise_level=%s, missing_rate=%s "
        "(discriminative score: generated vs clean windows)",
        args.gaussian_noise_level,
        args.missing_rate,
    )
    validate_ts_gaussian_pipeline(args)
    irregular_main(args)


if __name__ == "__main__":
    main()
