"""
STL Decomposition Utilities for Decomposition-Enhanced DiffEM.

Provides batch STL decomposition (trend + seasonal + residual)
and STL-informed initialization for time series with missing values.

References:
    - STL: Cleveland et al., "STL: A Seasonal-Trend Decomposition Procedure
      Based on Loess" (1990)
    - STDiffusion: Ma & Huang, "Effective Series Decomposition and Components
      Learning for Time Series Generation" (2025)
"""

import gc
import signal
import time
import warnings

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL


def auto_detect_period(seq_len):
    """
    Auto-detect period for STL based on sequence length.
    STL requires period >= 3 and odd.
    """
    if seq_len >= 48:
        period = 24
    elif seq_len >= 24:
        period = 12
    elif seq_len >= 14:
        period = 7
    else:
        period = max(3, seq_len // 4)

    if period % 2 == 0:
        period += 1
    period = max(3, period)

    if period >= seq_len:
        period = max(3, (seq_len // 2) - 1)
        if period % 2 == 0:
            period -= 1

    return period


def stl_decompose_single(ts_np, period=None, robust=True):
    """
    STL decomposition of a single 1-D numpy series.
    Returns (trend, seasonal, residual) as numpy arrays.
    """
    seq_len = len(ts_np)
    if period is None:
        period = auto_detect_period(seq_len)
    if period >= seq_len:
        period = max(3, (seq_len // 2) - 1)
        if period % 2 == 0:
            period -= 1

    try:
        result = STL(ts_np, period=period, robust=robust).fit()
        return result.trend, result.seasonal, result.resid
    except Exception:
        window = min(period, seq_len // 3)
        if window % 2 == 0:
            window += 1
        trend = np.convolve(ts_np, np.ones(window) / window, mode='same')
        half = window // 2
        for i in range(half):
            trend[i] = np.mean(ts_np[:i + half + 1])
            trend[-(i + 1)] = np.mean(ts_np[-(i + half + 1):])
        seasonal = np.zeros_like(ts_np)
        residual = ts_np - trend
        return trend, seasonal, residual


def robust_decompose_single(ts_np, period=None):
    """
    Robust decomposition with STL first, falling back to moving-average
    trend + periodic-averaging seasonal.

    Unlike stl_decompose_single, the fallback here produces a real seasonal
    component (via periodic averaging) rather than zeros.
    """
    seq_len = len(ts_np)
    if period is None:
        period = auto_detect_period(seq_len)
    if period >= seq_len:
        period = max(3, (seq_len // 2) - 1)
        if period % 2 == 0:
            period -= 1

    try:
        result = STL(ts_np, period=period, robust=True).fit()
        return result.trend, result.seasonal, result.resid
    except Exception:
        pass

    # Fallback: moving-average trend
    window = min(period, max(3, seq_len // 3))
    if window % 2 == 0:
        window += 1
    trend = np.convolve(ts_np, np.ones(window) / window, mode='same')
    half = window // 2
    for i in range(half):
        trend[i] = np.mean(ts_np[:i + half + 1])
        trend[-(i + 1)] = np.mean(ts_np[-(i + half + 1):])

    # Fallback: periodic averaging for seasonal
    detrended = ts_np - trend
    seasonal = np.zeros_like(ts_np)
    for k in range(period):
        indices = np.arange(k, seq_len, period)
        seasonal[indices] = detrended[indices].mean()

    residual = ts_np - trend - seasonal
    return trend, seasonal, residual


def robust_decompose_batch(x, period=None):
    """
    Batch robust decomposition for multivariate time series.

    Args:
        x: numpy array (batch, seq_len, features) or torch tensor -- NO NaN
        period: decomposition period (auto-detected if None)

    Returns:
        trends, seasonals, residuals -- same type and shape as x
    """
    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        device, dtype = x.device, x.dtype
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x

    B, T, C = x_np.shape
    if period is None:
        period = auto_detect_period(T)

    trends = np.zeros_like(x_np)
    seasonals = np.zeros_like(x_np)
    residuals = np.zeros_like(x_np)

    for b in range(B):
        for c in range(C):
            t, s, r = robust_decompose_single(x_np[b, :, c], period)
            trends[b, :, c] = t
            seasonals[b, :, c] = s
            residuals[b, :, c] = r

    if is_torch:
        return (torch.from_numpy(trends).to(device=device, dtype=dtype),
                torch.from_numpy(seasonals).to(device=device, dtype=dtype),
                torch.from_numpy(residuals).to(device=device, dtype=dtype))
    return trends, seasonals, residuals


def fast_decompose_batch(x, period=None):
    """
    Fully vectorized decomposition: moving-average trend + periodic-averaging
    seasonal, operating on the entire (N, T, C) array at once via scipy/numpy.

    ~10,000x faster than the serial STL loop for typical dataset sizes.
    For seq_len=24 with period=13, STL's iterative LOESS is overkill;
    this produces equivalent results in seconds instead of minutes.

    Args:
        x: numpy array (N, T, C) or torch tensor -- NO NaN
        period: decomposition period (auto-detected if None)

    Returns:
        trends, seasonals, residuals -- same type and shape as x
    """
    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        device, dtype = x.device, x.dtype
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float64)

    B, T, C = x_np.shape
    if period is None:
        period = auto_detect_period(T)

    window = min(period, max(3, T // 3))
    if window % 2 == 0:
        window += 1

    # Trend: 1-D moving average along the time axis (axis=1)
    # uniform_filter1d operates on the full (N, T, C) array in C
    trends = uniform_filter1d(x_np, size=window, axis=1, mode='nearest')

    # Seasonal: periodic averaging of the detrended signal
    detrended = x_np - trends
    seasonals = np.zeros_like(x_np)
    for k in range(period):
        idx = np.arange(k, T, period)
        # Mean across all occurrences of phase k: (N, len(idx), C) -> (N, 1, C)
        phase_mean = detrended[:, idx, :].mean(axis=1, keepdims=True)
        seasonals[:, idx, :] = phase_mean

    residuals = x_np - trends - seasonals

    if is_torch:
        return (torch.from_numpy(trends).to(device=device, dtype=dtype),
                torch.from_numpy(seasonals).to(device=device, dtype=dtype),
                torch.from_numpy(residuals).to(device=device, dtype=dtype))
    return trends, seasonals, residuals


def stl_decompose_batch(x, period=None, robust=True):
    """
    Batch STL decomposition for multivariate time series.

    Args:
        x: numpy array (batch, seq_len, features) or torch tensor — NO NaN allowed
        period: STL period (auto-detected if None)
        robust: use robust STL fitting

    Returns:
        trends, seasonals, residuals — same type and shape as x
    """
    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        device, dtype = x.device, x.dtype
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x

    B, T, C = x_np.shape
    if period is None:
        period = auto_detect_period(T)

    trends = np.zeros_like(x_np)
    seasonals = np.zeros_like(x_np)
    residuals = np.zeros_like(x_np)

    for b in range(B):
        for c in range(C):
            t, s, r = stl_decompose_single(x_np[b, :, c], period, robust)
            trends[b, :, c] = t
            seasonals[b, :, c] = s
            residuals[b, :, c] = r

    if is_torch:
        return (torch.from_numpy(trends).to(device=device, dtype=dtype),
                torch.from_numpy(seasonals).to(device=device, dtype=dtype),
                torch.from_numpy(residuals).to(device=device, dtype=dtype))
    return trends, seasonals, residuals


def initialize_with_stl(corrupted_data, obs_masks, period=None, seed=None):
    """
    STL-informed initialization for corrupted time series.

    Strategy:
    1. Linear-interpolate missing values per sample/feature
    2. STL-decompose the interpolated series
    3. For missing positions: keep interpolated trend + seasonal,
       resample residual from a Gaussian fitted on observed residuals
    4. Reconstruct as trend + seasonal + residual

    Args:
        corrupted_data: (N, T, C) numpy array with NaN for missing
        obs_masks:      (N, T) boolean, True = observed
        period:         STL period (auto-detected if None)
        seed:           random seed

    Returns:
        initial_recon:  (N, T, C) with all values filled
    """
    if seed is not None:
        np.random.seed(seed)

    N, T, C = corrupted_data.shape
    if period is None:
        period = auto_detect_period(T)

    initial_recon = corrupted_data.copy()

    for i in range(N):
        mask_i = obs_masks[i]  # (T,)
        obs_idx = np.where(mask_i)[0]
        mis_idx = np.where(~mask_i)[0]

        if len(obs_idx) < 3 or len(mis_idx) == 0:
            # Too few observations for STL; fall back to Gaussian fill
            for c in range(C):
                obs_vals = corrupted_data[i, obs_idx, c] if len(obs_idx) > 0 else np.array([0.5])
                mu, sigma = np.nanmean(obs_vals), max(np.nanstd(obs_vals), 1e-3)
                initial_recon[i, mis_idx, c] = np.random.normal(mu, sigma, size=len(mis_idx))
            continue

        for c in range(C):
            series = corrupted_data[i, :, c].copy()

            # Step 1: linear-interpolate missing values
            obs_vals = series[obs_idx]
            series[mis_idx] = np.interp(mis_idx, obs_idx, obs_vals)

            # Step 2: STL decompose the interpolated series
            trend_c, season_c, resid_c = stl_decompose_single(series, period)

            # Step 3: for missing positions, re-sample residual
            resid_obs = resid_c[obs_idx]
            mu_r, std_r = resid_obs.mean(), max(resid_obs.std(), 1e-4)
            resid_c[mis_idx] = np.random.normal(mu_r, std_r, size=len(mis_idx))

            initial_recon[i, :, c] = trend_c + season_c + resid_c

    initial_recon = np.clip(initial_recon, 0, 1)

    obs_pct = obs_masks.sum() / obs_masks.size * 100
    print(f"STL initialization complete. {N} sequences, {obs_pct:.1f}% observed, period={period}")
    return initial_recon


def initialize_with_knn_stl(corrupted_data, obs_masks, period=None, k=5, seed=None):
    """
    k-NN + STL initialization for corrupted time series (improved for high missing rates).

    Strategy:
    1. For each sample, find k nearest neighbors by distance on observed positions
    2. Fill missing positions with neighbor-averaged values (where neighbors have obs)
    3. Linear-interpolate any remaining gaps
    4. STL-decompose the filled series
    5. For missing positions: trend + seasonal from STL, residual from GLOBAL pooled stats
    6. Reconstruct as trend + seasonal + residual

    Args:
        corrupted_data: (N, T, C) numpy array with NaN for missing
        obs_masks:      (N, T) boolean, True = observed
        period:         STL period (auto-detected if None)
        k:              number of nearest neighbors (default 5)
        seed:           random seed

    Returns:
        initial_recon:  (N, T, C) with all values filled
    """
    if seed is not None:
        np.random.seed(seed)

    N, T, C = corrupted_data.shape
    if period is None:
        period = auto_detect_period(T)

    # Replace NaN with 0 for distance computation
    data_clean = np.nan_to_num(corrupted_data, nan=0.0)

    # Compute pairwise distances on overlapping observed positions
    # dist[i,j] = sqrt(sum over (t,c) where both i and j observed: (x_i - x_j)^2)
    dist = np.full((N, N), np.inf)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            both_obs = obs_masks[i] & obs_masks[j]
            if both_obs.sum() < 2:
                continue
            diff = (data_clean[i] - data_clean[j]) ** 2
            diff[~both_obs, :] = 0
            d = np.sqrt(diff.sum())
            dist[i, j] = d

    initial_recon = corrupted_data.copy()

    for i in range(N):
        mask_i = obs_masks[i]
        obs_idx = np.where(mask_i)[0]
        mis_idx = np.where(~mask_i)[0]

        if len(mis_idx) == 0:
            continue

        # Get k nearest neighbors (excluding self)
        row = dist[i].copy()
        row[i] = np.inf
        neighbors = np.argsort(row)[:k]

        for c in range(C):
            series = corrupted_data[i, :, c].copy()

            # Step 1: k-NN fill for missing positions
            for t in mis_idx:
                vals = []
                for j in neighbors:
                    if dist[i, j] == np.inf:
                        continue
                    if obs_masks[j, t] and not np.isnan(corrupted_data[j, t, c]):
                        vals.append(corrupted_data[j, t, c])
                if vals:
                    series[t] = np.mean(vals)
                # else: leave as NaN for linear interp below

            # Step 2: linear-interpolate any remaining NaN
            still_nan = np.isnan(series)
            if still_nan.any() and len(obs_idx) >= 2:
                series[still_nan] = np.interp(
                    np.where(still_nan)[0], obs_idx,
                    corrupted_data[i, obs_idx, c]
                )
            elif still_nan.any():
                series[still_nan] = np.nanmean(corrupted_data[:, :, c])

            # Step 3: STL decompose
            if len(obs_idx) < 3:
                initial_recon[i, :, c] = np.clip(series, 0, 1)
                continue

            try:
                trend_c, season_c, resid_c = stl_decompose_single(series, period)
            except Exception:
                initial_recon[i, :, c] = np.clip(series, 0, 1)
                continue

            # Step 4: resample residual for missing positions
            resid_obs = resid_c[obs_idx]
            mu_r = resid_obs.mean()
            std_r = max(resid_obs.std(), 1e-4)
            resid_c[mis_idx] = np.random.normal(mu_r, std_r, size=len(mis_idx))
            initial_recon[i, :, c] = trend_c + season_c + resid_c

    initial_recon = np.clip(initial_recon, 0, 1)

    obs_pct = obs_masks.sum() / obs_masks.size * 100
    print(f"k-NN+STL initialization complete. {N} sequences, {obs_pct:.1f}% observed, k={k}, period={period}")
    return initial_recon


# =============================================================================
# Experiment 1: GP with Composite Kernel
# =============================================================================

def initialize_with_gp(corrupted_data, obs_masks, period=None, seed=None):
    """
    Gaussian Process initialization for corrupted time series.

    Uses a composite kernel (RBF + ExpSineSquared + WhiteKernel) that explicitly
    encodes trend and seasonality. The GP fits only on observed points and
    predicts at all timepoints, so no pre-interpolation is needed.

    Reference:
        "An improved Gaussian process for filling missing data in GNSS
         position time series" (Nature Sci. Reports, 2024)

    Args:
        corrupted_data: (N, T, C) numpy array with NaN for missing
        obs_masks:      (N, T) boolean, True = observed
        period:         seasonal period (auto-detected if None)
        seed:           random seed

    Returns:
        initial_recon: (N, T, C) with all values filled
    """
    if seed is not None:
        np.random.seed(seed)

    N, T, C = corrupted_data.shape
    if period is None:
        period = auto_detect_period(T)

    initial_recon = corrupted_data.copy()
    all_t = np.arange(T).reshape(-1, 1)

    kernel = (
        RBF(length_scale=T / 4.0, length_scale_bounds=(1.0, T * 2.0))
        + ExpSineSquared(
            length_scale=1.0, periodicity=float(period),
            length_scale_bounds=(0.5, T),
            periodicity_bounds=(max(2.0, period * 0.5), period * 2.0),
        )
        + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
    )

    rng = np.random.RandomState(seed)
    n_gp_ok = 0
    n_fallback = 0

    for i in range(N):
        mask_i = obs_masks[i]
        obs_idx = np.where(mask_i)[0]
        mis_idx = np.where(~mask_i)[0]

        if len(mis_idx) == 0:
            continue

        if len(obs_idx) < 3:
            for c in range(C):
                obs_vals = corrupted_data[i, obs_idx, c] if len(obs_idx) > 0 else np.array([0.5])
                mu, sigma = np.nanmean(obs_vals), max(np.nanstd(obs_vals), 1e-3)
                initial_recon[i, mis_idx, c] = rng.normal(mu, sigma, size=len(mis_idx))
            n_fallback += 1
            continue

        X_obs = obs_idx.reshape(-1, 1)

        for c in range(C):
            obs_vals = corrupted_data[i, obs_idx, c]

            # Per-feature NaN filtering: the obs_mask is per-timestep (any
            # feature), so individual features can still be NaN at "observed"
            # timesteps.
            valid = ~np.isnan(obs_vals)
            if valid.sum() < 3:
                feat_vals = corrupted_data[i, :, c]
                mu = np.nanmean(feat_vals) if np.any(~np.isnan(feat_vals)) else 0.5
                initial_recon[i, mis_idx, c] = mu + rng.normal(0, 0.01, size=len(mis_idx))
                n_fallback += 1
                continue

            try:
                gp = GaussianProcessRegressor(
                    kernel=kernel, n_restarts_optimizer=2, random_state=seed,
                    alpha=1e-6, normalize_y=True,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(X_obs[valid], obs_vals[valid])

                y_pred, y_std = gp.predict(all_t, return_std=True)

                filled = corrupted_data[i, :, c].copy()
                filled[mis_idx] = y_pred[mis_idx]
                initial_recon[i, :, c] = filled
                n_gp_ok += 1
            except Exception:
                filled = corrupted_data[i, :, c].copy()
                valid_all = ~np.isnan(filled)
                if valid_all.sum() >= 2:
                    filled[~valid_all] = np.interp(
                        np.where(~valid_all)[0],
                        np.where(valid_all)[0],
                        filled[valid_all]
                    )
                else:
                    mu = np.nanmean(filled) if np.any(valid_all) else 0.5
                    filled[:] = mu + rng.normal(0, 0.01, size=T)
                initial_recon[i, :, c] = filled
                n_fallback += 1

    initial_recon = np.clip(initial_recon, 0, 1)

    obs_pct = obs_masks.sum() / obs_masks.size * 100
    print(f"GP initialization complete. {N} sequences, {obs_pct:.1f}% observed, period={period}")
    print(f"  GP OK: {n_gp_ok}, Fallback (interp/mean): {n_fallback}")
    return initial_recon


# =============================================================================
# Experiment 2: Iterative Decompose-Impute-Redecompose
# =============================================================================

def initialize_with_iterative_stl(corrupted_data, obs_masks, period=None,
                                   n_iters=5, seed=None):
    """
    Iterative STL initialization for corrupted time series.

    Instead of interpolate-once then decompose, this method iterates:
      1. Initialize missing with per-feature mean
      2. Repeat n_iters times:
         a. STL decompose the full series
         b. Impute missing = trend + seasonal + resampled residual
         c. Restore observed values
    Each iteration refines the decomposition since imputed values
    progressively better reflect the true structure.

    Reference:
        STLinterp (Harrington, GitHub) — iterative STL-based NA estimation

    Args:
        corrupted_data: (N, T, C) numpy array with NaN for missing
        obs_masks:      (N, T) boolean, True = observed
        period:         STL period (auto-detected if None)
        n_iters:        number of decompose-impute iterations (default 5)
        seed:           random seed

    Returns:
        initial_recon: (N, T, C) with all values filled
    """
    if seed is not None:
        np.random.seed(seed)

    N, T, C = corrupted_data.shape
    if period is None:
        period = auto_detect_period(T)

    initial_recon = corrupted_data.copy()

    for i in range(N):
        mask_i = obs_masks[i]
        obs_idx = np.where(mask_i)[0]
        mis_idx = np.where(~mask_i)[0]

        if len(obs_idx) < 3 or len(mis_idx) == 0:
            for c in range(C):
                obs_vals = corrupted_data[i, obs_idx, c] if len(obs_idx) > 0 else np.array([0.5])
                mu, sigma = np.nanmean(obs_vals), max(np.nanstd(obs_vals), 1e-3)
                initial_recon[i, mis_idx, c] = np.random.normal(mu, sigma, size=len(mis_idx))
            continue

        for c in range(C):
            series = corrupted_data[i, :, c].copy()
            obs_vals_orig = series[obs_idx].copy()

            # Initialize missing with per-feature observed mean
            feat_mean = np.nanmean(series)
            series[mis_idx] = feat_mean

            for k in range(n_iters):
                trend_c, season_c, resid_c = stl_decompose_single(series, period)

                resid_obs = resid_c[obs_idx]
                mu_r, std_r = resid_obs.mean(), max(resid_obs.std(), 1e-4)
                resid_c[mis_idx] = np.random.normal(mu_r, std_r, size=len(mis_idx))

                series = trend_c + season_c + resid_c
                # Restore observed values to prevent drift
                series[obs_idx] = obs_vals_orig

            initial_recon[i, :, c] = series

    initial_recon = np.clip(initial_recon, 0, 1)

    obs_pct = obs_masks.sum() / obs_masks.size * 100
    print(f"Iterative STL initialization complete ({n_iters} iters). "
          f"{N} sequences, {obs_pct:.1f}% observed, period={period}")
    return initial_recon


# =============================================================================
# Experiment 3: Kalman Smoother with Structural TS Model
# =============================================================================

def _kalman_timeout_handler(signum, frame):
    raise TimeoutError("Kalman fit exceeded time limit")


def _kalman_fallback_stl(series, obs_idx, mis_idx, period):
    """Iterative STL fallback for a single feature series."""
    obs_vals_orig = series[obs_idx].copy()
    feat_mean = np.nanmean(series)
    series[mis_idx] = feat_mean
    for k in range(3):
        trend_c, season_c, resid_c = stl_decompose_single(series, period)
        resid_obs = resid_c[obs_idx]
        mu_r, std_r = resid_obs.mean(), max(resid_obs.std(), 1e-4)
        resid_c[mis_idx] = np.random.normal(mu_r, std_r, size=len(mis_idx))
        series = trend_c + season_c + resid_c
        series[obs_idx] = obs_vals_orig
    return series


def initialize_with_kalman(corrupted_data, obs_masks, period=None, seed=None,
                           per_fit_timeout=None, max_seconds=None):
    """
    Kalman smoother initialization using statsmodels UnobservedComponents.

    The structural time series model explicitly decomposes into
    level (trend) + seasonal + irregular. The Kalman smoother natively
    handles missing values by skipping the measurement update step.

    Includes per-fit timeout, a global time budget with
    automatic fallback to iterative STL, and periodic progress logging.

    Reference:
        Harvey (1989), "Forecasting, Structural Time Series Models
        and the Kalman Filter"

    Args:
        corrupted_data: (N, T, C) numpy array with NaN for missing
        obs_masks:      (N, T) boolean, True = observed
        period:         seasonal period (auto-detected if None)
        seed:           random seed
        per_fit_timeout: seconds per model.fit() call (default 5)
        max_seconds:    global time budget in seconds (default 1800)

    Returns:
        initial_recon: (N, T, C) with all values filled
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    if seed is not None:
        np.random.seed(seed)

    N, T, C = corrupted_data.shape
    if period is None:
        period = auto_detect_period(T)

    initial_recon = corrupted_data.copy()
    n_kalman_ok = 0
    n_fallback = 0
    n_timeout = 0
    n_budget_fallback = 0

    MAX_KALMAN_SECONDS = max_seconds if max_seconds is not None else 1800
    PER_FIT_TIMEOUT = per_fit_timeout if per_fit_timeout is not None else 5
    start_time = time.time()
    budget_exceeded = False

    prev_alarm_handler = signal.signal(signal.SIGALRM, _kalman_timeout_handler)

    try:
        for i in range(N):
            if (i + 1) % 200 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"  Kalman init: {i+1}/{N} sequences "
                      f"({elapsed:.0f}s, {rate:.1f} seq/s, "
                      f"kalman={n_kalman_ok}, fallback={n_fallback}, "
                      f"timeout={n_timeout})")

            if not budget_exceeded and time.time() - start_time > MAX_KALMAN_SECONDS:
                budget_exceeded = True
                remaining = N - i
                print(f"  Global time budget ({MAX_KALMAN_SECONDS}s) exceeded at "
                      f"sequence {i}/{N}. Falling back to iterative STL "
                      f"for remaining {remaining} sequences.")

            mask_i = obs_masks[i]
            obs_idx = np.where(mask_i)[0]
            mis_idx = np.where(~mask_i)[0]

            if len(mis_idx) == 0:
                continue

            if len(obs_idx) < 3:
                for c in range(C):
                    obs_vals = corrupted_data[i, obs_idx, c] if len(obs_idx) > 0 else np.array([0.5])
                    mu, sigma = np.nanmean(obs_vals), max(np.nanstd(obs_vals), 1e-3)
                    initial_recon[i, mis_idx, c] = np.random.normal(mu, sigma, size=len(mis_idx))
                n_fallback += 1
                continue

            for c in range(C):
                series = corrupted_data[i, :, c].copy().astype(np.float64)

                feat_valid = ~np.isnan(series)
                if feat_valid.sum() < 3:
                    mu = np.nanmean(series) if feat_valid.any() else 0.5
                    initial_recon[i, :, c] = mu + np.random.normal(0, 0.01, size=T)
                    n_fallback += 1
                    continue

                if not feat_valid.all():
                    series[~feat_valid] = np.interp(
                        np.where(~feat_valid)[0],
                        np.where(feat_valid)[0],
                        series[feat_valid]
                    )

                if budget_exceeded:
                    initial_recon[i, :, c] = _kalman_fallback_stl(
                        series, obs_idx, mis_idx, period)
                    n_budget_fallback += 1
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = UnobservedComponents(
                            series, level='local linear trend',
                            seasonal=period, stochastic_seasonal=True,
                        )
                        signal.alarm(PER_FIT_TIMEOUT)
                        result = model.fit(disp=False, maxiter=30)
                        signal.alarm(0)

                    smoothed_state = result.smoothed_state
                    smoothed = smoothed_state[0, :]  # level
                    if model.k_states > 2:
                        smoothed = smoothed + smoothed_state[2, :]  # + seasonal

                    filled = series.copy()
                    filled[mis_idx] = smoothed[mis_idx]

                    if np.isnan(filled[mis_idx]).any():
                        raise ValueError("Kalman smoother produced NaN")

                    obs_residuals = series[obs_idx] - smoothed[obs_idx]
                    obs_residuals = obs_residuals[~np.isnan(obs_residuals)]
                    if len(obs_residuals) > 1:
                        std_r = max(np.std(obs_residuals), 1e-4)
                        filled[mis_idx] += np.random.normal(0, std_r * 0.5, size=len(mis_idx))

                    initial_recon[i, :, c] = filled
                    n_kalman_ok += 1

                    del model, result, smoothed_state, smoothed, filled
                    gc.collect()

                except TimeoutError:
                    signal.alarm(0)
                    n_timeout += 1
                    initial_recon[i, :, c] = _kalman_fallback_stl(
                        series, obs_idx, mis_idx, period)
                    n_fallback += 1
                    try:
                        del model, result
                    except NameError:
                        pass
                    gc.collect()

                except Exception:
                    signal.alarm(0)
                    initial_recon[i, :, c] = _kalman_fallback_stl(
                        series, obs_idx, mis_idx, period)
                    n_fallback += 1
                    try:
                        del model, result
                    except NameError:
                        pass
                    gc.collect()

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_alarm_handler)

    initial_recon = np.clip(initial_recon, 0, 1)

    elapsed_total = time.time() - start_time
    obs_pct = obs_masks.sum() / obs_masks.size * 100
    print(f"Kalman initialization complete in {elapsed_total:.0f}s. "
          f"{N} sequences, {obs_pct:.1f}% observed, period={period}")
    print(f"  Kalman OK: {n_kalman_ok}, Fallback (STL): {n_fallback}, "
          f"Timeouts: {n_timeout}, Budget fallback: {n_budget_fallback}")
    return initial_recon


# =============================================================================
# Experiment 3b: Kalman Smoother for continuous-time irregular timestamps
# =============================================================================


def initialize_with_kalman_continuous(y_obs, timestamps, seq_len,
                                       period=None, seed=None,
                                       per_fit_timeout=None, max_seconds=None):
    """
    Kalman smoother init for continuous-time irregular time series.

    Used by run_co_evolving_em_continuous.py.  For each sample:

    1. Linear-interpolate (timestamps, y_obs) onto the regular grid
       [0, seq_len-1] to produce a dense pseudo-observation sequence.
    2. Run a local-linear-trend + seasonal UnobservedComponents model on that
       dense sequence; the smoothed level + seasonal is the structural init.
    3. On any Kalman failure (timeout / numerical) fall back to the
       piecewise-linear interpolant alone (Section 5.1 warm start).

    The difference from :func:`initialize_with_kalman` is that the input is
    not a grid-aligned series with NaNs at unobserved positions -- it is a
    pair (y_obs, timestamps) with timestamps in [0, seq_len - 1] floating
    point.  We pre-project onto the grid via numpy.interp so the downstream
    Kalman path is identical to the missing-data version.

    Args:
        y_obs:       (N, N_obs, C) float array of observations.
        timestamps:  (N, N_obs) float array of sample times in [0, seq_len-1].
        seq_len:     output grid length T.
        period:      seasonal period (auto-detected if None).
        seed:        random seed (used for the tiny residual jitter).
        per_fit_timeout: seconds per model.fit() call (default 5).
        max_seconds: global time budget in seconds (default 1800).

    Returns:
        initial_recon: (N, seq_len, C) float32, clipped to [0, 1].
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    if seed is not None:
        np.random.seed(seed)

    if y_obs.ndim != 3 or timestamps.ndim != 2:
        raise ValueError(
            f"initialize_with_kalman_continuous expects y_obs [N, N_obs, C] and "
            f"timestamps [N, N_obs]; got y_obs {y_obs.shape}, "
            f"timestamps {timestamps.shape}"
        )

    N, N_obs, C = y_obs.shape
    T = int(seq_len)
    if period is None:
        period = auto_detect_period(T)

    grid = np.arange(T, dtype=np.float64)

    # Step 1: dense linear-interpolation init.  This is the fallback when
    # Kalman fails and the starting point for the Kalman smoother.
    x_interp = np.zeros((N, T, C), dtype=np.float64)
    for i in range(N):
        order = np.argsort(timestamps[i])
        t_sorted = timestamps[i, order].astype(np.float64)
        y_sorted = y_obs[i, order, :].astype(np.float64)
        if t_sorted.size > 1:
            keep = np.concatenate([[True], np.diff(t_sorted) > 0])
            t_sorted = t_sorted[keep]
            y_sorted = y_sorted[keep, :]
        if t_sorted.size == 0:
            x_interp[i, :, :] = 0.5
            continue
        for c in range(C):
            x_interp[i, :, c] = np.interp(grid, t_sorted, y_sorted[:, c])

    initial_recon = np.clip(x_interp, 0.0, 1.0).astype(np.float32)

    MAX_KALMAN_SECONDS = max_seconds if max_seconds is not None else 1800
    PER_FIT_TIMEOUT = per_fit_timeout if per_fit_timeout is not None else 5
    start_time = time.time()
    budget_exceeded = False
    n_kalman_ok = 0
    n_fallback = 0
    n_timeout = 0

    prev_alarm_handler = signal.signal(signal.SIGALRM, _kalman_timeout_handler)

    try:
        for i in range(N):
            if (i + 1) % 200 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"  Kalman continuous init: {i+1}/{N} sequences "
                      f"({elapsed:.0f}s, {rate:.1f} seq/s, "
                      f"kalman={n_kalman_ok}, fallback={n_fallback}, "
                      f"timeout={n_timeout})")

            if not budget_exceeded and time.time() - start_time > MAX_KALMAN_SECONDS:
                budget_exceeded = True
                remaining = N - i
                print(f"  Global time budget ({MAX_KALMAN_SECONDS}s) exceeded "
                      f"at sequence {i}/{N}. Falling back to linear interp "
                      f"init for remaining {remaining} sequences.")

            if budget_exceeded:
                n_fallback += 1
                continue

            for c in range(C):
                series = x_interp[i, :, c].astype(np.float64)
                if not np.isfinite(series).all():
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = UnobservedComponents(
                            series, level='local linear trend',
                            seasonal=period, stochastic_seasonal=True,
                        )
                        signal.alarm(PER_FIT_TIMEOUT)
                        result = model.fit(disp=False, maxiter=30)
                        signal.alarm(0)

                    smoothed_state = result.smoothed_state
                    smoothed = smoothed_state[0, :]
                    if model.k_states > 2:
                        smoothed = smoothed + smoothed_state[2, :]

                    if np.isnan(smoothed).any():
                        raise ValueError("Kalman smoother produced NaN")

                    # Add small residual noise (matched to std of observation
                    # residuals) to avoid an over-smoothed init collapsing the
                    # denoiser to near-constant sequences.
                    residuals = series - smoothed
                    residuals = residuals[np.isfinite(residuals)]
                    if residuals.size > 1:
                        std_r = max(np.std(residuals), 1e-4)
                        smoothed = smoothed + np.random.normal(
                            0, std_r * 0.5, size=T,
                        )

                    initial_recon[i, :, c] = smoothed.astype(np.float32)
                    n_kalman_ok += 1

                    del model, result, smoothed_state, smoothed
                    gc.collect()

                except TimeoutError:
                    signal.alarm(0)
                    n_timeout += 1
                    n_fallback += 1
                    try:
                        del model, result
                    except NameError:
                        pass
                    gc.collect()

                except Exception:
                    signal.alarm(0)
                    n_fallback += 1
                    try:
                        del model, result
                    except NameError:
                        pass
                    gc.collect()

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_alarm_handler)

    initial_recon = np.clip(initial_recon, 0.0, 1.0)

    elapsed_total = time.time() - start_time
    avg_obs_per_sample = float(y_obs.shape[1])
    print(f"Kalman continuous init complete in {elapsed_total:.0f}s. "
          f"{N} sequences, {avg_obs_per_sample:.1f} obs/sample on a grid of T={T}, "
          f"period={period}")
    print(f"  Kalman OK (channel-level): {n_kalman_ok}, "
          f"Fallback: {n_fallback}, Timeouts: {n_timeout}")
    return initial_recon


# =============================================================================
# Experiment 4: Seasonal-Aware Interpolation (Deseasonalize-Interpolate-Reseasonalize)
# =============================================================================

def initialize_with_seasonal_interp(corrupted_data, obs_masks, period=None, seed=None):
    """
    Seasonal-aware interpolation: deseasonalize, interpolate, reseasonalize.

    Seasonality is preserved by construction because it is never interpolated
    through. The deseasonalized signal is smooth, making interpolation accurate
    even with simple methods.

    Inspired by imputeTS::na_seadec (R package).

    Strategy:
      1. Estimate seasonal pattern from observed data only (periodic averaging)
      2. Deseasonalize observed points
      3. PCHIP-interpolate the smooth deseasonalized signal
      4. Reseasonalize: filled = deseasonalized_interp + seasonal
      5. STL decompose for residual resampling at missing positions

    Args:
        corrupted_data: (N, T, C) numpy array with NaN for missing
        obs_masks:      (N, T) boolean, True = observed
        period:         seasonal period (auto-detected if None)
        seed:           random seed

    Returns:
        initial_recon: (N, T, C) with all values filled
    """
    if seed is not None:
        np.random.seed(seed)

    N, T, C = corrupted_data.shape
    if period is None:
        period = auto_detect_period(T)

    initial_recon = corrupted_data.copy()

    for i in range(N):
        mask_i = obs_masks[i]
        obs_idx = np.where(mask_i)[0]
        mis_idx = np.where(~mask_i)[0]

        if len(obs_idx) < 3 or len(mis_idx) == 0:
            for c in range(C):
                obs_vals = corrupted_data[i, obs_idx, c] if len(obs_idx) > 0 else np.array([0.5])
                mu, sigma = np.nanmean(obs_vals), max(np.nanstd(obs_vals), 1e-3)
                initial_recon[i, mis_idx, c] = np.random.normal(mu, sigma, size=len(mis_idx))
            continue

        for c in range(C):
            series = corrupted_data[i, :, c].copy()
            obs_vals = series[obs_idx]

            # Step 1: estimate seasonal pattern from observed data only
            seasonal = np.zeros(T)
            global_mean = np.nanmean(obs_vals)
            for k in range(period):
                phase_positions = np.arange(k, T, period)
                phase_obs = [t for t in phase_positions if mask_i[t]]
                if len(phase_obs) >= 1:
                    seasonal[phase_positions] = np.mean(series[phase_obs]) - global_mean
                else:
                    seasonal[phase_positions] = 0.0

            # Step 2: deseasonalize observed points
            deseason = np.full(T, np.nan)
            deseason[obs_idx] = obs_vals - seasonal[obs_idx]

            # Step 3: PCHIP-interpolate the deseasonalized signal
            deseason_obs = deseason[obs_idx]
            if len(obs_idx) >= 2:
                pchip = PchipInterpolator(obs_idx, deseason_obs)
                deseason_filled = pchip(np.arange(T))
            else:
                deseason_filled = np.full(T, deseason_obs[0] if len(deseason_obs) > 0 else 0.5)

            # Step 4: reseasonalize
            filled = deseason_filled + seasonal

            # Step 5: STL decompose for residual resampling
            trend_c, season_c, resid_c = stl_decompose_single(filled, period)
            resid_obs = resid_c[obs_idx]
            mu_r, std_r = resid_obs.mean(), max(resid_obs.std(), 1e-4)
            resid_c[mis_idx] = np.random.normal(mu_r, std_r, size=len(mis_idx))

            initial_recon[i, :, c] = trend_c + season_c + resid_c

    initial_recon = np.clip(initial_recon, 0, 1)

    obs_pct = obs_masks.sum() / obs_masks.size * 100
    print(f"Seasonal-aware interpolation init complete. {N} sequences, "
          f"{obs_pct:.1f}% observed, period={period}")
    return initial_recon


# =============================================================================
# Experiment 5: PCHIP Interpolation + STL
# =============================================================================

def initialize_with_pchip_stl(corrupted_data, obs_masks, period=None, seed=None):
    """
    PCHIP + STL initialization — drop-in replacement of linear interpolation
    with shape-preserving cubic Hermite interpolation.

    PCHIP preserves monotonicity between observations and captures curvature,
    producing a better pre-fill for STL decomposition than linear interpolation.

    Args:
        corrupted_data: (N, T, C) numpy array with NaN for missing
        obs_masks:      (N, T) boolean, True = observed
        period:         STL period (auto-detected if None)
        seed:           random seed

    Returns:
        initial_recon: (N, T, C) with all values filled
    """
    if seed is not None:
        np.random.seed(seed)

    N, T, C = corrupted_data.shape
    if period is None:
        period = auto_detect_period(T)

    initial_recon = corrupted_data.copy()

    for i in range(N):
        mask_i = obs_masks[i]
        obs_idx = np.where(mask_i)[0]
        mis_idx = np.where(~mask_i)[0]

        if len(obs_idx) < 3 or len(mis_idx) == 0:
            for c in range(C):
                obs_vals = corrupted_data[i, obs_idx, c] if len(obs_idx) > 0 else np.array([0.5])
                mu, sigma = np.nanmean(obs_vals), max(np.nanstd(obs_vals), 1e-3)
                initial_recon[i, mis_idx, c] = np.random.normal(mu, sigma, size=len(mis_idx))
            continue

        for c in range(C):
            series = corrupted_data[i, :, c].copy()
            obs_vals = series[obs_idx]

            # PCHIP interpolation (shape-preserving cubic Hermite)
            pchip = PchipInterpolator(obs_idx, obs_vals)
            series[mis_idx] = pchip(mis_idx)

            # STL decompose the PCHIP-interpolated series
            trend_c, season_c, resid_c = stl_decompose_single(series, period)

            # Resample residual at missing positions
            resid_obs = resid_c[obs_idx]
            mu_r, std_r = resid_obs.mean(), max(resid_obs.std(), 1e-4)
            resid_c[mis_idx] = np.random.normal(mu_r, std_r, size=len(mis_idx))

            initial_recon[i, :, c] = trend_c + season_c + resid_c

    initial_recon = np.clip(initial_recon, 0, 1)

    obs_pct = obs_masks.sum() / obs_masks.size * 100
    print(f"PCHIP+STL initialization complete. {N} sequences, {obs_pct:.1f}% observed, period={period}")
    return initial_recon


# =============================================================================
# Experiment 6: LOWESS Interpolation + STL
# =============================================================================

def initialize_with_lowess_stl(corrupted_data, obs_masks, period=None,
                                frac=0.3, seed=None):
    """
    LOWESS + STL initialization — uses locally weighted scatterplot smoothing
    to fit a smooth curve on observed points, then evaluates at all positions.

    LOWESS captures non-linear local structure better than linear interpolation.
    The smoothed curve is then decomposed via STL for residual resampling.

    Args:
        corrupted_data: (N, T, C) numpy array with NaN for missing
        obs_masks:      (N, T) boolean, True = observed
        period:         STL period (auto-detected if None)
        frac:           LOWESS bandwidth (fraction of data used, default 0.3)
        seed:           random seed

    Returns:
        initial_recon: (N, T, C) with all values filled
    """
    if seed is not None:
        np.random.seed(seed)

    N, T, C = corrupted_data.shape
    if period is None:
        period = auto_detect_period(T)

    initial_recon = corrupted_data.copy()

    for i in range(N):
        mask_i = obs_masks[i]
        obs_idx = np.where(mask_i)[0]
        mis_idx = np.where(~mask_i)[0]

        if len(obs_idx) < 3 or len(mis_idx) == 0:
            for c in range(C):
                obs_vals = corrupted_data[i, obs_idx, c] if len(obs_idx) > 0 else np.array([0.5])
                mu, sigma = np.nanmean(obs_vals), max(np.nanstd(obs_vals), 1e-3)
                initial_recon[i, mis_idx, c] = np.random.normal(mu, sigma, size=len(mis_idx))
            continue

        for c in range(C):
            series = corrupted_data[i, :, c].copy()
            obs_vals = series[obs_idx]

            # Fit LOWESS on observed points
            # lowess returns (n, 2) array sorted by x: [[x0, y0], [x1, y1], ...]
            lowess_result = lowess(
                obs_vals, obs_idx.astype(float),
                frac=max(frac, 3.0 / len(obs_idx)),  # ensure >= 3 points in window
                return_sorted=True,
            )
            lowess_x = lowess_result[:, 0]
            lowess_y = lowess_result[:, 1]

            # Interpolate the smooth LOWESS curve at missing positions
            series[mis_idx] = np.interp(mis_idx, lowess_x, lowess_y)

            # STL decompose the LOWESS-filled series
            trend_c, season_c, resid_c = stl_decompose_single(series, period)

            # Resample residual at missing positions
            resid_obs = resid_c[obs_idx]
            mu_r, std_r = resid_obs.mean(), max(resid_obs.std(), 1e-4)
            resid_c[mis_idx] = np.random.normal(mu_r, std_r, size=len(mis_idx))

            initial_recon[i, :, c] = trend_c + season_c + resid_c

    initial_recon = np.clip(initial_recon, 0, 1)

    obs_pct = obs_masks.sum() / obs_masks.size * 100
    print(f"LOWESS+STL initialization complete. {N} sequences, "
          f"{obs_pct:.1f}% observed, frac={frac}, period={period}")
    return initial_recon
