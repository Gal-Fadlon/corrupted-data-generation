"""
Shared frequency gating + TS decomposition projection for MMPS (A = I Gaussian case).

Mirrors run_diffem_mmps_kalman_freqgate_decomp_proj.py utilities without importing run scripts.
"""

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from utils.utils_stl import auto_detect_period


def decompose_and_smooth_ts(ts_batch, period=None, trend_smooth_window=None,
                            residual_keep_fraction=0.8):
    """
    Vectorized decomposition + smoothing for a batch of time series.

    Args:
        ts_batch: (B, T, C) numpy array — no NaN
        period: decomposition period (auto-detected if None)
        trend_smooth_window: additional smoothing window for trend.
            None = auto-scale with T (max(5, T // 50)).
        residual_keep_fraction: fraction of residual energy to keep (0-1).

    Returns:
        projected: (B, T, C) numpy array
    """
    b, t, c = ts_batch.shape
    if period is None:
        period = auto_detect_period(t)
    if trend_smooth_window is None:
        trend_smooth_window = max(5, t // 50)

    window = min(period, max(3, t // 3))
    if window % 2 == 0:
        window += 1

    trends = uniform_filter1d(ts_batch, size=window, axis=1, mode='nearest')

    if trend_smooth_window > 1:
        sw = (trend_smooth_window if trend_smooth_window % 2 == 1
              else trend_smooth_window + 1)
        trends = uniform_filter1d(trends, size=sw, axis=1, mode='nearest')

    detrended = ts_batch - trends
    seasonals = np.zeros_like(ts_batch)
    for k in range(period):
        idx = np.arange(k, t, period)
        phase_mean = detrended[:, idx, :].mean(axis=1, keepdims=True)
        seasonals[:, idx, :] = phase_mean

    residuals = ts_batch - trends - seasonals
    return trends + seasonals + residual_keep_fraction * residuals


def lowpass_filter_image(x, cutoff_fraction):
    """Low-pass filter an image tensor via FFT truncation. x: (B, C, H, W)."""
    if cutoff_fraction >= 1.0:
        return x
    if cutoff_fraction <= 0.0:
        return torch.zeros_like(x)

    fft = torch.fft.rfft2(x.float())
    h, w_half = fft.shape[-2], fft.shape[-1]

    freq_h = torch.arange(h, device=x.device).float()
    freq_h = torch.min(freq_h, h - freq_h) / (h // 2)
    freq_w = torch.arange(w_half, device=x.device).float() / (w_half - 1)

    freq_grid = torch.sqrt(freq_h[:, None] ** 2 + freq_w[None, :] ** 2)
    mask = (freq_grid <= cutoff_fraction).float()

    filtered_fft = fft * mask[None, None, :, :]
    return torch.fft.irfft2(filtered_fft, s=x.shape[-2:]).to(x.dtype)


def compute_frequency_gate(sigma, sigma_max, gate_schedule='linear'):
    """Fraction of frequency content to keep in guidance (higher sigma -> lower cutoff)."""
    ratio = float(sigma) / float(sigma_max)
    ratio = max(0.0, min(1.0, ratio))

    if gate_schedule == 'cosine':
        cutoff = 0.1 + 0.9 * (1.0 - np.cos(np.pi * (1.0 - ratio)) / 2.0)
    else:
        cutoff = 0.1 + 0.9 * (1.0 - ratio)

    return cutoff
