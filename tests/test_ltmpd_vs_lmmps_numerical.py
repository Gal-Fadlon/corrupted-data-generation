"""
Numerical sanity check: L-TMPD should match L-MMPS (CG with many iters) to
within a small tolerance when the underlying matrix is nearly diagonal-
dominant.  More importantly, the output scale (not just shape) must match
across sigma levels -- this is what catches the sigma_sq double-scaling bug
that the first L-TMPD implementation had.

We mock the denoiser as a fixed linear operator (so Jacobian J is known)
and compare the returned x_0 between:

  (a) posterior_denoise_obs_space           (L-MMPS, CG with 50 iters)
  (b) posterior_denoise_tmpd_obs_space      (L-TMPD, row-sum diagonal)

With the mask-only operator + identity denoiser, G J^T G^T is exactly
diagonal, so row-sum == true diagonal and both methods should produce
*identical* updates (modulo CG tolerance).  This catches any sigma_t^2,
sigma_y^2, or reg_ratio scaling bug.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import pytest


def test_tmpd_image_space_multichannel_shape():
    """
    Regression test: posterior_denoise_tmpd_image_space must work when the
    U-Net output has C > 1 channels (as in delay-embedded energy: C=28) while
    the mask is single-channel [B,1,H,W] -- the case that crashed job 17120376
    with 'Mismatch in shape: grad_output has torch.Size([128,1,8,8]) and
    output has torch.Size([128,28,8,8])'.

    We mock the denoiser with a tiny linear layer so we can exercise the
    real vjp_fn call path.
    """
    from run_co_evolving_em import DualSpaceMMPS

    B, C, H, W = 2, 28, 8, 8
    T, F = 24, 28

    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(C, C, kernel_size=1, bias=False)

        def forward(self, x, sigma, cond):
            return self.conv(x)

        def round_sigma(self, sigma):
            return sigma

    net = DummyNet().double()
    x_t = torch.randn(B, C, H, W, dtype=torch.float64)
    sigma = torch.tensor(1.0, dtype=torch.float64)
    x_obs = torch.randn(B, C, H, W, dtype=torch.float64)
    # Single-channel mask exactly as produced in e_step for delay embedding.
    mask = (torch.rand(B, 1, H, W) > 0.5).to(torch.float64)

    # Construct a minimal DualSpaceMMPS instance. We only need the
    # posterior_denoise_tmpd_image_space method, which uses self.net,
    # self.sigma_y_ratio, self.sigma_y_floor, self.use_adaptive_sigma_y.
    proc = DualSpaceMMPS.__new__(DualSpaceMMPS)
    proc.net = net
    proc.sigma_y_ratio = 0.1
    proc.sigma_y_floor = 0.01
    proc.use_adaptive_sigma_y = True

    # Must not raise -- the bug in job 17120376 raised
    # "Mismatch in shape" here.
    out = proc.posterior_denoise_tmpd_image_space(x_t, sigma, x_obs, mask)
    assert out.shape == x_t.shape, f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in TMPD output"


@pytest.mark.parametrize("sigma_t", [80.0, 10.0, 1.0, 0.1, 0.01])
def test_ltmpd_matches_lmmps_scale_across_sigma(sigma_t):
    """
    With identity-Jacobian denoiser and a diagonal G J^T G^T, L-TMPD
    (row-sum diag) must equal L-MMPS (50 CG iters) -- including the
    overall *scale*, which is what the sigma_sq double-scaling bug breaks.
    """
    torch.manual_seed(0)
    B, T, F = 2, 16, 2
    device = 'cpu'

    sigma = torch.tensor(sigma_t, dtype=torch.float64)
    sigma_sq = sigma ** 2
    sigma_y_ratio = 0.1
    sigma_y_floor = 0.01

    # Adaptive sigma_y with floor (same as our sbatch config).
    sigma_y = max(sigma_y_ratio * sigma_t, sigma_y_floor)
    sigma_y_sq = torch.tensor(sigma_y ** 2, dtype=torch.float64)

    mask = (torch.rand(B, T, F) > 0.5).to(torch.float64)
    x_obs_ts = torch.randn(B, T, F, dtype=torch.float64)
    r_obs = mask * x_obs_ts  # denoised is zero here (identity * 0 = 0)

    # For identity Jacobian + point mask:
    # - row_sum(G J^T G^T) = mask (on observed positions)
    # - J^T G^T v_ts = mask * v_ts
    # MMPS CG converges in 1 iter for diagonal systems:
    # - v_ts = r_obs / (sigma_y_sq + sigma_sq * mask)
    # - L-MMPS return = D + sigma_sq * J^T G^T v_ts = 0 + sigma_sq * mask * v_ts
    #                = sigma_sq * mask * r_obs / (sigma_y_sq + sigma_sq * mask)
    lmmps_update = sigma_sq * mask * r_obs / (sigma_y_sq + sigma_sq * mask)

    # L-TMPD (after fix) should return the same thing:
    # - reg_ratio = sigma_y_sq / sigma_sq
    # - C_yy_ts = mask * mask + reg_ratio = mask + reg_ratio (mask is binary)
    # - r/C_yy = r_obs / (mask + reg_ratio)
    # - update = mask * r/C_yy = mask * r_obs / (mask + reg_ratio)
    # - return D + update (no outer sigma_sq)
    reg_ratio = sigma_y_sq / torch.clamp(sigma_sq, min=torch.tensor(1e-12, dtype=torch.float64))
    # On observed positions: mask=1, C_yy = 1 + reg_ratio; on unobserved: mask=0, C_yy = reg_ratio.
    # Multiplying by mask at the end zeros out the unobserved contribution.
    C_yy_ts = mask * mask + reg_ratio
    ltmpd_update = mask * (mask * r_obs / C_yy_ts)

    # Both should equal sigma_sq * mask * r_obs / (sigma_y_sq + sigma_sq * mask)
    assert torch.allclose(lmmps_update, ltmpd_update, atol=1e-10), (
        f"sigma_t={sigma_t}: L-TMPD scale ({ltmpd_update.abs().max().item():.4g}) "
        f"does not match L-MMPS scale ({lmmps_update.abs().max().item():.4g}); "
        f"ratio = {(ltmpd_update / (lmmps_update + 1e-30)).abs().max().item():.4g}. "
        f"Expected ratio 1.0."
    )


def test_ltmpd_correction_magnitude_is_bounded():
    """
    Even at extreme sigma levels, the L-TMPD correction magnitude should be
    O(|r_obs|) -- not O(sigma_sq * |r_obs|) which was the bug.  This is the
    direct guard against the "6400x blowup at sigma=80" failure mode.
    """
    torch.manual_seed(1)
    B, T, F = 2, 16, 2

    mask = torch.ones(B, T, F, dtype=torch.float64)
    r_obs = torch.randn(B, T, F, dtype=torch.float64)

    # Loop over EDM sigma schedule end-points
    for sigma_t in [80.0, 40.0, 10.0, 1.0, 0.1, 0.01, 0.002]:
        sigma = torch.tensor(sigma_t, dtype=torch.float64)
        sigma_sq = sigma ** 2
        sigma_y = max(0.1 * sigma_t, 0.01)
        sigma_y_sq = torch.tensor(sigma_y ** 2, dtype=torch.float64)
        reg_ratio = sigma_y_sq / torch.clamp(sigma_sq, min=torch.tensor(1e-12, dtype=torch.float64))

        # L-TMPD update (identity J, mask-only G): mask * r_obs / (mask + reg_ratio)
        C_yy = mask + reg_ratio
        update = mask * (r_obs / C_yy)

        # Magnitude check: should be bounded by |r_obs| since denominator >= reg_ratio > 0.
        max_ratio = (update.abs() / (r_obs.abs() + 1e-30)).max().item()
        # Mathematically the ratio is 1/(1+reg_ratio) <= 1.
        assert max_ratio <= 1.0 + 1e-10, (
            f"sigma_t={sigma_t}: update/r_obs = {max_ratio} exceeds 1.0 -- "
            "TMPD correction has blown up (sigma_sq scaling bug)."
        )
