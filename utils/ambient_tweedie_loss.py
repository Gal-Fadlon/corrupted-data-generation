"""
Prototype: Ambient Denoising Score Matching (Variance-Exploding) for noisy-only training.

Reference: "Consistent Diffusion Meets Tweedie" (ICML 2024), Theorem (Ambient DSM).
For sigma_t > sigma_n, with x_{t_n} = x_0 + sigma_n * z and
x_t = x_{t_n} + sqrt(sigma_t^2 - sigma_n^2) * eta, the optimal denoiser h
minimizes the squared error in Eq. (ambient DSM) in the paper.

This module adds an *auxiliary* loss on top of standard Karras EDM training so
DiffEM's M-step can encourage consistency with the Ambient Tweedie objective
when training data are reconstructions from noisy observations (sigma_n set to
corruption_noise_level in the same units as diffusion sigmas — tune if needed).

Not a full replacement for the paper's training pipeline; intended for experiments.
"""

import torch
import torch.nn.functional as F

from utils.ambient_net_input import concat_ones_mask


def ambient_ve_dsm_residual(net, x_img, sigma_n, P_mean=-1.2, P_std=1.2,
                            ambient_concat_further_mask=False, num_features=None):
    """
    Single-batch Ambient DSM term: || w1 * h(x_t,t) + w2 * x_t - x_{t_n} ||^2
    with weights from the paper (VE parameterization).

    Args:
        net: denoiser D_theta
        x_img: (B, C, H, W) clean *reconstructions* (M-step targets)
        sigma_n: scalar, minimum noise (nature); same dtype/device as sigmas
        P_mean, P_std: log-sigma prior (match TS2img_Karras defaults)
        ambient_concat_further_mask: pass-through to concat_ones_mask
        num_features: channels for mask strip (unused if concat uses ones only)

    Returns:
        scalar loss
    """
    device = x_img.device
    dtype = x_img.dtype
    b = x_img.shape[0]
    rnd = torch.randn([b, 1, 1, 1], device=device, dtype=dtype)
    sigma_t = (rnd * P_std + P_mean).exp()
    sn = torch.as_tensor(sigma_n, device=device, dtype=dtype).view(1, 1, 1, 1)
    sn = sn.expand(b, 1, 1, 1)
    sigma_t = torch.maximum(sigma_t, sn * 1.05 + 1e-4)

    z0 = torch.randn_like(x_img)
    z1 = torch.randn_like(x_img)
    x_tn = x_img + sn * z0
    x_t = x_tn + torch.sqrt(torch.clamp(sigma_t ** 2 - sn ** 2, min=0)) * z1

    x_in = concat_ones_mask(x_t, ambient_concat_further_mask)
    h = net(x_in, sigma_t, None)
    if ambient_concat_further_mask and num_features is not None:
        h = h[:, :num_features, :, :]

    w1 = (sigma_t ** 2 - sn ** 2) / (sigma_t ** 2)
    w2 = sn ** 2 / (sigma_t ** 2)
    target_combo = w1 * h + w2 * x_t
    return F.mse_loss(target_combo, x_tn)


def consistency_denoiser_stub(net, x_img, sigma_a, sigma_b, P_mean=-1.2, P_std=1.2,
                              ambient_concat_further_mask=False, num_features=None):
    """
    Lightweight consistency regularizer: |h(x',t') - stopgrad(h(x'',t''))|^2
    with one-step bridge (prototype; full version samples from p_theta).
    """
    device = x_img.device
    dtype = x_img.dtype
    b = x_img.shape[0]
    rnd = torch.randn([b, 1, 1, 1], device=device, dtype=dtype)
    sigma_mid = (rnd * P_std + P_mean).exp()
    sigma_mid = torch.clamp(sigma_mid, min=sigma_a, max=sigma_b)

    z = torch.randn_like(x_img)
    x_t = x_img + sigma_mid * z
    x_in = concat_ones_mask(x_t, ambient_concat_further_mask)
    h1 = net(x_in, sigma_mid, None)
    if ambient_concat_further_mask and num_features is not None:
        h1 = h1[:, :num_features, :, :]

    eps = 0.02
    sigma_lo = torch.clamp(sigma_mid - eps, min=0.002)
    x_t2 = x_img + sigma_lo * torch.randn_like(x_img)
    x_in2 = concat_ones_mask(x_t2, ambient_concat_further_mask)
    h2 = net(x_in2, sigma_lo, None)
    if ambient_concat_further_mask and num_features is not None:
        h2 = h2[:, :num_features, :, :]

    return F.mse_loss(h1, h2.detach())
