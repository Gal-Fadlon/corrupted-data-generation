"""Helpers for Ambient Diffusion–style UNet inputs (mask concatenation)."""

import torch


def concat_mask_channel(
    x_data: torch.Tensor,
    mask_1ch: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """Append a single-channel mask after image channels (EDM / Daras et al. setup)."""
    if not enabled:
        return x_data
    m = mask_1ch
    if m.dim() == 3:
        m = m.unsqueeze(1)
    m = m[:, :1].to(device=x_data.device, dtype=x_data.dtype)
    return torch.cat([x_data, m], dim=1)


def concat_ones_mask(x_data: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Append an all-ones mask (full visibility) for unconditional sampling."""
    if not enabled:
        return x_data
    ones = torch.ones(
        x_data.shape[0],
        1,
        x_data.shape[2],
        x_data.shape[3],
        device=x_data.device,
        dtype=x_data.dtype,
    )
    return torch.cat([x_data, ones], dim=1)
