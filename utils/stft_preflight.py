"""Pre-flight audit for the STFT embedder.

Before burning GPU-hours on a co-evolving EM run that depends on the lift
being a well-behaved linear operator with an exact autograd adjoint, we
sanity-check three invariants:

  1. Linearity:             ts_to_img(a*x + b*y) == a*ts_to_img(x) + b*ts_to_img(y)
  2. Roundtrip:              img_to_ts(ts_to_img(x)) == x (up to COLA edge effects)
  3. Adjoint(ts_to_img):     <ts_to_img(x), y>_img == <x, ts_to_img_adjoint(y)>_ts
  4. Adjoint(img_to_ts):     <img_to_ts(I), v>_ts  == <I, img_to_ts_adjoint(v)>_img

Failing (1), (3), or (4) means observation-space CG operates on a wrongly-
scaled (though still SPD) system and converges to a biased posterior mean.
Failing (2) means the lift has a non-trivial nullspace at this resolution
and the paper's "near-isometry" claim needs to be softened.

The gate is a no-op for the delay embedder (which has no adjoint/padding
issues and a simpler verification story).
"""
from __future__ import annotations

import logging
import torch


def _rand_ts(batch, seq_len, feat, device, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(batch, seq_len, feat, generator=g).to(device)


def audit_stft_embedder(
    embedder,
    feat_dim,
    device,
    tol_linearity=5e-5,
    tol_roundtrip=1e-2,
    tol_adjoint=1e-3,
    batch=4,
    train_data=None,
    train_pixel_std_band=(0.3, 0.8),
    hard_gate_train_pixel_std=False,
):
    # Note on tolerances under scale_mode='zscore':
    #   Under zscore scaling the smallest per-bin scale can be as tight as
    #   ~1e-5 at long sequences (L>=768), and the division amplifies float32
    #   STFT roundoff. At L=24 the image pixel std on a random probe is ~50,
    #   at L=768 it is ~1200, so absolute linearity errors scale with the
    #   image norm. We therefore compare the linearity error **relative** to
    #   the RHS max-abs (post-scaling), keeping the threshold tight at 1e-4
    #   while being scale-invariant.
    """Run the three gates against a live STFTEmbedder with cached scales.

    Args:
        embedder: STFTEmbedder instance with scale_real / scale_imag already set.
        feat_dim: number of TS features (channels).
        device: torch device.
        tol_linearity: max per-element abs error for linearity check.
        tol_roundtrip: max relative error for ts -> img -> ts roundtrip.
        tol_adjoint: max relative deviation of <Lx, y> / <x, L^T y> from 1.
        batch: batch size for the probes.

    Returns:
        dict with the three numeric metrics for logging.

    Raises:
        RuntimeError if any gate fails.
    """
    if embedder.scale_real is None:
        raise RuntimeError(
            "STFT preflight: embedder.scale_real is None; call "
            "cache_embedder_stats(train_tensor) first."
        )

    seq_len = embedder.seq_len
    x = _rand_ts(batch, seq_len, feat_dim, device, seed=0)
    y = _rand_ts(batch, seq_len, feat_dim, device, seed=1)

    with torch.no_grad():
        lhs = embedder.ts_to_img(2.5 * x + (-0.7) * y)
        rhs = 2.5 * embedder.ts_to_img(x) + (-0.7) * embedder.ts_to_img(y)
        # Relative linearity error: absolute error divided by the RHS scale.
        # Pure-absolute would spuriously fail at long sequences where per-bin
        # zscore amplifies float32 roundoff into the image-pixel-std range.
        lin_abs = (lhs - rhs).abs().max().item()
        lin_ref = rhs.abs().max().item()
        lin_err = lin_abs / max(lin_ref, 1e-12)

    with torch.no_grad():
        img_x = embedder.ts_to_img(x)
        x_rt = embedder.img_to_ts(img_x)
        rt_err = ((x_rt - x).norm() / x.norm().clamp_min(1e-8)).item()

    img_probe = embedder.ts_to_img(x)
    with torch.no_grad():
        Ltx = embedder.ts_to_img_adjoint(img_probe)
        lhs_inner = (embedder.ts_to_img(x) * img_probe).sum().item()
        rhs_inner = (x * Ltx).sum().item()
    denom = max(abs(lhs_inner), abs(rhs_inner), 1e-12)
    adj_err = abs(lhs_inner - rhs_inner) / denom

    # (4) Adjoint identity for img_to_ts:
    #     <img_to_ts(I), v>_ts == <I, img_to_ts_adjoint(v)>_img
    # This is the adjoint consumed by obs-space CG (G^T = img_to_ts_adjoint
    # pre-masked), and was previously silently approximated as ts_to_img
    # (= L) which is only correct for unitary lifts.
    img_probe2 = embedder.ts_to_img(x)
    v_probe = _rand_ts(batch, seq_len, feat_dim, device, seed=2)
    with torch.no_grad():
        lhs_inner2 = (embedder.img_to_ts(img_probe2) * v_probe).sum().item()
        iLv = embedder.img_to_ts_adjoint(v_probe)
        rhs_inner2 = (img_probe2 * iLv).sum().item()
    denom2 = max(abs(lhs_inner2), abs(rhs_inner2), 1e-12)
    inv_adj_err = abs(lhs_inner2 - rhs_inner2) / denom2

    # Advisory: image pixel std on the random probe `x` (~ N(0, 1) in TS
    # space). This is NOT directly comparable to sigma_data because we
    # divide by per-bin scales cached on TRAINING data, not on `x`. Under
    # zscore scaling it will typically read >> 1 because random noise has
    # relatively uniform per-bin energy while training data does not. We
    # log it for inspection but do not gate on it. For the true
    # "image pixel std on training data" check, look at the runtime
    # `stft_scale/*` metrics during M-step warm-up.
    with torch.no_grad():
        img_std = embedder.ts_to_img(x).std().item()

    # Fix 2 gate: measure pixel std on REAL training data (not a random probe)
    # and hard-gate that it lies in [0.3, 0.8] so EDM's sigma_data=0.5
    # preconditioning is in-spec. Only active when training data is provided.
    train_pixel_std = None
    train_pixel_mean = None
    if train_data is not None:
        td = train_data if torch.is_tensor(train_data) else torch.as_tensor(train_data)
        td = td.to(device=device, dtype=torch.float32)
        if td.ndim != 3:
            raise ValueError(
                f'stft_preflight: expected train_data of shape (B,T,F), got {td.shape}'
            )
        probe_n = min(td.shape[0], 256)
        with torch.no_grad():
            img_train = embedder.ts_to_img(td[:probe_n])
            train_pixel_std = float(img_train.std().item())
            train_pixel_mean = float(img_train.mean().item())

    metrics = {
        'linearity_rel_err': lin_err,
        'linearity_abs_err': lin_abs,
        'linearity_ref_scale': lin_ref,
        'roundtrip_rel_err': rt_err,
        'adjoint_rel_err': adj_err,
        'adjoint_inner_product_ratio': (
            lhs_inner / rhs_inner if abs(rhs_inner) > 1e-12 else float('inf')
        ),
        'inverse_adjoint_rel_err': inv_adj_err,
        'inverse_adjoint_inner_product_ratio': (
            lhs_inner2 / rhs_inner2 if abs(rhs_inner2) > 1e-12 else float('inf')
        ),
        'image_pixel_std': img_std,
    }
    if train_pixel_std is not None:
        metrics['train_image_pixel_std'] = train_pixel_std
        metrics['train_image_pixel_mean'] = train_pixel_mean

    msgs = [
        f"STFT preflight: linearity_rel={lin_err:.3e} (tol={tol_linearity:.0e}; "
        f"abs={lin_abs:.3e} ref={lin_ref:.3e})",
        f"STFT preflight: roundtrip={rt_err:.3e} (tol={tol_roundtrip:.0e})",
        f"STFT preflight: adjoint_rel_err={adj_err:.3e} (tol={tol_adjoint:.0e}) "
        f"ratio={metrics['adjoint_inner_product_ratio']:.6f}",
        f"STFT preflight: inverse_adjoint_rel_err={inv_adj_err:.3e} "
        f"(tol={tol_adjoint:.0e}) "
        f"ratio={metrics['inverse_adjoint_inner_product_ratio']:.6f}",
        f"STFT preflight: image_pixel_std={img_std:.4f} "
        f"(random probe; not directly comparable to sigma_data=0.5)",
    ]
    if train_pixel_std is not None:
        band_lo, band_hi = train_pixel_std_band
        msgs.append(
            f"STFT preflight: train_image_pixel_std={train_pixel_std:.4f} "
            f"(target band=[{band_lo:.2f}, {band_hi:.2f}] around sigma_data=0.5; "
            f"hard_gate={hard_gate_train_pixel_std})"
        )
    for m in msgs:
        logging.info(m)
        print(m)

    problems = []
    if lin_err > tol_linearity:
        problems.append(
            f"linearity {lin_err:.3e} > {tol_linearity:.0e}"
        )
    if rt_err > tol_roundtrip:
        problems.append(
            f"roundtrip {rt_err:.3e} > {tol_roundtrip:.0e}"
        )
    if adj_err > tol_adjoint:
        problems.append(
            f"adjoint {adj_err:.3e} > {tol_adjoint:.0e}"
        )
    if inv_adj_err > tol_adjoint:
        problems.append(
            f"inverse_adjoint {inv_adj_err:.3e} > {tol_adjoint:.0e}"
        )
    if (hard_gate_train_pixel_std
            and train_pixel_std is not None):
        band_lo, band_hi = train_pixel_std_band
        if not (band_lo <= train_pixel_std <= band_hi):
            problems.append(
                f"train_image_pixel_std {train_pixel_std:.4f} outside "
                f"[{band_lo:.2f}, {band_hi:.2f}] - EDM sigma_data=0.5 mismatch "
                f"will collapse the denoiser. Enable --stft_global_rescale."
            )

    if problems:
        raise RuntimeError(
            "STFT preflight failed: " + "; ".join(problems)
            + ". Inspect models/img_transformations.py::STFTEmbedder before "
            "launching the EM run."
        )

    return metrics


def run(uncond_model, feat_dim, device, logger=None,
        train_data=None, hard_gate_train_pixel_std=False,
        train_pixel_std_band=(0.3, 0.8)):
    """Convenience entry point: no-op for non-STFT embedders.

    Args:
        uncond_model: TS2img_Karras (has .ts_img, .embedder_kind).
        feat_dim: number of TS features.
        device: torch device.
        logger: optional logger with .log(key, value, step).
        train_data: optional (B, T, F) tensor; when provided, preflight
            additionally measures the image pixel std on real training data
            and (if hard_gate_train_pixel_std) gates on it being within
            `train_pixel_std_band` around EDM's sigma_data=0.5.
        hard_gate_train_pixel_std: see above.
        train_pixel_std_band: (lo, hi) band for the Fix 2 gate.

    Returns:
        metrics dict (or None if not STFT).
    """
    if getattr(uncond_model, 'embedder_kind', 'delay') != 'stft':
        return None

    metrics = audit_stft_embedder(
        uncond_model.ts_img, feat_dim=feat_dim, device=device,
        train_data=train_data,
        hard_gate_train_pixel_std=hard_gate_train_pixel_std,
        train_pixel_std_band=train_pixel_std_band,
    )
    if logger is not None:
        for k, v in metrics.items():
            try:
                logger.log(f'stft_preflight/{k}', float(v), 0)
            except Exception:
                pass
    return metrics
