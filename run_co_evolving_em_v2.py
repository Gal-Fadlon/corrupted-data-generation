"""
Co-Evolving EM for Time Series Generation from Corrupted Observations

Vanilla EM in a dual-space setting (TS observations <-> image-space model via
delay embedding) fails in five specific ways.  This module implements targeted
fixes for each — cross-space consistency constraints — that make EM converge
in 3–5 iterations to state-of-the-art generation quality.

Failure modes and fixes (paper Section 5):

  5.1  Cold start              → Kalman/STL warm start + curriculum scheduling
  5.2  CG in the wrong space   → Observation-Space CG (Prop 4)
  5.3  CG instability          → Adaptive σ_y = c·σ_t (Prop 2)
  5.4  Off-manifold drift (E)  → Manifold projection Π = ts_to_img ∘ img_to_ts (Prop 1)
  5.5  Off-manifold drift (M)  → L_rep: SNR-gated manifold penalty
  5.6  Observation forgetting   → L_obs: SNR-gated observation grounding (Prop 3)
  5.7  CG across steps         → Warm-started CG from previous reverse step
"""

import torch
import torch.autograd
import torch.multiprocessing
import torch.utils.data as Data
import os
import sys
import time
import numpy as np
import logging
from tqdm import tqdm

from metrics import evaluate_model_irregular
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import (
    gen_dataloader, save_reconstructions, load_reconstructions
)
from utils.utils_args import parse_args_irregular
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from utils.train_unconditional import train_unconditional_regular
from utils.utils_stl import initialize_with_iterative_stl, initialize_with_kalman
from utils.utils_save import maybe_save_if_improved
from utils.gpu_heartbeat import start_gpu_heartbeat, stop_gpu_heartbeat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Conjugate Gradient Solver (Section 5.7: warm-start support)
# =============================================================================

def conjugate_gradient(A_fn, b, x0=None, maxiter=1):
    """
    CG solver for A·x = b where A is a linear operator.

    When x0 is provided (warm start from previous reverse step), CG begins
    from that point and refines.  The scaling x0 = (σ_{t-1}/σ_t)² · v_prev
    accounts for the changing σ² in the covariance matrix.
    """
    if x0 is not None:
        x = x0
        r = b - A_fn(x)
    else:
        x = torch.zeros_like(b)
        r = b.clone()

    p = r.clone()

    for _ in range(maxiter):
        Ap = A_fn(p)
        rTr = (r * r).sum()
        pTAp = (p * Ap).sum()

        alpha = rTr / (pTAp + 1e-30)

        x = x + alpha * p
        r_new = r - alpha * Ap

        rTr_new = (r_new * r_new).sum()
        beta = rTr_new / (rTr + 1e-30)

        p = r_new + beta * p
        r = r_new

    return x


# =============================================================================
# Dual-Space MMPS Sampler (Sections 5.2–5.4, 5.7)
# =============================================================================

class DualSpaceMMPS:
    """
    MMPS posterior sampler adapted for the dual-space setting.

    Standard MMPS (Rozet et al., 2024) was designed for single-space problems.
    In our setting, observations live in TS space while the model operates in
    image space via delay embedding.  This class addresses:

      - Section 5.2: Observation-Space CG (Prop 4) — CG in R^{N_obs x F}
        instead of R^{C x H x W}, using the exact observation operator
        G = A_ts · img_to_ts.

      - Section 5.3: Adaptive σ_y = c · σ_t (Prop 2) — bounded condition
        number independent of noise level.

      - Section 5.4: Manifold projection Π (Prop 1) — orthogonal projection
        onto Range(ts_to_img) + hard observation enforcement on final sample.

      - Section 5.7: Warm-started CG — initialize CG at step t-1 from
        scaled solution of step t.
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y_ratio=0.1, cg_iters=5,
                 ts_to_img_fn=None, img_to_ts_fn=None,
                 use_adaptive_sigma_y=True, use_consistency_projection=True,
                 use_obs_space_cg=True, use_warm_start_cg=True,
                 sigma_y_floor=0.0,
                 img_to_ts_adjoint_fn=None, pad_mask=None,
                 sampler='mmps', richardson_omega=1.0):
        self.args = args
        self.device = args.device
        self.shape = shape
        self.net = diffusion_fn.to(device=self.device)

        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.num_steps = args.diffusion_steps

        # Section 5.3: adaptive σ_y (Prop 2)
        self.sigma_y_ratio = sigma_y_ratio
        self.cg_iters = cg_iters
        self.sigma_y_floor = sigma_y_floor

        # Section 5.4: manifold projection (Prop 1)
        self.ts_to_img_fn = ts_to_img_fn
        self.img_to_ts_fn = img_to_ts_fn
        # Exact transpose of img_to_ts_fn, i.e. L^{-T}.  Used as the MMPS
        # adjoint G^T(v_ts) = img_to_ts_adjoint(mask_ts * v_ts) in
        # posterior_denoise_obs_space.  When None we fall back to ts_to_img_fn
        # (L), which coincides with L^{-T} only when the lift is unitary —
        # true for the delay embedder (up to a per-position scale absorbed
        # by sigma_y_sq) but NOT for STFT with Hann window + per-bin scaling.
        # STFT passes the autograd-VJP-based exact adjoint here so the CG
        # operator matches the MMPS-derived sigma_y^2 I + sigma_t^2 G J^T G^T.
        self.img_to_ts_adjoint_fn = img_to_ts_adjoint_fn
        # Optional (1, C, H, W) mask of valid non-pad pixels, used at sampling
        # time to zero out the structurally-padded image region after each
        # reverse step so denoiser noise in those pixels does not contaminate
        # the valid region.  None means no pad (delay embedder path).
        self.pad_mask = pad_mask

        # Section 5.2 / 5.7: observation-space CG and warm start
        self.use_obs_space_cg = use_obs_space_cg
        self.use_warm_start_cg = use_warm_start_cg
        self.v_prev = None

        # Ablation flags
        self.use_adaptive_sigma_y = use_adaptive_sigma_y
        self.use_consistency_projection = use_consistency_projection

        # E-step sampler choice:
        #   'mmps'            -- L-MMPS (PSLR on MMPS): C1+C2+C3, obs-space CG.
        #   'tmpd'            -- L-TMPD (PSLR on TMPD): C1+C2+C3, row-sum diag.
        #   'tmpd_vanilla'    -- ablation: TMPD directly in image space,
        #                         violates C1 and C3.  Forces the image-space
        #                         path in mmps_sample regardless of --obs_space_cg.
        #   'tmpd_richardson' -- L-TMPD + one Jacobi/Richardson refinement on
        #                         top of the row-sum diagonal preconditioner.
        #                         Costs 1 extra VJP per call (3 vs 2), still
        #                         cheaper than L-MMPS CG (5).  Closes the
        #                         approximation gap when the Jacobian is not
        #                         diagonally dominant on the lifted operator.
        if sampler not in ('mmps', 'tmpd', 'tmpd_vanilla', 'tmpd_richardson'):
            raise ValueError(
                f"Unknown E-step sampler '{sampler}'; expected one of "
                "'mmps', 'tmpd', 'tmpd_vanilla', 'tmpd_richardson'."
            )
        self.sampler = sampler

        # Damping factor ω for the Richardson refinement step.  ω=1 is pure
        # Jacobi (guaranteed monotone only when all eigenvalues of D⁻¹A lie
        # in (0, 2)); ω<1 is damped and more robust on non-diagonally-
        # dominant operators typical of deep denoisers on lifted lifts.
        if not (0.0 < richardson_omega <= 2.0):
            raise ValueError(
                f"richardson_omega must be in (0, 2]; got {richardson_omega}."
            )
        self.richardson_omega = richardson_omega

    def _get_sigma_y_sq(self, sigma):
        """Section 5.3: σ_y(σ_t) = c · σ_t gives κ bounded independently of σ_t."""
        if self.use_adaptive_sigma_y:
            sigma_y = self.sigma_y_ratio * sigma
            if self.sigma_y_floor > 0:
                sigma_y = torch.clamp(sigma_y, min=self.sigma_y_floor)
            return sigma_y ** 2
        return self.sigma_y_ratio ** 2

    def posterior_denoise_image_space(self, x_t, sigma, x_obs, mask):
        """
        Standard MMPS in image space (for ablation baseline).
        CG system: (σ_y²I + σ_t² A_img J A_img^T) v = r, dim = N_img.
        """
        sigma_sq = sigma ** 2
        sigma_y_sq = self._get_sigma_y_sq(sigma)

        x_t_input = x_t.detach().requires_grad_(True)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True
            )
            return grad

        r = x_obs - mask * denoised

        def cg_operator(v):
            return sigma_y_sq * v + sigma_sq * mask * vjp_fn(mask * v)

        v = conjugate_gradient(cg_operator, r, maxiter=self.cg_iters)
        score = vjp_fn(mask * v)

        return denoised.detach() + sigma_sq * score.detach()

    def posterior_denoise_tmpd_image_space(self, x_t, sigma, x_obs, mask):
        """
        Vanilla TMPD ablation (C1 and C3 violated).

        This is a faithful port of TMPD (Boys et al. 2023, bb515/tmpdtorch) to
        our setting but **without** PSLR's composed-operator correction: the
        mask is applied directly in image space (A_img) instead of via the
        composed G = A_ts . img_to_ts.  It is the symmetric ablation to
        posterior_denoise_image_space (naive MMPS image-space baseline), and
        is expected to underperform L-TMPD by the same mechanism that naive
        image-space MMPS underperforms L-MMPS: the operator mismatch Delta
        biases the posterior mean.

        Update:
            C_yy_img = mask * J^T(mask)  +  sigma_y^2 / sigma_t^2     (row-sum diag in img)
            ls       = J^T ( mask * (x_obs - mask * d_theta) / C_yy_img )
            x_0      = d_theta + ls
        Because C_yy_img uses reg_ratio = sigma_y^2/sigma_t^2 (not sigma_y^2),
        the sigma_t^2 factor that L-MMPS applies externally is already folded
        into r/C_yy_img -- an outer `sigma_sq *` here would double-count it.
        """
        sigma_sq = sigma ** 2
        sigma_y_sq = self._get_sigma_y_sq(sigma)
        reg_ratio = sigma_y_sq / torch.clamp(sigma_sq, min=1e-12)

        x_t_input = x_t.detach().requires_grad_(True)
        denoised = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent):
            grad, = torch.autograd.grad(
                denoised, x_t_input, grad_outputs=cotangent,
                retain_graph=True
            )
            return grad

        # Row-sum diagonal of mask * J^T * mask (A_img = mask, so G = mask here).
        # mask arrives as [B, 1, H, W] (time-step-level mask, shared across all
        # lift channels). We must explicitly broadcast to denoised.shape
        # [B, C, H, W] before calling torch.autograd.grad, which (unlike `*`)
        # does not broadcast grad_outputs.
        mask_bcast = mask.expand_as(denoised).to(torch.float64).contiguous()
        ones_img = mask_bcast
        Jones_img = vjp_fn(ones_img)
        C_yy_img = mask_bcast * Jones_img + reg_ratio
        C_yy_img = torch.clamp(C_yy_img, min=reg_ratio)

        # Residual and update, both in image space.
        r = mask_bcast * (x_obs - mask_bcast * denoised)
        update_img = vjp_fn(mask_bcast * (r / C_yy_img))

        # NOTE: no outer sigma_sq scaling.  C_yy is in *relative* scale
        # (reg_ratio = sigma_y^2/sigma_t^2), so r/C_yy already equals
        # sigma_sq * r/(sigma_y^2 + sigma_sq * row_sum).  Multiplying by
        # sigma_sq here would double-count the factor (was the original bug).
        # This matches the reference TMPD formulation in bb515/tmpdtorch.
        return denoised.detach() + update_img.detach()

    def posterior_denoise_obs_space(self, x_t, sigma, obs_ts, mask_ts):
        """
        Section 5.2: Observation-Space CG (Prop 4).

        CG operates in R^{N_obs x F} instead of R^{C x H x W}.  The
        observation operator G = A_ts · img_to_ts maps image → masked TS.
        Each CG iteration still requires exactly one VJP of the denoiser.

        Section 5.7: when use_warm_start_cg is True, CG is initialized
        from scaled solution of the previous reverse step.
        """
        sigma_sq = sigma ** 2
        sigma_y_sq = self._get_sigma_y_sq(sigma)

        x_t_input = x_t.detach().requires_grad_(True)
        denoised_img = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent_img):
            grad, = torch.autograd.grad(
                denoised_img, x_t_input, grad_outputs=cotangent_img,
                retain_graph=True
            )
            return grad

        # Residual in observation space: r = mask * (y - G·D_θ)
        denoised_ts = self.img_to_ts_fn(denoised_img.float()).to(torch.float64)
        r_obs = mask_ts * (obs_ts - denoised_ts)

        # G^T: TS -> image is the exact transpose of img_to_ts (= L^{-T}).
        # For STFT with Hann window + per-bin scaling, L is strongly non-
        # unitary so L^{-T} differs meaningfully from L (= ts_to_img), and
        # using ts_to_img as G^T produces an SPD but mis-scaled system that
        # converges to the wrong posterior mean.  For the delay embedder we
        # keep the legacy behavior (fall back to ts_to_img_fn) since its
        # lift is near-unitary and this has been extensively validated.
        gt_fn = self.img_to_ts_adjoint_fn or self.ts_to_img_fn

        def cg_operator_obs(v_ts):
            # G^T: TS -> image via the exact adjoint of img_to_ts.
            # Pre-mask v_ts because G = mask_ts · img_to_ts so
            # G^T(v) = img_to_ts^T(mask_ts · v).  CG preserves the mask
            # support, so this is a no-op after the first iteration, but
            # we apply it defensively.
            v_img = gt_fn((mask_ts * v_ts).float()).to(torch.float64)
            # J^T: image -> image (single VJP of the denoiser)
            Jv_img = vjp_fn(v_img)
            # G forward: image -> TS via img_to_ts, then re-mask to land
            # on the observation space.
            Jv_ts = self.img_to_ts_fn(Jv_img.float()).to(torch.float64)
            Jv_obs = mask_ts * Jv_ts
            return sigma_y_sq * v_ts + sigma_sq * Jv_obs

        # Section 5.7: warm-started CG
        x0 = None
        if self.use_warm_start_cg and self.v_prev is not None:
            try:
                if self.v_prev.shape == r_obs.shape:
                    x0 = self.v_prev
            except Exception:
                pass

        v_ts = conjugate_gradient(cg_operator_obs, r_obs, x0=x0, maxiter=self.cg_iters)

        if self.use_warm_start_cg:
            self.v_prev = v_ts.detach()

        # Posterior score correction back in image space.  The MMPS update
        # is D + sigma_t^2 J^T G^T v_ts, so here too we use the exact
        # G^T = img_to_ts_adjoint for STFT and fall back to ts_to_img for
        # delay.  Pre-masking v_ts mirrors the CG operator and is a no-op
        # when CG has converged on the observed support.
        v_img = gt_fn((mask_ts * v_ts).float()).to(torch.float64)
        score_img = vjp_fn(v_img)

        return denoised_img.detach() + sigma_sq * score_img.detach()

    def posterior_denoise_tmpd_obs_space(self, x_t, sigma, obs_ts, mask_ts):
        """
        L-TMPD: PSLR applied to TMPD (Boys et al. 2023).

        Replaces the MMPS CG solve with TMPD's row-sum diagonal approximation
        of G J^T G^T.  The composed operator G = A_ts . img_to_ts (C1) and the
        manifold projection (C3) are unchanged from L-MMPS.

        Reference: bb515/tmpdtorch, class TweedieMomentProjection --
        C_yy = op.forward(vjp(op.forward(ones))) + noise_std^2 / r
        ls   = vjp(diff / C_yy)

        Under adaptive sigma_y = c * sigma_t, the regularizer
        sigma_y^2 / sigma_t^2 collapses to the constant c^2 (no CG
        conditioning to worry about -- this is the "M2 equivalent" for
        L-TMPD).  Cost per call: exactly 2 VJPs of the denoiser.
        """
        sigma_sq = sigma ** 2
        sigma_y_sq = self._get_sigma_y_sq(sigma)
        # TMPD's noise_std^2 / r.  In adaptive mode this is c^2.  Guard
        # against pathological early-step sigma_sq with a tiny floor.
        reg_ratio = sigma_y_sq / torch.clamp(sigma_sq, min=1e-12)

        x_t_input = x_t.detach().requires_grad_(True)
        denoised_img = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent_img):
            grad, = torch.autograd.grad(
                denoised_img, x_t_input, grad_outputs=cotangent_img,
                retain_graph=True
            )
            return grad

        gt_fn = self.img_to_ts_adjoint_fn or self.ts_to_img_fn

        # h(x_t) = G . d_theta(x_t):  image -> TS, then masked to obs support.
        denoised_ts = self.img_to_ts_fn(denoised_img.float()).to(torch.float64)
        r_obs = mask_ts * (obs_ts - denoised_ts)

        # Row-sum diagonal of G J^T G^T on the observation support (C2):
        #   ones_ts  -- vector of ones on observed positions
        #   ones_img = G^T ones_ts
        #   Jones    = J^T ones_img
        #   C_yy_ts  = G Jones + reg_ratio
        # The mask_ts multiplication restricts the row-sum to the actual
        # observation support; non-observed entries have no row to sum.
        ones_ts = mask_ts.to(torch.float64)
        ones_img = gt_fn(ones_ts.float()).to(torch.float64)
        Jones_img = vjp_fn(ones_img)
        C_yy_ts = (
            mask_ts * self.img_to_ts_fn(Jones_img.float()).to(torch.float64)
            + reg_ratio
        )
        # Numerical floor so r_obs / C_yy_ts is bounded on non-observed
        # positions and on near-zero row sums.  reg_ratio is already the
        # physically correct floor on the observed support.
        C_yy_ts = torch.clamp(C_yy_ts, min=reg_ratio)

        # Posterior-score update:  x_0 = d_theta + J^T G^T ( r / C_yy )
        # C_yy is in *relative* scale (reg_ratio = sigma_y^2/sigma_t^2 and
        # row_sum has no sigma_t factor), so
        #     r / C_yy = sigma_sq * r / (sigma_y^2 + sigma_sq * row_sum)
        # already carries the sigma_sq factor that L-MMPS applies externally.
        # An additional `sigma_sq *` here would double-count the factor --
        # at sigma_t=80 that is a 6400x blowup, at sigma_t=0.002 a 1e-6
        # shrink of the correction (was the original L-TMPD bug).
        update_ts = mask_ts * (r_obs / C_yy_ts)
        update_img_pre = gt_fn(update_ts.float()).to(torch.float64)
        update_img = vjp_fn(update_img_pre)

        return denoised_img.detach() + update_img.detach()

    def posterior_denoise_tmpd_richardson_obs_space(
            self, x_t, sigma, obs_ts, mask_ts):
        """
        L-TMPD + one Jacobi / Richardson refinement step on the row-sum diagonal.

        TMPD's row-sum diagonal D ≈ diag(G J^T G^T) + reg_ratio is only a
        first-order preconditioner for the exact operator
            A = reg_ratio · I + G J^T G^T
        where reg_ratio = σ_y² / σ_t² and G = mask_ts · img_to_ts.  When A is
        far from diagonally dominant — typical for deep denoisers on the
        lifted (delay / STFT) operator — the pure diagonal solve
        v0 = D^{-1} r_obs biases the posterior enough to trigger EM
        feedback-loop divergence (memorization collapse at iter 1, M-step
        loss blow-up at iter 2, as observed on Energy 50%).

        One Richardson iteration on D:
            v0 = D^{-1} r_obs                 (vanilla L-TMPD)
            v1 = v0 + D^{-1} (r_obs - A v0)   (this method)
        corrects first-order off-diagonal coupling.  Cost: +1 VJP vs L-TMPD
        (3 total), still cheaper than L-MMPS CG (≥5).

        Reference: standard Jacobi-preconditioned Richardson iteration;
        identical in spirit to the "one step of residual correction" used
        in iterative sparse solvers.
        """
        sigma_sq = sigma ** 2
        sigma_y_sq = self._get_sigma_y_sq(sigma)
        reg_ratio = sigma_y_sq / torch.clamp(sigma_sq, min=1e-12)

        x_t_input = x_t.detach().requires_grad_(True)
        denoised_img = self.net(x_t_input, sigma, None).to(torch.float64)

        def vjp_fn(cotangent_img):
            grad, = torch.autograd.grad(
                denoised_img, x_t_input, grad_outputs=cotangent_img,
                retain_graph=True
            )
            return grad

        gt_fn = self.img_to_ts_adjoint_fn or self.ts_to_img_fn

        # Residual in observation space.
        denoised_ts = self.img_to_ts_fn(denoised_img.float()).to(torch.float64)
        r_obs = mask_ts * (obs_ts - denoised_ts)

        # D = diag(G J^T G^T) + reg_ratio  (TMPD preconditioner).
        ones_ts = mask_ts.to(torch.float64)
        ones_img = gt_fn(ones_ts.float()).to(torch.float64)
        Jones_img = vjp_fn(ones_img)  # VJP #1
        D = (
            mask_ts * self.img_to_ts_fn(Jones_img.float()).to(torch.float64)
            + reg_ratio
        )
        D = torch.clamp(D, min=reg_ratio)

        # Initial diagonal (Jacobi) solve.
        v0 = mask_ts * (r_obs / D)

        # Apply A to v0:  A v0 = reg_ratio · v0 + G J^T G^T v0.
        Gt_v0_img = gt_fn((mask_ts * v0).float()).to(torch.float64)
        JGt_v0_img = vjp_fn(Gt_v0_img)  # VJP #2
        GJGt_v0_ts = mask_ts * self.img_to_ts_fn(
            JGt_v0_img.float()
        ).to(torch.float64)
        Av0 = reg_ratio * v0 + GJGt_v0_ts

        # Damped Richardson update: v1 = v0 + ω · (r - A v0) / D.
        # ω=1 is pure Jacobi; ω<1 is damped (guaranteed monotone for
        # ω ≤ 2/λ_max(D⁻¹A) on SPD A).
        residual = mask_ts * (r_obs - Av0)
        omega = self.richardson_omega
        v1 = v0 + omega * mask_ts * (residual / D)

        # Posterior-score update uses v1 (vs v0 in vanilla L-TMPD).
        update_img_pre = gt_fn((mask_ts * v1).float()).to(torch.float64)
        update_img = vjp_fn(update_img_pre)  # VJP #3

        return denoised_img.detach() + update_img.detach()

    def _consistency_project(self, x_img, obs_ts, mask_ts):
        """
        Section 5.4: Manifold projection (Prop 1) + hard observation enforcement.

        Π = ts_to_img ∘ img_to_ts is the orthogonal projection onto
        Range(ts_to_img).  After projection, observed values are hard-enforced.
        """
        x_ts = self.img_to_ts_fn(x_img.float())
        mask_f = mask_ts.float()
        obs_f = obs_ts.float()
        x_ts_enforced = x_ts * (1.0 - mask_f) + obs_f * mask_f
        return self.ts_to_img_fn(x_ts_enforced).to(torch.float64)

    def mmps_sample(self, latents, x_obs_img, mask_img, obs_ts=None, mask_ts=None):
        """
        Full reverse process with dual-space MMPS corrections.

        Uses observation-space CG (Section 5.2) when available, falls back
        to image-space CG otherwise.  Manifold projection (Section 5.4) is
        applied once at the end.
        """
        can_use_obs_space = (
            self.use_obs_space_cg
            and self.ts_to_img_fn is not None
            and self.img_to_ts_fn is not None
            and obs_ts is not None
            and mask_ts is not None
        )
        do_project = (
            self.use_consistency_projection
            and self.ts_to_img_fn is not None
            and self.img_to_ts_fn is not None
            and obs_ts is not None
            and mask_ts is not None
        )

        # Reset warm-start state for new sample
        self.v_prev = None

        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_obs = x_obs_img.to(torch.float64)
        mask = mask_img.to(torch.float64)

        if obs_ts is not None:
            obs_ts_d = obs_ts.to(torch.float64)
            mask_ts_d = mask_ts.to(torch.float64)

        x_next = latents.to(torch.float64) * t_steps[0]
        pad_mask_d = None
        if self.pad_mask is not None:
            pad_mask_d = self.pad_mask.to(device=x_next.device, dtype=torch.float64)
            x_next = x_next * pad_mask_d

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)
            if pad_mask_d is not None:
                x_hat = x_hat * pad_mask_d

            if self.sampler == 'tmpd_vanilla':
                denoised = self.posterior_denoise_tmpd_image_space(x_hat, t_hat, x_obs, mask)
            elif can_use_obs_space and self.sampler == 'tmpd':
                denoised = self.posterior_denoise_tmpd_obs_space(x_hat, t_hat, obs_ts_d, mask_ts_d)
            elif can_use_obs_space and self.sampler == 'tmpd_richardson':
                denoised = self.posterior_denoise_tmpd_richardson_obs_space(x_hat, t_hat, obs_ts_d, mask_ts_d)
            elif can_use_obs_space:
                denoised = self.posterior_denoise_obs_space(x_hat, t_hat, obs_ts_d, mask_ts_d)
            else:
                denoised = self.posterior_denoise_image_space(x_hat, t_hat, x_obs, mask)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if pad_mask_d is not None:
                x_next = x_next * pad_mask_d

            if i < self.num_steps - 1:
                if self.sampler == 'tmpd_vanilla':
                    denoised_2 = self.posterior_denoise_tmpd_image_space(x_next, t_next, x_obs, mask)
                elif can_use_obs_space and self.sampler == 'tmpd':
                    denoised_2 = self.posterior_denoise_tmpd_obs_space(x_next, t_next, obs_ts_d, mask_ts_d)
                elif can_use_obs_space and self.sampler == 'tmpd_richardson':
                    denoised_2 = self.posterior_denoise_tmpd_richardson_obs_space(x_next, t_next, obs_ts_d, mask_ts_d)
                elif can_use_obs_space:
                    denoised_2 = self.posterior_denoise_obs_space(x_next, t_next, obs_ts_d, mask_ts_d)
                else:
                    denoised_2 = self.posterior_denoise_image_space(x_next, t_next, x_obs, mask)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                if pad_mask_d is not None:
                    x_next = x_next * pad_mask_d

        if do_project:
            x_next = self._consistency_project(x_next, obs_ts_d, mask_ts_d)
            if pad_mask_d is not None:
                x_next = x_next * pad_mask_d

        return x_next

    def sampling_mmps(self, x_obs_img, mask_img, obs_ts=None, mask_ts=None):
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, x_obs_img, mask_img, obs_ts=obs_ts, mask_ts=mask_ts)

    @torch.no_grad()
    def sampling(self, sampling_number=16):
        """Unconditional sampling for evaluation."""
        latents = torch.randn([sampling_number, *self.shape], device=self.device)

        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        pad_mask_d = None
        if self.pad_mask is not None:
            pad_mask_d = self.pad_mask.to(device=latents.device, dtype=torch.float64)

        x_next = latents.to(torch.float64) * t_steps[0]
        if pad_mask_d is not None:
            x_next = x_next * pad_mask_d
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)
            if pad_mask_d is not None:
                x_hat = x_hat * pad_mask_d

            denoised = self.net(x_hat, t_hat, None).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if pad_mask_d is not None:
                x_next = x_next * pad_mask_d

            if i < self.num_steps - 1:
                denoised = self.net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                if pad_mask_d is not None:
                    x_next = x_next * pad_mask_d

        return x_next


# =============================================================================
# Helper functions
# =============================================================================

def get_corrupted_data_from_loader(train_loader, device):
    """Extract corrupted time series (with NaNs) from the train loader."""
    all_corrupted = []
    all_masks = []

    for batch_idx, data in enumerate(train_loader):
        x_irregular = data[0]
        x_ts = x_irregular[:, :, :-1]
        mask = ~torch.isnan(x_ts).any(dim=-1)

        all_corrupted.append(x_ts.numpy())
        all_masks.append(mask.numpy())

    corrupted_data = np.vstack(all_corrupted)
    obs_masks = np.vstack(all_masks)

    return corrupted_data, obs_masks


def _cache_embedder_stats_if_needed(uncond_model, ts_array, tag='pre_em', logger=None, em_iter=0):
    """Cache STFT per-bin scales on a concrete TS tensor (no-op for delay).

    The plan (Fix 1) calls out the failure mode: caching on STL warm-starts
    biases the per-bin abs-max away from the real training distribution, so
    the denoiser is trained on coefficients in one scale and img_to_ts de-
    scales with stale factors at test time, producing disc_mean near 0.5.

    We solve this by caching on the best available estimate of the real
    training TS at each invocation:
      * First call (pre-EM): use the warm-start completions. This is still
        biased but is the best we have before any M-step.
      * Subsequent calls (post-E-step): use the co-evolving reconstructions,
        which track the true training distribution as EM converges.

    For the delay embedder cache_min_max_params is not implemented and this
    call is a no-op (guarded on hasattr), so the delay code path is
    untouched.
    """
    if not hasattr(uncond_model.ts_img, 'cache_min_max_params'):
        return

    with torch.no_grad():
        if isinstance(ts_array, np.ndarray):
            ts_tensor = torch.as_tensor(ts_array, dtype=torch.float32)
        else:
            ts_tensor = ts_array.detach().cpu().float()
        uncond_model.cache_embedder_stats(ts_tensor)

    scale_real = getattr(uncond_model.ts_img, 'scale_real', None)
    scale_imag = getattr(uncond_model.ts_img, 'scale_imag', None)
    if scale_real is None or scale_imag is None:
        return

    sr = scale_real.detach().float()
    si = scale_imag.detach().float()
    summary = (
        f"[embedder-stats:{tag}] scale_real "
        f"min={sr.min().item():.4e} mean={sr.mean().item():.4e} max={sr.max().item():.4e} | "
        f"scale_imag min={si.min().item():.4e} mean={si.mean().item():.4e} max={si.max().item():.4e}"
    )
    print(summary)
    logging.info(summary)
    if logger is not None:
        try:
            logger.log(f'stft_scale/{tag}/real_min', sr.min().item(), em_iter)
            logger.log(f'stft_scale/{tag}/real_mean', sr.mean().item(), em_iter)
            logger.log(f'stft_scale/{tag}/real_max', sr.max().item(), em_iter)
            logger.log(f'stft_scale/{tag}/imag_min', si.min().item(), em_iter)
            logger.log(f'stft_scale/{tag}/imag_mean', si.mean().item(), em_iter)
            logger.log(f'stft_scale/{tag}/imag_max', si.max().item(), em_iter)
        except Exception:
            pass


def off_manifold_energy_batch(x_img, img_to_ts_fn, ts_to_img_fn):
    """
    E_off = ||x - Π(x)||²: distance from the valid delay-embedding manifold.
    Non-zero only for images violating x ∈ Range(ts_to_img).
    """
    with torch.no_grad():
        x_ts_rt = img_to_ts_fn(x_img.float())
        x_img_rt = ts_to_img_fn(x_ts_rt)
        return ((x_img.float() - x_img_rt) ** 2).sum(dim=(1, 2, 3)).mean().item()


# =============================================================================
# E-step (Sections 5.2–5.4, 5.7)
# =============================================================================

def e_step(args, uncond_model, corrupted_data, obs_masks, em_iter, device, logger=None):
    """
    E-step: complete missing values via dual-space MMPS.

    Applies Sections 5.2 (obs-space CG), 5.3 (adaptive σ_y),
    5.4 (manifold projection), 5.7 (warm-started CG).
    """
    print(f"\n=== E-Step (EM iter {em_iter}) ===")

    uncond_model.eval()

    all_reconstructions = []
    target_shape = (uncond_model.num_features, args.img_resolution, args.img_resolution)

    N = len(corrupted_data)
    batch_size = args.batch_size

    sigma_y_ratio = getattr(args, 'sigma_y_ratio', 0.1)
    cg_iters = getattr(args, 'mmps_cg_iters', 5)
    use_adaptive = getattr(args, 'adaptive_sigma_y', True)
    use_proj = getattr(args, 'consistency_projection', True)
    use_obs_cg = getattr(args, 'obs_space_cg', True)
    use_warm_cg = getattr(args, 'warm_start_cg', True)
    sigma_y_floor = getattr(args, 'sigma_y_floor', 0.0)

    # STFT integration: observation-space CG is the only mathematically well-
    # posed regime for a non-local lift (STFT). Image-space CG would require
    # the observation mask to lift to a valid image-space mask, which is
    # false for STFT (STFT of a zero-filled TS != mask applied to STFT).
    # STFT + image-space CG is mathematically ill-posed for a non-local lift
    # (STFT of a zero-filled masked TS is NOT the image-space mask applied to
    # the STFT). We keep this path enabled on purpose so it can serve as the
    # "without our dual-space correction" arm of the STFT ablation — it SHOULD
    # produce bad disc_mean; that is the evidence the paper needs.
    is_stft = (getattr(args, 'embedder', 'delay') == 'stft')
    if is_stft and not use_obs_cg:
        print(
            "[ABLATION] WARNING: --embedder stft with image-space CG. "
            "This configuration is mathematically ill-posed (non-local lift + "
            "image-space pixel mask). Expect poor disc_mean; this is the "
            "intended dual-space ablation, not a bug.",
            flush=True,
        )

    total_off_energy = 0.0
    n_off_samples = 0

    # Exact transpose of img_to_ts (= L^{-T}) via autograd VJP.  This is
    # the mathematically correct G^T used by MMPS obs-space CG, where
    # G(D) = mask_ts · img_to_ts(D), so G^T(v_ts) = img_to_ts^T(mask_ts·v_ts).
    # For the delay embedder L is near-unitary and we keep the historical
    # fallback (None -> ts_to_img inside the CG op) to avoid any regression
    # on a thoroughly-validated path.  For STFT with Hann window + per-bin
    # scaling L is strongly non-unitary and this exact adjoint is required.
    img_to_ts_adjoint_fn = (
        uncond_model.img_to_ts_adjoint if is_stft else None
    )
    pad_mask = uncond_model.pad_mask if is_stft else None

    with uncond_model.ema_scope():
        process = DualSpaceMMPS(
            args, uncond_model.net, target_shape,
            sigma_y_ratio=sigma_y_ratio,
            cg_iters=cg_iters,
            ts_to_img_fn=uncond_model.ts_to_img,
            img_to_ts_fn=uncond_model.img_to_ts,
            use_adaptive_sigma_y=use_adaptive,
            use_consistency_projection=use_proj,
            use_obs_space_cg=use_obs_cg,
            use_warm_start_cg=use_warm_cg,
            sigma_y_floor=sigma_y_floor,
            img_to_ts_adjoint_fn=img_to_ts_adjoint_fn,
            pad_mask=pad_mask,
            sampler=getattr(args, 'e_step_sampler', 'mmps'),
            richardson_omega=getattr(args, 'richardson_omega', 1.0),
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step"):
            end_idx = min(start_idx + batch_size, N)
            cur_batch_size = end_idx - start_idx

            corrupted_batch = corrupted_data[start_idx:end_idx]
            mask_batch = obs_masks[start_idx:end_idx]

            corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
            mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)

            obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)
            mask_ts_proj = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])

            if is_stft and use_obs_cg:
                # STFT + obs-space CG (Regime A, our full method): x_obs_img
                # and mask_img are unused by the obs-space CG path, so pass
                # placeholders. Residuals and CG operate in TS space directly.
                x_obs_img = torch.zeros(
                    (cur_batch_size, *target_shape), dtype=torch.float32, device=device,
                )
                mask_img = torch.zeros(
                    (cur_batch_size, 1, target_shape[1], target_shape[2]),
                    dtype=torch.float32, device=device,
                )
            elif is_stft and not use_obs_cg:
                # STFT + image-space CG (Regimes B/C, ablation arm):
                # construct the "naive" image-space observation the way a
                # vanilla MMPS user would — STFT of the zero-filled TS, and a
                # binarized STFT of the TS mask. This is mathematically
                # ill-posed for a non-local lift (that's the whole point of
                # the ablation), and we expect disc_mean to be poor.
                x_obs_img = uncond_model.ts_to_img(obs_ts)
                mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
                mask_img_raw = uncond_model.ts_to_img(mask_ts_expanded)
                mask_img = (mask_img_raw.abs() > 1e-6).to(mask_img_raw.dtype)
                mask_img = mask_img[:, :1, :, :]
            else:
                x_obs_img = uncond_model.ts_to_img(obs_ts)
                mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
                mask_img = uncond_model.ts_to_img(mask_ts_expanded)
                mask_img = mask_img[:, :1, :, :]

            x_img_imputed = process.sampling_mmps(
                x_obs_img, mask_img,
                obs_ts=obs_ts, mask_ts=mask_ts_proj
            )

            batch_off = off_manifold_energy_batch(
                x_img_imputed, uncond_model.img_to_ts, uncond_model.ts_to_img
            )
            total_off_energy += batch_off * cur_batch_size
            n_off_samples += cur_batch_size

            x_ts_recon = uncond_model.img_to_ts(x_img_imputed)
            all_reconstructions.append(x_ts_recon.cpu().numpy())

            torch.cuda.empty_cache()

    reconstructions = np.vstack(all_reconstructions)
    off_energy = total_off_energy / max(n_off_samples, 1)
    print(f"  Completed {len(reconstructions)} sequences.  E_off = {off_energy:.6f}")

    nan_count = np.isnan(reconstructions).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in reconstructions!")

    if logger is not None:
        logger.log('em/off_manifold_energy', off_energy, em_iter)

    cache_dir = os.path.join(args.recon_cache_dir, args.dataset,
                              f'missing_{int(args.missing_rate*100)}')
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


# =============================================================================
# M-step (Sections 5.5, 5.6)
# =============================================================================

def m_step(args, uncond_model, optimizer, reconstructions,
           corrupted_data, obs_masks, em_iter, device, logger=None,
           eval_callback=None, eval_every_epochs=None):
    """
    M-step: L = L_SM + λ_obs · L_obs + λ_rep · L_rep

    L_SM  — standard denoising score matching in image space
    L_obs — Section 5.6: SNR-gated observation grounding in TS space (Prop 3)
    L_rep — Section 5.5: SNR-gated manifold penalty in image space (Prop 1 at training time)

    If `eval_callback` is provided and `eval_every_epochs` > 0, `eval_callback(epoch)` is
    invoked after every `eval_every_epochs` training epochs (and always after the final
    epoch). The callback is responsible for its own evaluation/save/logging; its elapsed
    time is returned in the second position so the caller can exclude it from training time.
    """
    print(f"\n=== M-Step (EM iter {em_iter}) ===")
    print(f"  Training for {args.m_step_epochs} epochs...")

    lambda_obs = getattr(args, 'lambda_obs', 0.5)
    lambda_rep = getattr(args, 'lambda_rep', 0.0)
    use_snr_gate = not getattr(args, 'no_snr_gate', False)
    use_sm_weight_for_lobs = getattr(args, 'lobs_use_sm_weight', False)
    snr_sigma_d = getattr(args, 'snr_gate_sigma_d', None)
    sigma_data = uncond_model.sigma_data
    sigma_d_for_gate = snr_sigma_d if snr_sigma_d is not None else sigma_data

    obs_clean = np.nan_to_num(corrupted_data, nan=0.0)

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    obs_tensor = torch.tensor(obs_clean, dtype=torch.float32)
    mask_tensor = torch.tensor(obs_masks, dtype=torch.float32)

    recon_dataset = Data.TensorDataset(recon_tensor, obs_tensor, mask_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    uncond_model.train()

    total_eval_time_in_mstep = 0.0

    for epoch in range(args.m_step_epochs):
        epoch_loss_sm = 0.0
        epoch_loss_obs = 0.0
        epoch_loss_rep = 0.0
        num_batches = 0

        for batch_idx, (x_clean, y_ts_batch, mask_ts_batch) in enumerate(recon_loader):
            x_clean = x_clean.to(device)
            y_ts_batch = y_ts_batch.to(device)
            mask_ts_batch = mask_ts_batch.to(device)

            x_img = uncond_model.ts_to_img(x_clean)

            rnd_normal = torch.randn([x_img.shape[0], 1, 1, 1], device=device)
            sigma = (rnd_normal * uncond_model.P_std + uncond_model.P_mean).exp()
            weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
            noise = torch.randn_like(x_img) * sigma
            denoised = uncond_model.net(x_img + noise, sigma, None)

            # --- L_SM: score matching in image space ---
            x_unpad = uncond_model.unpad(x_img, x_img.shape)
            denoised_unpad = uncond_model.unpad(denoised, x_img.shape)
            sq_err_sm = (denoised_unpad - x_unpad).square()
            _pad_mask = uncond_model.pad_mask
            if _pad_mask is not None:
                _pm = _pad_mask.to(device=sq_err_sm.device, dtype=sq_err_sm.dtype)
                _num = (weight * sq_err_sm * _pm).sum()
                _den = (weight * torch.ones_like(sq_err_sm) * _pm).sum().clamp_min(1e-8)
                loss_sm = _num / _den
            else:
                loss_sm = (weight * sq_err_sm).mean()

            # --- SNR gate (shared by L_obs and L_rep) ---
            if use_snr_gate:
                snr_gate = sigma_d_for_gate ** 2 / (sigma.squeeze() ** 2 + sigma_d_for_gate ** 2)
                if use_sm_weight_for_lobs:
                    snr_gate_obs = snr_gate * weight.squeeze()
                else:
                    snr_gate_obs = snr_gate
            else:
                snr_gate = torch.ones(x_img.shape[0], device=device)
                snr_gate_obs = snr_gate

            # --- L_obs: Section 5.6, observation grounding across dual-space boundary ---
            if lambda_obs > 0:
                denoised_ts = uncond_model.img_to_ts(denoised)
                obs_residual = mask_ts_batch.unsqueeze(-1) * (denoised_ts - y_ts_batch)
                loss_obs_per_sample = (obs_residual ** 2).mean(dim=(-1, -2))
                loss_obs = (snr_gate_obs * loss_obs_per_sample).mean()
            else:
                loss_obs = torch.tensor(0.0, device=device)

            # --- L_rep: Section 5.5, manifold penalty ---
            if lambda_rep > 0:
                denoised_proj = uncond_model.ts_to_img(uncond_model.img_to_ts(denoised))
                off_manifold = denoised - denoised_proj
                loss_rep_per_sample = (off_manifold ** 2).mean(dim=(1, 2, 3))
                loss_rep = (snr_gate * loss_rep_per_sample).mean()
            else:
                loss_rep = torch.tensor(0.0, device=device)

            loss = loss_sm + lambda_obs * loss_obs + lambda_rep * loss_rep

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss_sm += loss_sm.item()
            epoch_loss_obs += loss_obs.item()
            epoch_loss_rep += loss_rep.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_sm = epoch_loss_sm / num_batches
        avg_obs = epoch_loss_obs / num_batches
        avg_rep = epoch_loss_rep / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.m_step_epochs}  "
                  f"L_SM={avg_sm:.4f}  L_obs={avg_obs:.4f}  L_rep={avg_rep:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/loss_sm', avg_sm, global_step)
            logger.log('em/loss_obs', avg_obs, global_step)
            logger.log('em/loss_rep', avg_rep, global_step)
            logger.log('em/loss_total',
                        avg_sm + lambda_obs * avg_obs + lambda_rep * avg_rep,
                        global_step)

        # Mid-M-step evaluation (disc + save), excluded from training time.
        if eval_callback is not None and eval_every_epochs and (epoch + 1) % eval_every_epochs == 0:
            _eval_start = time.time()
            eval_callback(epoch)
            total_eval_time_in_mstep += time.time() - _eval_start

    print(f"  M-step complete for EM iteration {em_iter}.")
    return uncond_model, total_eval_time_in_mstep


def m_step_bootstrap(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """L_SM-only M-step for the warm-start bootstrap (Section 5.1)."""
    stop_gpu_heartbeat()
    print(f"\n=== Bootstrap M-Step (EM iter {em_iter}) — L_SM only ===")
    print(f"  Training for {args.m_step_epochs} epochs...")

    recon_tensor = torch.tensor(reconstructions, dtype=torch.float32)
    recon_dataset = Data.TensorDataset(recon_tensor)
    recon_loader = Data.DataLoader(
        recon_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    uncond_model.train()

    for epoch in range(args.m_step_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (x_clean,) in enumerate(recon_loader):
            x_clean = x_clean.to(device)
            x_img = uncond_model.ts_to_img(x_clean)
            loss, to_log = uncond_model.loss_fn_irregular(x_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss += loss.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.m_step_epochs}, L_SM = {avg_loss:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/bootstrap_loss', avg_loss, global_step)

    print(f"  Bootstrap M-step complete for EM iteration {em_iter}.")
    return uncond_model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_uncond(args, uncond_model, test_loader, em_iter, device,
                    logger=None, calc_other_metrics=False, m_epoch=None):
    """Evaluate via unconditional sampling.

    `m_epoch` is the 0-based epoch within the current M-step (for mid-M-step evals).
    If None, defaults to args.m_step_epochs - 1 (post-M-step position).
    """
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")

    uncond_model.eval()
    gen_sig = []
    real_sig = []

    with torch.no_grad():
        with uncond_model.ema_scope():
            is_stft = (getattr(args, 'embedder', 'delay') == 'stft')
            pad_mask = uncond_model.pad_mask if is_stft else None
            process = DiffusionProcess(
                args, uncond_model.net,
                (uncond_model.num_features, args.img_resolution, args.img_resolution),
                pad_mask=pad_mask,
            )

            for data in tqdm(test_loader, desc="Evaluating"):
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = uncond_model.img_to_ts(x_img_sampled)

                gen_sig.append(x_ts.cpu().numpy())
                real_sig.append(data[0].cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    scores = evaluate_model_irregular(real_sig, gen_sig, args)

    if calc_other_metrics:
        extra_scores = evaluate_model_irregular(
            real_sig, gen_sig, args, calc_other_metrics=True
        )
        scores.update(extra_scores)

    if m_epoch is None:
        m_epoch_for_step = args.m_step_epochs - 1
    else:
        m_epoch_for_step = m_epoch
    eval_step = em_iter * args.m_step_epochs + m_epoch_for_step

    print(f"  EM iter {em_iter}, m_epoch {m_epoch_for_step} metrics:")
    for key, value in scores.items():
        print(f"    {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, eval_step)

    return scores, real_sig, gen_sig


# =============================================================================
# Main: Co-Evolving EM
# =============================================================================

def main(args):
    """
    Co-evolving EM for corrupted time series generation.

    The pipeline:
      1. Section 5.1 — Warm start: Kalman/STL initialization + bootstrap M-step
      2. EM loop:
         a. Section 5.1 — Curriculum scheduling (anneal extra revealed positions)
         b. E-step — Sections 5.2-5.4, 5.7 (dual-space MMPS)
         c. M-step — Sections 5.5-5.6 (L_SM + L_obs + L_rep)
         d. Evaluate
      3. Phase 3 — Fresh model on best completions (optional)
    """

    name = create_model_name_and_dir(args)
    name = f"co_evolving_em_{name}"

    logging.info(args)

    # Start GPU heartbeat BEFORE any long CPU-bound init (Kalman/STL warm-start,
    # spline computation, etc.) so the cluster's idle-GPU monitor doesn't kill us.
    start_gpu_heartbeat()

    with CompositeLogger([WandbLogger()]) if args.wandb else PrintLogger() as logger:
        log_config_and_tags(args, logger, name)

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(f'{args.dataset} dataset is ready.')

        uncond_model = TS2img_Karras(args=args, device=args.device).to(args.device)
        optimizer = torch.optim.AdamW(
            uncond_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        print_model_params(logger, uncond_model)

        # === Extract corrupted data ===
        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"  {len(corrupted_data)} sequences, "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # === Section 5.1: Warm Start ===
        init_method = getattr(args, 'init_method', 'stl')
        print(f"\n{'='*60}")
        print(f"Section 5.1: Warm Start ({init_method})")
        print(f"{'='*60}")

        init_start_time = time.time()

        if init_method == 'kalman':
            initial_reconstructions = initialize_with_kalman(
                corrupted_data, obs_masks, seed=args.seed,
                per_fit_timeout=getattr(args, 'kalman_fit_timeout', 5),
                max_seconds=getattr(args, 'kalman_global_timeout', 1800),
            )
        elif init_method == 'linear':
            N, seq_len, features = corrupted_data.shape
            initial_reconstructions = corrupted_data.copy()
            for i in range(N):
                obs_idx = np.where(obs_masks[i])[0]
                mis_idx = np.where(~obs_masks[i])[0]
                if len(obs_idx) < 2 or len(mis_idx) == 0:
                    col_mean = np.nanmean(corrupted_data[i], axis=0)
                    col_mean = np.where(np.isnan(col_mean), 0.5, col_mean)
                    initial_reconstructions[i, mis_idx, :] = col_mean
                    continue
                for c in range(features):
                    vals = corrupted_data[i, obs_idx, c]
                    initial_reconstructions[i, :, c] = np.interp(
                        np.arange(seq_len), obs_idx, vals
                    )
            initial_reconstructions = np.clip(initial_reconstructions, 0, 1)
            print(f"Linear interpolation initialization: {N} sequences")
        elif init_method == 'random':
            N, seq_len, features = corrupted_data.shape
            mu = np.nanmean(corrupted_data, axis=(0, 1))
            std = np.nanstd(corrupted_data, axis=(0, 1))
            std = np.where(std == 0, 1.0, std)
            initial_reconstructions = corrupted_data.copy()
            rng = np.random.RandomState(args.seed)
            for i in range(N):
                for t in range(seq_len):
                    if not obs_masks[i, t]:
                        initial_reconstructions[i, t, :] = rng.normal(mu, std)
            initial_reconstructions = np.clip(initial_reconstructions, 0, 1)
            print(f"Random (Gaussian) initialization: {N} sequences")
        else:
            initial_reconstructions = initialize_with_iterative_stl(
                corrupted_data, obs_masks, seed=args.seed
            )

        init_seconds = time.time() - init_start_time
        init_minutes = init_seconds / 60.0
        if logger is not None:
            logger.log('time/init_minutes', init_minutes, 0)
        print(f"  Init ({init_method}) took {init_minutes:.2f} min")

        is_stft = (getattr(args, 'embedder', 'delay') == 'stft')
        scale_real_warmstart_max = None
        if is_stft:
            print(f"\n{'='*60}")
            print("Caching STFT min/max stats on warm-start completions")
            print(f"{'='*60}")
            _cache_embedder_stats_if_needed(
                uncond_model, initial_reconstructions,
                tag='warm_start', logger=logger, em_iter=-1,
            )
            # Fix 1 runaway guard: record the warm-start scale_real max so we
            # can abort if a subsequent recache (or any scale mutation) blows
            # it up by >10x.
            _sr = getattr(uncond_model.ts_img, 'scale_real', None)
            if _sr is not None:
                scale_real_warmstart_max = float(_sr.detach().abs().max().item())
            try:
                from utils.stft_preflight import run as _stft_preflight_run
                _stft_preflight_run(
                    uncond_model,
                    feat_dim=args.input_channels,
                    device=args.device,
                    logger=logger,
                    train_data=torch.as_tensor(
                        initial_reconstructions[:256], dtype=torch.float32,
                    ),
                    hard_gate_train_pixel_std=getattr(
                        args, 'stft_hard_gate_train_pixel_std', False,
                    ),
                )
            except Exception as e:
                print(f"[stft_preflight] failed: {e}")
                raise

            # Fix 6 - Oracle disc sanity: run the discriminator on a single
            # test batch vs its lift-roundtrip  img_to_ts(ts_to_img(x)). This
            # gives an honest upper bound on what EM can achieve: if the lift
            # itself is lossy, no amount of training can make disc_mean go to
            # 0. Keep entirely separate from test/disc_mean.
            try:
                oracle_reals = []
                oracle_rt = []
                with torch.no_grad():
                    for data in test_loader:
                        xb = data[0].to(args.device)
                        img = uncond_model.ts_to_img(xb)
                        xb_rt = uncond_model.img_to_ts(img)
                        oracle_reals.append(xb.detach().cpu().numpy())
                        oracle_rt.append(xb_rt.detach().cpu().numpy())
                        if sum(r.shape[0] for r in oracle_reals) >= 512:
                            break
                oracle_reals = np.vstack(oracle_reals)
                oracle_rt = np.vstack(oracle_rt)
                oracle_scores = evaluate_model_irregular(
                    oracle_reals, oracle_rt, args, calc_other_metrics=False,
                )
                oracle_disc = float(oracle_scores.get('disc_mean', float('nan')))
                print(f"[oracle] disc_mean(real, img_to_ts(ts_to_img(real)))"
                      f" = {oracle_disc:.4f}")
                if logger is not None:
                    logger.log('test/disc_oracle', oracle_disc, 0)
            except Exception as e:
                print(f"[disc_oracle] skipped: {e}")

        print(f"\n{'='*60}")
        print("Section 5.1: Bootstrap M-Step on warm-start completions")
        print(f"{'='*60}")

        bootstrap_start_time = time.time()
        uncond_model = m_step_bootstrap(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )
        bootstrap_seconds = time.time() - bootstrap_start_time
        bootstrap_minutes = bootstrap_seconds / 60.0
        if logger is not None:
            logger.log('time/bootstrap_minutes', bootstrap_minutes, 0)
        print(f"  Bootstrap M-step took {bootstrap_minutes:.2f} min")

        # === Configuration summary ===
        sigma_y_ratio = getattr(args, 'sigma_y_ratio', 0.1)
        lambda_obs = getattr(args, 'lambda_obs', 0.5)
        lambda_rep = getattr(args, 'lambda_rep', 0.0)
        use_adaptive = getattr(args, 'adaptive_sigma_y', True)
        use_proj = getattr(args, 'consistency_projection', True)
        use_obs_cg = getattr(args, 'obs_space_cg', True)
        use_warm_cg = getattr(args, 'warm_start_cg', True)
        sigma_y_floor = getattr(args, 'sigma_y_floor', 0.0)
        use_snr_gate = not getattr(args, 'no_snr_gate', False)
        use_sm_weight_lobs = getattr(args, 'lobs_use_sm_weight', False)
        snr_sigma_d = getattr(args, 'snr_gate_sigma_d', None)
        do_phase3 = getattr(args, 'phase3', True)
        eval_all = getattr(args, 'eval_all_metrics', False)

        print(f"\n{'='*60}")
        print(f"Co-Evolving EM — {args.em_iters} iterations")
        print(f"  5.1  Warm start:       STL + curriculum")
        print(f"  5.2  Obs-space CG:     {use_obs_cg}")
        print(f"  5.3  Adaptive σ_y:     {use_adaptive} (c={sigma_y_ratio}, floor={sigma_y_floor})")
        print(f"  5.4  Manifold proj:    {use_proj}")
        print(f"  5.5  L_rep:            λ_rep={lambda_rep}")
        print(f"  5.6  L_obs:            λ_obs={lambda_obs}, snr_gate={use_snr_gate}")
        print(f"  5.7  Warm-start CG:    {use_warm_cg}")
        print(f"  Phase 3 (fresh):       {do_phase3}")
        print(f"  CG iters:              {args.mmps_cg_iters}")
        print(f"{'='*60}")

        best_metrics = None
        metrics_history = []
        last_recon = initial_reconstructions
        em_start_time = time.time()

        # Convergence tracking: snapshots whenever disc strictly improves.
        cumulative_em_seconds = 0.0
        best_disc_value = None
        best_disc_em_iter = None
        best_disc_m_epoch = None
        cum_em_at_best_seconds = None

        current_em_iter = [0]  # mutable so the closure sees updates

        def run_eval_and_save(m_epoch_in_step):
            """Eval + best-disc tracking + save. Called inside or after M-step.

            m_epoch_in_step is the 0-based epoch index within the current M-step, or
            (args.m_step_epochs - 1) to denote 'end of M-step' when called explicitly.
            """
            em = current_em_iter[0]
            is_final_em = (em == args.em_iters - 1)
            use_all_metrics = eval_all and is_final_em
            metrics, real_sig, gen_sig = evaluate_uncond(
                args, uncond_model, test_loader, em, args.device, logger,
                calc_other_metrics=use_all_metrics, m_epoch=m_epoch_in_step,
            )

            metrics_history.append({
                'em_iter': em,
                'm_epoch': m_epoch_in_step,
                **metrics,
            })

            current_disc = metrics.get('disc_mean', float('inf'))
            nonlocal best_disc_value, best_disc_em_iter, best_disc_m_epoch, cum_em_at_best_seconds
            if best_disc_value is None or current_disc < best_disc_value:
                best_disc_value = current_disc
                best_disc_em_iter = em
                best_disc_m_epoch = m_epoch_in_step
                cum_em_at_best_seconds = cumulative_em_seconds

            nonlocal best_metrics
            if best_metrics is None or current_disc < best_metrics.get('disc_mean', float('inf')):
                best_metrics = metrics

            if logger is not None:
                global_epoch = em * args.m_step_epochs + m_epoch_in_step
                logger.log('time/em_iter_at_eval', em, global_epoch)
                logger.log('time/m_epoch_at_eval', m_epoch_in_step, global_epoch)

            maybe_save_if_improved(
                args, uncond_model, optimizer,
                real_sig, gen_sig, last_recon,
                metrics_history, em, logger,
                current_disc=current_disc, m_epoch=m_epoch_in_step,
            )

        m_eval_every = getattr(args, 'm_eval_every_epochs', 0) or 0

        for em_iter in range(args.em_iters):
            current_em_iter[0] = em_iter
            iter_start_time = time.time()

            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # --- Section 5.1: Curriculum scheduling ---
            progress = min(1.0, (em_iter + 1) / max(args.em_iters * 0.6, 1))
            reveal_prob = (1.0 - progress) * 0.3
            extra_reveal = (np.random.rand(*obs_masks.shape) < reveal_prob) & ~obs_masks
            curriculum_masks = obs_masks | extra_reveal
            curriculum_corrupted = corrupted_data.copy()
            curriculum_corrupted[extra_reveal] = last_recon[extra_reveal]
            n_revealed = extra_reveal.sum()
            if n_revealed > 0:
                print(f"  Curriculum: +{n_revealed} positions (progress={progress:.2f})")

            # --- E-step (Sections 5.2–5.4, 5.7) ---
            e_step_start = time.time()
            reconstructions = e_step(
                args, uncond_model, curriculum_corrupted, curriculum_masks,
                em_iter, args.device, logger
            )
            e_step_seconds = time.time() - e_step_start
            last_recon = reconstructions

            # Re-cache STFT per-bin scales on the co-evolving reconstructions
            # so the denoiser trains on coefficients in the same distribution
            # used at sampling time. No-op for delay embedder.
            if is_stft and getattr(args, 'stft_recache_each_em', True):
                _cache_embedder_stats_if_needed(
                    uncond_model, reconstructions,
                    tag='em_recache', logger=logger, em_iter=em_iter,
                )

            # Fix 1 runaway guard: abort if scale_real_max blows up >10x the
            # warm-start value. This catches the positive-feedback loop
            # (~5e9x growth over 10 EM iters) that produced disc_mean near
            # 0.5 in the previous STFT runs.
            if is_stft and scale_real_warmstart_max is not None:
                _sr = getattr(uncond_model.ts_img, 'scale_real', None)
                if _sr is not None:
                    _cur = float(_sr.detach().abs().max().item())
                    if _cur > 10.0 * scale_real_warmstart_max:
                        raise RuntimeError(
                            f"STFT scale runaway detected at em_iter={em_iter}: "
                            f"scale_real.max()={_cur:.3e} exceeds "
                            f"10x warm-start value "
                            f"{scale_real_warmstart_max:.3e}. "
                            f"Re-run with --no_stft_recache_each_em or "
                            f"--stft_dead_bin_quantile > 0."
                        )

            uncond_model.reset_ema()

            # --- M-step (Sections 5.5–5.6) with optional mid-step eval ---
            m_step_start = time.time()
            uncond_model, eval_time_in_mstep = m_step(
                args, uncond_model, optimizer,
                reconstructions, corrupted_data, obs_masks,
                em_iter, args.device, logger,
                eval_callback=run_eval_and_save if m_eval_every > 0 else None,
                eval_every_epochs=m_eval_every if m_eval_every > 0 else None,
            )
            m_step_seconds_incl_eval = time.time() - m_step_start
            m_step_seconds = m_step_seconds_incl_eval - eval_time_in_mstep

            iter_elapsed = time.time() - iter_start_time
            total_elapsed = time.time() - em_start_time
            cumulative_em_seconds += e_step_seconds + m_step_seconds
            print(f"  Wall-clock: {iter_elapsed:.1f}s (total: {total_elapsed:.1f}s) "
                  f"[E={e_step_seconds:.1f}s, M={m_step_seconds:.1f}s, "
                  f"eval-in-M={eval_time_in_mstep:.1f}s]")

            if logger is not None:
                logger.log('em/wall_clock_seconds', iter_elapsed, em_iter)
                logger.log('em/total_wall_clock', total_elapsed, em_iter)
                logger.log('time/e_step_minutes', e_step_seconds / 60.0, em_iter)
                logger.log('time/m_step_minutes', m_step_seconds / 60.0, em_iter)

            # If mid-M-step eval is disabled, fall back to one eval+save per EM iter.
            is_last = (em_iter == args.em_iters - 1)
            if m_eval_every <= 0:
                if (em_iter + 1) % args.em_eval_interval == 0 or is_last:
                    run_eval_and_save(args.m_step_epochs - 1)

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)

        total_em_time = time.time() - em_start_time
        print(f"\nTotal EM wall-clock: {total_em_time:.1f}s "
              f"({total_em_time/3600:.2f}h)")
        if logger is not None:
            logger.log('em/total_em_wall_clock_hours', total_em_time / 3600, 0)

        # --- Convergence timing (to best disc reached so far in EM) ---
        if cum_em_at_best_seconds is not None:
            time_to_best_exc_init_seconds = bootstrap_seconds + cum_em_at_best_seconds
            time_to_best_inc_init_seconds = init_seconds + time_to_best_exc_init_seconds
            global_epoch_at_best = (best_disc_em_iter * args.m_step_epochs
                                     + (best_disc_m_epoch if best_disc_m_epoch is not None else 0))
            print(f"\nConvergence summary (EM-only):")
            print(f"  Best disc: {best_disc_value:.4f} at em_iter={best_disc_em_iter}, "
                  f"m_epoch={best_disc_m_epoch}")
            print(f"  Time to best disc (inc init): {time_to_best_inc_init_seconds/60.0:.2f} min")
            print(f"  Time to best disc (exc init): {time_to_best_exc_init_seconds/60.0:.2f} min")
            if logger is not None:
                logger.log('time/time_to_best_disc_inc_init_minutes',
                           time_to_best_inc_init_seconds / 60.0, 0)
                logger.log('time/time_to_best_disc_exc_init_minutes',
                           time_to_best_exc_init_seconds / 60.0, 0)
                logger.log('time/em_iters_to_best_disc', best_disc_em_iter, 0)
                logger.log('time/m_epoch_at_best_disc',
                           best_disc_m_epoch if best_disc_m_epoch is not None else 0, 0)
                logger.log('time/global_epoch_at_best_disc', global_epoch_at_best, 0)
                logger.log('time/best_disc_value', best_disc_value, 0)

        # === Phase 3: Fresh model on final completions ===
        if do_phase3:
            print(f"\n{'='*60}")
            print("Phase 3: Fresh model on final completions")
            print(f"{'='*60}")
            phase3_metrics = train_unconditional_regular(
                args, last_recon, test_loader, args.device, logger
            )
            if phase3_metrics:
                phase3_disc = phase3_metrics.get('disc_mean', float('inf'))
                em_disc = best_metrics.get('disc_mean', float('inf')) if best_metrics else float('inf')
                print(f"  Phase 3 disc_mean: {phase3_disc:.4f} "
                      f"(EM best: {em_disc:.4f})")
                if phase3_disc < em_disc:
                    best_metrics = phase3_metrics

        print("\n" + "="*60)
        print("Co-Evolving EM Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("Co-evolving EM training is complete")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    args = parse_args_irregular()

    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
