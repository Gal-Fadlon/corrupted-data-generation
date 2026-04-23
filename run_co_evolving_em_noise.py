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
from metrics.memorization import compute_memorization_metric
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
                 sigma_y_phys=0.0, proj_prior_std=1.0,
                 enforce_observations=True):
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
        # Physical observation noise std (Gaussian noise corruption: y = x + ε,
        # ε ~ N(0, σ_phys² I)). For missing-data corruption σ_phys = 0 and we
        # recover the original σ_y² = (c·σ_t)² formula. For noisy observations
        # we use σ_y² = σ_phys² + (c·σ_t)² so CG's implicit likelihood matches
        # the true physical noise and the posterior does not over-trust y.
        self.sigma_y_phys = float(sigma_y_phys)

        # Section 5.4 (extended for noise): Bayesian blend between observed y
        # and posterior sample for observation enforcement. For missing data
        # (hard enforcement) set enforce_observations=True, proj_prior_std is
        # unused. For noisy y, proj_prior_std controls the prior std used in
        # trust = σ_prior² / (σ_prior² + σ_phys²).
        self.proj_prior_std = float(proj_prior_std)
        self.enforce_observations = bool(enforce_observations)

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

    def _get_sigma_y_sq(self, sigma):
        """Section 5.3: σ_y²(σ_t) = σ_phys² + (c · σ_t)².

        For σ_phys = 0 (missing-data corruption) this collapses to (c·σ_t)²,
        which is the original adaptive σ_y from Prop 2: CG's condition number
        stays bounded independently of σ_t.  For σ_phys > 0 (Gaussian-noise
        corruption) we add the physical noise variance, so the observation
        likelihood CG implicitly solves matches the true generative model
        y = x + ε, ε ~ N(0, σ_phys² I).  The (c·σ_t)² term acts as a Tikhonov
        regulariser keeping κ bounded at late diffusion steps.
        """
        phys_sq = self.sigma_y_phys ** 2
        if self.use_adaptive_sigma_y:
            sigma_y = self.sigma_y_ratio * sigma
            if self.sigma_y_floor > 0:
                sigma_y = torch.clamp(sigma_y, min=self.sigma_y_floor)
            return phys_sq + sigma_y ** 2
        return phys_sq + self.sigma_y_ratio ** 2

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
        # is D + sigma_t^2 J^T G^T v_ts, so we reuse the same exact G^T.
        v_img = gt_fn((mask_ts * v_ts).float()).to(torch.float64)
        score_img = vjp_fn(v_img)

        return denoised_img.detach() + sigma_sq * score_img.detach()

    def _apply_manifold_projection(self, x_img):
        """
        Section 5.4 (Prop 1): orthogonal projection onto Range(ts_to_img).

        Π = ts_to_img ∘ img_to_ts.  Applied to the final reverse-diffusion
        iterate regardless of corruption type — it only depends on the lift,
        not on y.  This keeps §5.4 Prop 1 valid verbatim for noisy y.
        """
        x_ts = self.img_to_ts_fn(x_img.float())
        return self.ts_to_img_fn(x_ts).to(torch.float64)

    def _enforce_observations(self, x_img, obs_ts, mask_ts):
        """
        Corruption-aware observation enforcement in TS space.

        Missing data (σ_phys = 0):  hard enforcement x_ts[mask] ← y_ts[mask]
        — original behavior, consistent with y = A_ts x (noiseless).

        Gaussian noise (σ_phys > 0): Bayesian posterior blend
            trust = σ_prior² / (σ_prior² + σ_phys²),
            x_ts  ← trust · y_ts + (1 - trust) · x_ts      (on mask)
        so we do NOT inject the physical noise back into the reconstruction.
        As σ_phys → 0 trust → 1 and we recover hard enforcement; as σ_phys
        grows we trust the prior more.
        """
        x_ts = self.img_to_ts_fn(x_img.float())
        mask_f = mask_ts.float()
        obs_f = obs_ts.float()

        if self.sigma_y_phys <= 0.0:
            x_ts_out = x_ts * (1.0 - mask_f) + obs_f * mask_f
        else:
            prior_var = max(self.proj_prior_std, 1e-8) ** 2
            phys_var = self.sigma_y_phys ** 2
            trust = prior_var / (prior_var + phys_var)
            blended = trust * obs_f + (1.0 - trust) * x_ts
            x_ts_out = x_ts * (1.0 - mask_f) + blended * mask_f

        return self.ts_to_img_fn(x_ts_out).to(torch.float64)

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

            if can_use_obs_space:
                denoised = self.posterior_denoise_obs_space(x_hat, t_hat, obs_ts_d, mask_ts_d)
            else:
                denoised = self.posterior_denoise_image_space(x_hat, t_hat, x_obs, mask)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if pad_mask_d is not None:
                x_next = x_next * pad_mask_d

            if i < self.num_steps - 1:
                if can_use_obs_space:
                    denoised_2 = self.posterior_denoise_obs_space(x_next, t_next, obs_ts_d, mask_ts_d)
                else:
                    denoised_2 = self.posterior_denoise_image_space(x_next, t_next, x_obs, mask)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                if pad_mask_d is not None:
                    x_next = x_next * pad_mask_d

        if do_project:
            # Section 5.4 (Prop 1): manifold projection is unconditional,
            # depends only on the lift.  Observation enforcement is applied
            # afterwards and is corruption-aware (hard for missing, Bayesian
            # blend for Gaussian noise).
            x_next = self._apply_manifold_projection(x_next)
            if pad_mask_d is not None:
                x_next = x_next * pad_mask_d

            if self.enforce_observations:
                x_next = self._enforce_observations(x_next, obs_ts_d, mask_ts_d)
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


def get_noisy_data_from_loader(clean_loader, sigma_phys, seed):
    """
    Extract clean time series from a loader and add synthetic Gaussian noise.

    Used for the Gaussian-noise corruption arm (y = x + ε, ε ~ N(0, σ_phys² I)).
    `clean_loader` is expected to yield clean sequences (the project's
    `test_loader` uses `ori_data` — the pre-corruption signal — which is the
    correct source of ground truth for every script in this repo).

    Returns:
        noisy_data : np.ndarray [N, T, F]    clean + Gaussian noise
        obs_masks  : np.ndarray [N, T] bool  all True (every timestep observed)
    """
    all_clean = []
    for data in clean_loader:
        x_ts = data[0]
        all_clean.append(x_ts.numpy())

    clean_data = np.vstack(all_clean).astype(np.float32)
    if clean_data.ndim != 3:
        raise ValueError(
            f"get_noisy_data_from_loader expects [N, T, F] sequences; got "
            f"shape {clean_data.shape}. Make sure the loader yields clean "
            f"ori_data (test_loader), not the irregular train_loader which "
            f"has a time-index column appended."
        )
    rng = np.random.RandomState(int(seed))
    noise = rng.normal(0.0, float(sigma_phys), size=clean_data.shape).astype(np.float32)
    noisy_data = clean_data + noise

    obs_masks = np.ones(clean_data.shape[:2], dtype=bool)

    return noisy_data, obs_masks


# --- Noise-specific warm-start initializers (Section 5.1 for Gaussian noise) ---

def _stl_denoise_init(noisy_data, seed=0, period=None):
    """STL-residual denoising: treat y as a clean signal, decompose into
    trend + seasonal + residual, and return trend + seasonal as the denoised
    warm start.  Robust to moderate σ_phys (0.05–0.2) because STL rejects
    high-frequency residuals.
    """
    from statsmodels.tsa.seasonal import STL

    N, T, F = noisy_data.shape
    out = noisy_data.copy().astype(np.float32)

    if period is None:
        period = max(3, min(24, T // 4))
    if period % 2 == 0:
        period = period + 1

    for i in range(N):
        for c in range(F):
            series = noisy_data[i, :, c].astype(np.float64)
            try:
                stl = STL(series, period=period, robust=True).fit()
                out[i, :, c] = (stl.trend + stl.seasonal).astype(np.float32)
            except Exception:
                out[i, :, c] = series.astype(np.float32)

    return np.clip(out, 0.0, 1.0)


def _ma_smooth_init(noisy_data, window=5):
    """Moving-average smoothing warm start.  Cheaper than STL; useful as a
    bootstrap when STL is too slow.  Clipped to [0,1] to match the data scale.
    """
    N, T, F = noisy_data.shape
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    out = np.empty_like(noisy_data, dtype=np.float32)
    for i in range(N):
        for c in range(F):
            series = noisy_data[i, :, c]
            padded = np.pad(series, (pad, pad), mode='edge')
            kernel = np.ones(w, dtype=np.float32) / float(w)
            out[i, :, c] = np.convolve(padded, kernel, mode='valid').astype(np.float32)
    return np.clip(out, 0.0, 1.0)


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

    corruption_type = getattr(args, 'corruption_type', 'missing')
    sigma_y_phys = float(getattr(args, 'sigma_y_phys', 0.0))
    proj_prior_std = float(getattr(args, 'proj_prior_std', 1.0))
    # Observation enforcement is always on when consistency projection is on;
    # `_enforce_observations` internally dispatches hard (σ_phys=0) vs
    # Bayesian blend (σ_phys>0).  Expose a flag only to allow an ablation
    # that drops observation enforcement entirely (manifold projection alone).
    enforce_obs = not getattr(args, 'no_enforce_observations', False)

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

    # Exact transpose of ts_to_img (autograd VJP).  For delay this is
    # proportional to img_to_ts up to an absorbed scalar (delay keeps the
    # historical behavior via img_to_ts_adjoint_fn=None); for STFT this is
    # the only correct G^T that matches the MMPS-derived posterior.  Note
    # this is img_to_ts_adjoint (= L^{-T}), NOT ts_to_img_adjoint (= L^T).
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
            sigma_y_phys=sigma_y_phys,
            proj_prior_std=proj_prior_std,
            enforce_observations=enforce_obs,
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

    if corruption_type == 'gaussian_noise':
        cache_tag = f'noise_{int(sigma_y_phys * 100):03d}'
    else:
        cache_tag = f'missing_{int(args.missing_rate * 100)}'
    cache_dir = os.path.join(args.recon_cache_dir, args.dataset, cache_tag)
    save_reconstructions(reconstructions, cache_dir, em_iter)

    if logger is not None:
        logger.log('em/e_step_samples', len(reconstructions), em_iter)

    return reconstructions


# =============================================================================
# M-step (Sections 5.5, 5.6)
# =============================================================================

def m_step(args, uncond_model, optimizer, reconstructions,
           corrupted_data, obs_masks, em_iter, device, logger=None):
    """
    M-step: L = L_SM + λ_obs · L_obs + λ_rep · L_rep

    L_SM  — standard denoising score matching in image space
    L_obs — Section 5.6: SNR-gated observation grounding in TS space (Prop 3)
    L_rep — Section 5.5: SNR-gated manifold penalty in image space (Prop 1 at training time)
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

    corruption_type = getattr(args, 'corruption_type', 'missing')
    lobs_noise_mode = getattr(args, 'lobs_noise_mode', 'off')
    sigma_y_phys = float(getattr(args, 'sigma_y_phys', 0.0))

    # For Gaussian-noise corruption, `y = x + ε`, so training the denoiser to
    # minimise ||D_θ·ts - y||² on observed positions teaches it to reproduce
    # the noise.  We disable L_obs unless the user explicitly asks for the
    # MLE-weighted variant (Section 5.6 extension: divide the residual by
    # 2 σ_phys², giving the Gaussian-likelihood weighting of Prop 3 with the
    # correct noise scale).
    if corruption_type == 'gaussian_noise':
        if lobs_noise_mode == 'off':
            lambda_obs = 0.0

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
                if (corruption_type == 'gaussian_noise'
                        and lobs_noise_mode == 'gaussian_mle'
                        and sigma_y_phys > 0):
                    # Gaussian-MLE weighting: -log p(y | x) = (y-x)² / (2 σ²) + const.
                    # The SNR gate still down-weights the high-noise diffusion
                    # steps where (y - D_θ·ts) is dominated by diffusion noise.
                    loss_obs_per_sample = loss_obs_per_sample / (2.0 * sigma_y_phys ** 2)
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

    print(f"  M-step complete for EM iteration {em_iter}.")
    return uncond_model


def m_step_bootstrap(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """L_SM-only M-step for the warm-start bootstrap (Section 5.1)."""
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
                    logger=None, calc_other_metrics=False):
    """Evaluate via unconditional sampling."""
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

    print(f"  EM iter {em_iter} metrics:")
    for key, value in scores.items():
        print(f"    {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, em_iter)

    mem_plot_path = f"memorization_hist_em_iter_{em_iter}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig,
        generated_data=gen_sig,
        device=device,
        plot_path=mem_plot_path
    )

    print(f"  Memorization metrics:")
    for k, v in mem_stats.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        if logger is not None:
            logger.log(f'test/memorization/{k}', v, em_iter)

    if logger is not None:
        upload_successful = False
        try:
            logger.log_file('test/memorization/histogram', mem_plot_path, em_iter)
            upload_successful = True
        except Exception as e:
            print(f"  Failed to upload memorization plot: {e}")

        if upload_successful:
            try:
                if os.path.exists(mem_plot_path):
                    os.remove(mem_plot_path)
            except Exception:
                pass

    return scores


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
    corruption_type = getattr(args, 'corruption_type', 'missing')
    if corruption_type == 'gaussian_noise':
        name = f"co_evolving_em_noise_{name}"
    else:
        name = f"co_evolving_em_{name}"

    logging.info(args)

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
        sigma_y_phys = float(getattr(args, 'sigma_y_phys', 0.0))
        if corruption_type == 'gaussian_noise':
            print(
                f"Extracting clean data from test loader and adding synthetic "
                f"N(0, sigma_phys={sigma_y_phys}) noise..."
            )
            corrupted_data, obs_masks = get_noisy_data_from_loader(
                test_loader, sigma_phys=sigma_y_phys, seed=args.seed,
            )
            print(f"  {len(corrupted_data)} sequences, 100% observed (noisy)")
        else:
            print("Extracting corrupted data from train loader...")
            corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
            print(f"  {len(corrupted_data)} sequences, "
                  f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # === Section 5.1: Warm Start ===
        if corruption_type == 'gaussian_noise':
            noise_init = getattr(args, 'noise_init', 'stl_denoise')
            print(f"\n{'='*60}")
            print(f"Section 5.1: Warm Start (noise_init={noise_init})")
            print(f"{'='*60}")

            if noise_init == 'identity':
                initial_reconstructions = np.clip(corrupted_data, 0.0, 1.0).astype(np.float32)
                print(f"Identity initialization: use noisy y as the warm start "
                      f"({len(corrupted_data)} sequences)")
            elif noise_init == 'ma_smooth':
                initial_reconstructions = _ma_smooth_init(
                    corrupted_data, window=int(getattr(args, 'ma_window', 5))
                )
                print(f"Moving-average smoothing initialization: "
                      f"{len(initial_reconstructions)} sequences")
            else:  # stl_denoise (default)
                initial_reconstructions = _stl_denoise_init(
                    corrupted_data, seed=args.seed,
                    period=getattr(args, 'stl_period', None),
                )
                print(f"STL-residual denoising initialization: "
                      f"{len(initial_reconstructions)} sequences")
        else:
            init_method = getattr(args, 'init_method', 'stl')
            print(f"\n{'='*60}")
            print(f"Section 5.1: Warm Start ({init_method})")
            print(f"{'='*60}")

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

        uncond_model = m_step_bootstrap(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )

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

        noise_init_cfg = getattr(args, 'noise_init', 'stl_denoise')
        proj_prior_std_cfg = float(getattr(args, 'proj_prior_std', 0.3))
        lobs_noise_mode_cfg = getattr(args, 'lobs_noise_mode', 'off')
        no_enforce_obs_cfg = getattr(args, 'no_enforce_observations', False)

        print(f"\n{'='*60}")
        print(f"Co-Evolving EM — {args.em_iters} iterations")
        print(f"  Corruption:            {corruption_type}"
              + (f" (σ_phys={sigma_y_phys})" if corruption_type == 'gaussian_noise' else ""))
        if corruption_type == 'gaussian_noise':
            print(f"  5.1  Warm start:       {noise_init_cfg}")
            print(f"  5.4  Obs. enforce:     "
                  f"{'disabled' if no_enforce_obs_cfg else 'Bayesian blend'}"
                  f" (σ_prior={proj_prior_std_cfg})")
            print(f"  5.6  L_obs mode:       {lobs_noise_mode_cfg}"
                  + (" (λ_obs forced to 0)" if lobs_noise_mode_cfg == 'off' else ""))
        else:
            print(f"  5.1  Warm start:       STL + curriculum")
            print(f"  5.4  Obs. enforce:     hard enforcement (missing-data)")
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
        last_recon = initial_reconstructions
        em_start_time = time.time()

        for em_iter in range(args.em_iters):
            iter_start_time = time.time()

            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # --- Section 5.1: Curriculum scheduling ---
            # Curriculum is a missing-data-specific trick (gradually reveal
            # previously-unobserved positions from `last_recon`).  For
            # Gaussian-noise corruption every position is already observed,
            # so curriculum is a no-op and we pass through obs_masks and the
            # raw noisy observations unchanged.
            if corruption_type == 'gaussian_noise':
                curriculum_masks = obs_masks
                curriculum_corrupted = corrupted_data
            else:
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
            reconstructions = e_step(
                args, uncond_model, curriculum_corrupted, curriculum_masks,
                em_iter, args.device, logger
            )
            last_recon = reconstructions

            # Re-cache STFT per-bin scales on the co-evolving reconstructions
            # so the denoiser trains on coefficients in the same distribution
            # used at sampling time. No-op for delay embedder.
            if is_stft and getattr(args, 'stft_recache_each_em', True):
                _cache_embedder_stats_if_needed(
                    uncond_model, reconstructions,
                    tag='em_recache', logger=logger, em_iter=em_iter,
                )

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

            # --- M-step (Sections 5.5–5.6) ---
            uncond_model = m_step(
                args, uncond_model, optimizer,
                reconstructions, corrupted_data, obs_masks,
                em_iter, args.device, logger
            )

            iter_elapsed = time.time() - iter_start_time
            total_elapsed = time.time() - em_start_time
            print(f"  Wall-clock: {iter_elapsed:.1f}s (total: {total_elapsed:.1f}s)")

            if logger is not None:
                logger.log('em/wall_clock_seconds', iter_elapsed, em_iter)
                logger.log('em/total_wall_clock', total_elapsed, em_iter)

            is_last = (em_iter == args.em_iters - 1)
            if (em_iter + 1) % args.em_eval_interval == 0 or is_last:
                use_all_metrics = eval_all and is_last
                metrics = evaluate_uncond(
                    args, uncond_model, test_loader,
                    em_iter, args.device, logger,
                    calc_other_metrics=use_all_metrics
                )

                if best_metrics is None or metrics.get('disc_mean', float('inf')) < best_metrics.get('disc_mean', float('inf')):
                    best_metrics = metrics

            if logger is not None:
                logger.log('em/iteration', em_iter, em_iter)

        total_em_time = time.time() - em_start_time
        print(f"\nTotal EM wall-clock: {total_em_time:.1f}s "
              f"({total_em_time/3600:.2f}h)")
        if logger is not None:
            logger.log('em/total_em_wall_clock_hours', total_em_time / 3600, 0)

        # === Phase 3: Fresh model on final completions ===
        if do_phase3:
            print(f"\n{'='*60}")
            print("Phase 3: Fresh model on final completions")
            print(f"{'='*60}")
            # Critical for Gaussian-noise corruption: Phase 3 must train on
            # the DENOISED reconstructions produced by the last E-step, NOT
            # on the raw noisy observations `corrupted_data`.  The guard
            # below makes the contract explicit so a future refactor can't
            # silently swap in `corrupted_data` and poison the fresh model.
            assert last_recon is not None, "Phase 3 requires E-step reconstructions"
            if corruption_type == 'gaussian_noise':
                assert last_recon is not corrupted_data, (
                    "Phase 3 must train on denoised E-step reconstructions, "
                    "not the raw noisy observations."
                )
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

def _parse_noise_args():
    """Strip noise-specific CLI args from sys.argv before the shared parser runs.

    Keeps `utils.utils_args.parse_args_irregular` (used by every other script
    in this repo) untouched.  Returns a namespace with the noise flags.
    """
    import argparse as _argparse
    p = _argparse.ArgumentParser(add_help=False)
    p.add_argument(
        '--corruption_type', type=str, default='missing',
        choices=['missing', 'gaussian_noise'],
        help='Corruption model: missing (original) or gaussian_noise (y = x + ε).',
    )
    p.add_argument(
        '--sigma_y_phys', type=float, default=0.0,
        help='Physical Gaussian noise std σ_phys for --corruption_type gaussian_noise.',
    )
    p.add_argument(
        '--noise_init', type=str, default='stl_denoise',
        choices=['identity', 'stl_denoise', 'ma_smooth'],
        help='Warm-start for gaussian_noise corruption.',
    )
    p.add_argument(
        '--proj_prior_std', type=float, default=0.3,
        help='Prior std σ_prior used in the Bayesian observation-enforcement '
             'blend: trust = σ_prior² / (σ_prior² + σ_phys²).',
    )
    p.add_argument(
        '--lobs_noise_mode', type=str, default='off',
        choices=['off', 'gaussian_mle'],
        help='M-step L_obs behavior for gaussian_noise: off disables L_obs '
             '(default — avoids training the denoiser to reproduce noise); '
             'gaussian_mle uses the MLE-weighted residual (y-x)²/(2 σ_phys²).',
    )
    p.add_argument(
        '--no_enforce_observations', action='store_true', default=False,
        help='Ablation: drop observation enforcement at the end of sampling '
             '(keep manifold projection only).',
    )
    p.add_argument(
        '--stl_period', type=int, default=None,
        help='Seasonal period for STL warm start (default: min(24, T//4)).',
    )
    p.add_argument(
        '--ma_window', type=int, default=5,
        help='Window size for moving-average warm start (noise_init=ma_smooth).',
    )
    extra, remainder = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remainder
    return extra


if __name__ == '__main__':
    noise_args = _parse_noise_args()
    args = parse_args_irregular()
    for _k, _v in vars(noise_args).items():
        setattr(args, _k, _v)

    args.use_diffem = True

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
