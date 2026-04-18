"""
Cross-Space Consistency Framework for EM in Lifted Representations

EM with diffusion priors in a lifted representation (delay embedding) creates a
dual-space system: observations in TS space, model in image space. Without
explicit consistency enforcement, errors compound across EM iterations.

This module implements three cross-space consistency constraints:

  Constraint 1 — Structural Consistency (initialization & scheduling):
    Domain-informed warm start (Kalman/STL) + curriculum scheduling.
    Ensures TS-space structure is present from the start of the image-space loop.

  Constraint 2 — Representation Consistency (E-step):
    Noise-adaptive CG regularisation (Prop 2: σ_y = c·σ_t gives bounded κ)
    + manifold projection (Prop 1: orthogonal projection onto Range(ts_to_img)).
    Ensures completions live on the valid delay-embedding subspace.

  Constraint 3 — Observation Consistency (M-step):
    SNR-gated L_obs (Prop 3: minimal tractable alternative to intractable
    importance-weighted / variational EM).  Bridges image-space denoiser
    to TS-space observations; vanishes at convergence.
"""

import torch
import torch.autograd
import torch.multiprocessing
from torch import optim
import torch.nn.functional as F
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
from utils.utils_stl import initialize_with_iterative_stl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


# =============================================================================
# Conjugate Gradient Solver
# =============================================================================

def conjugate_gradient(A_fn, b, maxiter=1):
    """
    Conjugate gradient solver for linear system A·x = b, where A is given
    as a function (linear operator).

    Args:
        A_fn: callable, linear operator v -> A·v
        b: right-hand side tensor
        maxiter: number of CG iterations (default=1)

    Returns:
        x: approximate solution tensor (same shape as b)
    """
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
# Consistency-Constrained MMPS Sampler (Constraint 2: Representation Consistency)
# =============================================================================

class ConsistencyConstrainedMMPS:
    """
    MMPS sampler enforcing Constraint 2 (Representation Consistency).

    The delay embedding maps N_ts time points into N_img > N_ts image pixels.
    Valid images form a linear subspace of R^{N_img}.  Standard MMPS is unaware
    of this subspace, so two mechanisms are needed:

      (a) Noise-adaptive CG: σ_y(σ_t) = c · σ_t gives condition number
          κ = (c² + λ_max)/(c² + λ_min), bounded independently of σ_t
          (Proposition 2).

      (b) Manifold projection: Π = ts_to_img ∘ img_to_ts is the orthogonal
          projection onto Range(ts_to_img), applied to the final sample
          (Proposition 1).  Hard observation enforcement follows.

    Constraint 3 (Observation Consistency / L_obs) is in the m_step function.
    """

    def __init__(self, args, diffusion_fn, shape, sigma_y_ratio=0.1, cg_iters=5,
                 ts_to_img_fn=None, img_to_ts_fn=None,
                 use_adaptive_sigma_y=True, use_consistency_projection=True,
                 sigma_y_floor=0.0):
        self.args = args
        self.device = args.device
        self.shape = shape
        self.net = diffusion_fn.to(device=self.device)

        # EDM parameters
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.num_steps = args.diffusion_steps

        # Constraint 2a: noise-adaptive CG (Prop 2)
        self.sigma_y_ratio = sigma_y_ratio       # c in σ_y = c · σ_t
        self.cg_iters = cg_iters
        self.sigma_y_floor = sigma_y_floor

        # Constraint 2b: manifold projection (Prop 1)
        self.ts_to_img_fn = ts_to_img_fn
        self.img_to_ts_fn = img_to_ts_fn

        # Feature flags (allow ablating individual constraints)
        self.use_adaptive_sigma_y = use_adaptive_sigma_y
        self.use_consistency_projection = use_consistency_projection

    def posterior_denoise(self, x_t, sigma, x_obs, mask):
        """
        MMPS posterior denoiser with noise-adaptive regularisation (Prop 2).

        CG system: (σ_y²I + σ_t² AJAᵀ) v = r.
        With fixed σ_y the condition number κ ~ σ_t²/σ_y² is unbounded.
        Setting σ_y = c·σ_t factors to σ_t²(c²I + AJAᵀ)v = r, so
        κ = (c² + λ_max)/(c² + λ_min) — independent of σ_t.
        """
        sigma_sq = sigma ** 2

        if self.use_adaptive_sigma_y:
            sigma_y = self.sigma_y_ratio * sigma
            if self.sigma_y_floor > 0:
                sigma_y = torch.clamp(sigma_y, min=self.sigma_y_floor)
            sigma_y_sq = sigma_y ** 2
        else:
            sigma_y_sq = self.sigma_y_ratio ** 2

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

        result = denoised.detach() + sigma_sq * score.detach()
        return result

    def _consistency_project(self, x_img, obs_ts, mask_ts):
        """
        Manifold projection + hard observation enforcement (Prop 1).

        Π = ts_to_img ∘ img_to_ts is the orthogonal projection onto
        Range(ts_to_img) — the minimum-norm solution to the representation
        consistency constraint x ∈ Range(ts_to_img).  After projection,
        observed time points are hard-enforced (noiseless case: y = Ax).
        """
        x_ts = self.img_to_ts_fn(x_img.float())
        x_ts_enforced = x_ts * (1.0 - mask_ts) + obs_ts * mask_ts
        return self.ts_to_img_fn(x_ts_enforced).to(torch.float64)

    def mmps_sample(self, latents, x_obs_img, mask_img, obs_ts=None, mask_ts=None):
        """
        MMPS sampling with representation consistency applied to the final output.

        During the reverse process, the posterior denoiser enforces observations
        softly via CG (Prop 2: bounded condition number).  Manifold projection
        (Prop 1: orthogonal, optimal) + hard observation enforcement are applied
        once at the end — projecting at every step destroys the noise schedule
        because img_to_ts averages overlapping pixels, reducing noise variance.

        Args:
            latents: initial noise (batch, C, H, W)
            x_obs_img: observed data in image space (batch, C, H, W)
            mask_img: binary mask in image space (batch, 1, H, W)
            obs_ts: observed time series (batch, seq_len, features)
            mask_ts: observation mask (batch, seq_len, features)
        """
        do_project = (self.use_consistency_projection
                      and self.ts_to_img_fn is not None
                      and self.img_to_ts_fn is not None
                      and obs_ts is not None
                      and mask_ts is not None)

        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_obs = x_obs_img.to(torch.float64)
        mask = mask_img.to(torch.float64)

        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self.posterior_denoise(x_hat, t_hat, x_obs, mask)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised_2 = self.posterior_denoise(x_next, t_next, x_obs, mask)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # Project onto delay-embedding manifold and enforce observations
        # only on the final clean sample (not during the noisy reverse process)
        if do_project:
            x_next = self._consistency_project(x_next, obs_ts, mask_ts)

        return x_next

    def sampling_mmps(self, x_obs_img, mask_img, obs_ts=None, mask_ts=None):
        """Convenience wrapper for MMPS sampling."""
        batch_size = x_obs_img.shape[0]
        latents = torch.randn([batch_size, *self.shape], device=self.device)
        return self.mmps_sample(latents, x_obs_img, mask_img, obs_ts=obs_ts, mask_ts=mask_ts)

    @torch.no_grad()
    def sampling(self, sampling_number=16):
        """Standard unconditional sampling (for evaluation)."""
        latents = torch.randn([sampling_number, *self.shape], device=self.device)

        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self.net(x_hat, t_hat, None).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised = self.net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

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


def off_manifold_energy_batch(x_img, img_to_ts_fn, ts_to_img_fn):
    """
    Representation consistency diagnostic: E_off = ||x - Π(x)||².

    Measures distance from the valid delay-embedding manifold Range(ts_to_img).
    Non-zero only for images that violate the representation constraint.
    Must be called on raw MMPS output (before projection).
    """
    with torch.no_grad():
        x_ts_rt = img_to_ts_fn(x_img.float())
        x_img_rt = ts_to_img_fn(x_ts_rt)
        return ((x_img.float() - x_img_rt) ** 2).sum(dim=(1, 2, 3)).mean().item()


# =============================================================================
# Core EM functions
# =============================================================================

def e_step(args, uncond_model, corrupted_data, obs_masks, em_iter, device, logger=None):
    """
    E-step: complete missing values via consistency-constrained MMPS.

    Applies Constraint 2 (Representation Consistency):
      - Noise-adaptive CG (Prop 2) for stable posterior sampling
      - Manifold projection (Prop 1) + hard observation enforcement

    Returns:
        reconstructions: numpy array (N, seq_len, features) — fully completed
    """
    print(f"\n=== E-Step (EM iter {em_iter}) — Representation Consistency ===")

    uncond_model.eval()

    all_reconstructions = []
    target_shape = (args.input_channels, args.img_resolution, args.img_resolution)

    N = len(corrupted_data)
    batch_size = args.batch_size

    sigma_y_ratio = getattr(args, 'sigma_y_ratio', 0.1)
    cg_iters = getattr(args, 'mmps_cg_iters', 5)
    use_adaptive = getattr(args, 'adaptive_sigma_y', True)
    use_proj = getattr(args, 'consistency_projection', True)
    sigma_y_floor = getattr(args, 'sigma_y_floor', 0.0)

    total_off_energy = 0.0
    n_off_samples = 0

    with uncond_model.ema_scope():
        process = ConsistencyConstrainedMMPS(
            args, uncond_model.net, target_shape,
            sigma_y_ratio=sigma_y_ratio,
            cg_iters=cg_iters,
            ts_to_img_fn=uncond_model.ts_to_img,
            img_to_ts_fn=uncond_model.img_to_ts,
            use_adaptive_sigma_y=use_adaptive,
            use_consistency_projection=use_proj,
            sigma_y_floor=sigma_y_floor,
        )

        for start_idx in tqdm(range(0, N, batch_size), desc="E-step MMPS"):
            end_idx = min(start_idx + batch_size, N)
            cur_batch_size = end_idx - start_idx

            corrupted_batch = corrupted_data[start_idx:end_idx]
            mask_batch = obs_masks[start_idx:end_idx]

            corrupted_ts = torch.tensor(corrupted_batch, dtype=torch.float32, device=device)
            mask_ts = torch.tensor(mask_batch, dtype=torch.float32, device=device)

            obs_ts = torch.nan_to_num(corrupted_ts, nan=0.0)

            x_obs_img = uncond_model.ts_to_img(obs_ts)

            mask_ts_expanded = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])
            mask_img = uncond_model.ts_to_img(mask_ts_expanded)
            mask_img = mask_img[:, :1, :, :]

            mask_ts_proj = mask_ts.unsqueeze(-1).expand(-1, -1, corrupted_ts.shape[-1])

            x_img_imputed = process.sampling_mmps(
                x_obs_img, mask_img,
                obs_ts=obs_ts, mask_ts=mask_ts_proj
            )

            # Representation consistency diagnostic (before projection)
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
    print(f"E-step complete. Imputed {len(reconstructions)} sequences.")
    print(f"  Off-manifold energy: {off_energy:.6f}")

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


def m_step(args, uncond_model, optimizer, reconstructions,
           corrupted_data, obs_masks, em_iter, device, logger=None):
    """
    M-step with Constraint 3 (Observation Consistency).

    L = L_SM + λ_obs · L_obs

    L_SM:  standard denoising score matching in image space
    L_obs: SNR-gated observation grounding in TS space (Prop 3).
           Bridges image ↔ TS space; vanishes at convergence.
           The minimal tractable alternative — importance-weighted EM and
           variational EM are structurally intractable for diffusion posteriors.

    Args:
        reconstructions: completed sequences from E-step (N, seq_len, features)
        corrupted_data: original corrupted data with NaN (N, seq_len, features)
        obs_masks: observation masks (N, seq_len), True=observed
    """
    print(f"\n=== M-Step (EM iter {em_iter}) — Observation Consistency ===")
    print(f"Training for {args.m_step_epochs} epochs...")

    lambda_obs = getattr(args, 'lambda_obs', 0.5)
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

    for epoch in range(args.m_step_epochs):
        epoch_loss_sm = 0.0
        epoch_loss_obs = 0.0
        num_batches = 0

        for batch_idx, (x_clean, y_ts_batch, mask_ts_batch) in enumerate(recon_loader):
            x_clean = x_clean.to(device)
            y_ts_batch = y_ts_batch.to(device)
            mask_ts_batch = mask_ts_batch.to(device)

            x_img = uncond_model.ts_to_img(x_clean)

            # Sample sigma and compute denoised output inline
            # (need access to sigma for SNR gate)
            rnd_normal = torch.randn([x_img.shape[0], 1, 1, 1], device=device)
            sigma = (rnd_normal * uncond_model.P_std + uncond_model.P_mean).exp()
            weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
            noise = torch.randn_like(x_img) * sigma
            denoised = uncond_model.net(x_img + noise, sigma, None)

            # --- L_SM: score matching in image space ---
            x_unpad = uncond_model.unpad(x_img, x_img.shape)
            denoised_unpad = uncond_model.unpad(denoised, x_img.shape)
            loss_sm = (weight * (denoised_unpad - x_unpad).square()).mean()

            # --- L_obs: observation consistency across dual-space boundary (Prop 3) ---
            if lambda_obs > 0:
                if use_snr_gate:
                    snr_gate = sigma_d_for_gate ** 2 / (sigma.squeeze() ** 2 + sigma_d_for_gate ** 2)
                    if use_sm_weight_for_lobs:
                        snr_gate = snr_gate * weight.squeeze()
                else:
                    snr_gate = torch.ones(x_img.shape[0], device=device)

                denoised_ts = uncond_model.img_to_ts(denoised)
                obs_residual = mask_ts_batch.unsqueeze(-1) * (denoised_ts - y_ts_batch)
                loss_obs_per_sample = (obs_residual ** 2).mean(dim=(-1, -2))
                loss_obs = (snr_gate * loss_obs_per_sample).mean()
            else:
                loss_obs = torch.tensor(0.0, device=device)

            loss = loss_sm + lambda_obs * loss_obs

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncond_model.parameters(), 1.0)
            optimizer.step()
            uncond_model.on_train_batch_end()

            epoch_loss_sm += loss_sm.item()
            epoch_loss_obs += loss_obs.item()
            num_batches += 1
            torch.cuda.empty_cache()

        avg_sm = epoch_loss_sm / num_batches
        avg_obs = epoch_loss_obs / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.m_step_epochs}  L_SM={avg_sm:.4f}  "
                  f"L_obs={avg_obs:.4f}  (λ={lambda_obs})")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/m_step_loss_sm', avg_sm, global_step)
            logger.log('em/m_step_loss_obs', avg_obs, global_step)
            logger.log('em/m_step_loss_total', avg_sm + lambda_obs * avg_obs, global_step)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


def m_step_plain(args, uncond_model, optimizer, reconstructions, em_iter, device, logger=None):
    """Plain M-step (L_SM only, no observation consistency). Used for warm-start bootstrap."""
    print(f"\n=== M-Step Plain (EM iter {em_iter}) — L_SM only ===")
    print(f"Training unconditional model for {args.m_step_epochs} epochs...")

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
            print(f"  Epoch {epoch+1}/{args.m_step_epochs}, loss: {avg_loss:.4f}")

        if logger is not None:
            global_step = em_iter * args.m_step_epochs + epoch
            logger.log('em/m_step_loss', avg_loss, global_step)

    print(f"M-step complete for EM iteration {em_iter}.")
    return uncond_model


def evaluate_uncond(args, uncond_model, test_loader, em_iter, device,
                    logger=None, calc_other_metrics=False):
    """Evaluate via unconditional sampling (same as run_irregular.py)."""
    print(f"\n=== Evaluation (EM iter {em_iter}) ===")

    uncond_model.eval()
    gen_sig = []
    real_sig = []

    with torch.no_grad():
        with uncond_model.ema_scope():
            process = DiffusionProcess(
                args, uncond_model.net,
                (args.input_channels, args.img_resolution, args.img_resolution)
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

    print(f"EM iter {em_iter} metrics:")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
        if logger is not None:
            logger.log(f'test/{key}', value, em_iter)

    mem_plot_path = f"memorization_hist_em_iter_{em_iter}.png"
    mem_stats = compute_memorization_metric(
        real_data=real_sig,
        generated_data=gen_sig,
        device=device,
        plot_path=mem_plot_path
    )

    print(f"EM iter {em_iter} memorization metrics:")
    for k, v in mem_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if logger is not None:
            logger.log(f'test/memorization/{k}', v, em_iter)

    if logger is not None:
        upload_successful = False
        try:
            logger.log_file('test/memorization/histogram', mem_plot_path, em_iter)
            upload_successful = True
        except Exception as e:
            print(f"Failed to upload memorization plot: {e}")

        if upload_successful:
            try:
                if os.path.exists(mem_plot_path):
                    os.remove(mem_plot_path)
            except Exception:
                pass

    return scores


# =============================================================================
# Main: Co-Evolving EM with Cross-Space Consistency
# =============================================================================

def main(args):
    """Cross-space consistency framework for EM in delay-embedded time series."""

    name = create_model_name_and_dir(args)
    name = f"diffem_consistency_{name}"

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

        # === Extract corrupted data and masks ===
        print("Extracting corrupted data from train loader...")
        corrupted_data, obs_masks = get_corrupted_data_from_loader(train_loader, args.device)
        print(f"Extracted {len(corrupted_data)} sequences with "
              f"{obs_masks.sum() / obs_masks.size * 100:.1f}% observed")

        # === Constraint 1a: Structural Consistency — Domain-Informed Warm Start ===
        print(f"\n{'='*60}")
        print("Constraint 1 (Structural): Domain-Informed Warm Start")
        print(f"{'='*60}")

        initial_reconstructions = initialize_with_iterative_stl(
            corrupted_data, obs_masks, seed=args.seed
        )

        # === Constraint 1a (cont): Bootstrap M-step on warm-start completions ===
        print(f"\n{'='*60}")
        print("Constraint 1 (Structural): Bootstrap M-Step on Warm-Start Data")
        print(f"{'='*60}")

        uncond_model = m_step_plain(
            args, uncond_model, optimizer,
            initial_reconstructions, em_iter=-1, device=args.device, logger=logger
        )

        # === Co-Evolving EM with Cross-Space Consistency ===
        sigma_y_ratio = getattr(args, 'sigma_y_ratio', 0.1)
        lambda_obs = getattr(args, 'lambda_obs', 0.5)
        use_adaptive = getattr(args, 'adaptive_sigma_y', True)
        use_proj = getattr(args, 'consistency_projection', True)
        sigma_y_floor = getattr(args, 'sigma_y_floor', 0.0)
        use_snr_gate = not getattr(args, 'no_snr_gate', False)
        use_sm_weight_lobs = getattr(args, 'lobs_use_sm_weight', False)
        snr_sigma_d = getattr(args, 'snr_gate_sigma_d', None)
        do_phase3 = getattr(args, 'phase3', True)
        eval_all = getattr(args, 'eval_all_metrics', False)

        print(f"\n{'='*60}")
        print(f"Co-Evolving EM — {args.em_iters} iterations, 3 consistency constraints")
        print(f"  [C1] Structural:      warm_start=STL, curriculum=annealed")
        print(f"  [C2] Representation:  adaptive_σy={use_adaptive} (c={sigma_y_ratio}), "
              f"CG_iters={args.mmps_cg_iters}, projection={use_proj}")
        print(f"       σ_y_floor:       {sigma_y_floor}")
        print(f"  [C3] Observation:     λ_obs={lambda_obs}, snr_gate={use_snr_gate}")
        print(f"       lobs_sm_weight:  {use_sm_weight_lobs}, "
              f"σ_d={snr_sigma_d or 'model default'}")
        print(f"  Phase 3 (fresh):      {do_phase3}")
        print(f"  Eval all metrics:     {eval_all}")
        print(f"{'='*60}")

        best_metrics = None
        last_recon = initial_reconstructions
        em_start_time = time.time()

        for em_iter in range(args.em_iters):
            iter_start_time = time.time()

            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{args.em_iters}")
            print(f"{'='*60}")

            # --- Constraint 1b (Structural): curriculum scheduling ---
            progress = min(1.0, (em_iter + 1) / max(args.em_iters * 0.6, 1))
            reveal_prob = (1.0 - progress) * 0.3
            extra_reveal = (np.random.rand(*obs_masks.shape) < reveal_prob) & ~obs_masks
            curriculum_masks = obs_masks | extra_reveal
            curriculum_corrupted = corrupted_data.copy()
            curriculum_corrupted[extra_reveal] = last_recon[extra_reveal]
            n_revealed = extra_reveal.sum()
            if n_revealed > 0:
                print(f"  Curriculum: revealed {n_revealed} extra positions "
                      f"(progress={progress:.2f})")

            reconstructions = e_step(
                args, uncond_model, curriculum_corrupted, curriculum_masks,
                em_iter, args.device, logger
            )
            last_recon = reconstructions

            uncond_model.reset_ema()

            uncond_model = m_step(
                args, uncond_model, optimizer,
                reconstructions, corrupted_data, obs_masks,
                em_iter, args.device, logger
            )

            iter_elapsed = time.time() - iter_start_time
            total_elapsed = time.time() - em_start_time
            print(f"  EM iter {em_iter} wall-clock: {iter_elapsed:.1f}s "
                  f"(total: {total_elapsed:.1f}s)")

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

        # === Phase 3: Fresh model on consistency-cleaned data ===
        if do_phase3:
            print(f"\n{'='*60}")
            print("Phase 3: Fresh model on consistency-cleaned data")
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
        print("Cross-Space Consistency EM Complete!")
        print("="*60)
        if best_metrics:
            print("Best metrics:")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")

        logging.info("Cross-space consistency EM training is complete")


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
